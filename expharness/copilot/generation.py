import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from verl import DataProto
import requests
import json

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str = None
    topk: int = 3
    include_information: bool = False  # Whether to include search results in feedback
    generator_llm: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    output_context_dir: str = None
    
class LLMGenerationManager:
    """
    Rollout manager for the ExpHarness retrieval copilot.

    The copilot (a small policy LLM trained with PPO) reads the task and emits retrieval
    controls <search>R:W</search> (or <search>none</search> to skip retrieval). The
    experience graph server is queried with the task as the semantic query and (R, W) as
    controls; the retrieved experiences are stored in non_tensor_batch for the reward
    manager, where the frozen executor is run with and without them.
    """
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,  # Worker group for the search copilot (to be trained)
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.timing_raw = {}
        # Load prompt templates for the generator LLM
        self.output_context_dir = config.output_context_dir
        # Initialize tensor helper for handling tensors
        from .tensor_helper import TensorHelper, TensorConfig
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        

    def _load_zeroshot_answers(self, filename):
        """Load zeroshot answers from file."""
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, IOError):
            print(f"Zeroshot answers file {filename} not found.")
            return {}

    def _load_prompt(self, filename):
        """Load prompt template from file."""
        try:
            with open(filename, 'r') as file:
                return file.read().strip()
        except (FileNotFoundError, IOError):
            # Return a default prompt if file not found
            raise ValueError(f"Prompt file {filename} not found.")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str], List[str], List[bool]]:
        """Process responses to extract search queries (supports both <search> and <query> formats)."""
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )

        # Ensure responses end with closing tag if present
        new_responses_str = []
        for resp in responses_str:
            if '</search>' in resp:
                resp = resp.split('</search>')[0] + '</search>'
                if '<search>' not in resp:
                    resp = '<search>\n' + resp
            elif '</query>' in resp:
                resp = resp.split('</query>')[0] + '</query>'
                if '<query>' not in resp:
                    resp = '<query>\n' + resp
            new_responses_str.append(resp)
        responses_str = new_responses_str

        # Extract query information
        queries = []
        search_complete_flags = []

        for resp in responses_str:
            # Try <search>R:query</search> format first
            search_match = re.search(r'<search>(.*?)</search>', resp, re.DOTALL)
            # Fallback to <query> format
            query_match = re.search(r'<query>(.*?)</query>', resp, re.DOTALL)
            search_complete_match = re.search(r'<search_complete>(.*?)</search_complete>', resp, re.DOTALL)

            # Check for search completion flag
            search_complete = False
            if search_complete_match:
                complete_text = search_complete_match.group(1).strip().lower()
                search_complete = complete_text in ("true", "yes", "1", "y")

            # Extract R:W parameters from <search>R:W</search>
            # R and W are retrieval control parameters, NOT the query itself.
            # The actual query is the task description (from self.original_questions).
            # Supported: "70:40", "50.5:30.2" → clamp to [0,100] integers
            # Anything else → R=50, W=50 (default, not R=0 W=0)
            R_val, W_val = 50, 50
            action_is_search = False
            skip_search = False
            if search_match:
                raw_text = search_match.group(1).strip()
                if raw_text.lower() in ('none', 'skip', 'no'):
                    skip_search = True
                else:
                    rw_match = re.match(r'^([\d.]+)\s*:\s*([\d.]+)$', raw_text)
                    if rw_match:
                        try:
                            R_val = max(0, min(int(float(rw_match.group(1))), 100))
                            W_val = max(0, min(int(float(rw_match.group(2))), 100))
                        except (ValueError, OverflowError):
                            R_val, W_val = 50, 50
                    action_is_search = True
            # Store as "R:XX W:XX" prefix — _batch_search will parse these
            # and use original_questions as the actual semantic query
            if skip_search:
                queries.append("SKIP")
            else:
                queries.append(f"R:{R_val} W:{W_val}")

            search_complete_flags.append(search_complete)

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str, queries, search_complete_flags

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> DataProto:
        """
        Run the copilot retrieval loop.
        """
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        # Reset conversation histories and retrieved experiences for this batch
        self.conversation_histories = [""] * len(active_mask)
        self._retrieved_experiences = [""] * len(active_mask)
        
        # Extract the original questions from the initial inputs
        self.original_questions = [""] * len(active_mask)
        initial_inputs_str = self.tokenizer.batch_decode(
            initial_input_ids, 
            skip_special_tokens=True
        )
        
        # Extract questions from the initial inputs
        for i, input_text in enumerate(initial_inputs_str):
            question_matches = re.findall(r'<question>(.*?)</question>', input_text, re.DOTALL)
            if question_matches:
                # Use the last match of <question>...</question>
                self.original_questions[i] = question_matches[-1].strip()
            else:
                print(f"No <question>...</question> tags found in the initial input {input_text}")

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
                
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # Generate with active sequences
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # Process outputs 
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, queries, search_complete_flags = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute search and get feedback from the environment
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            # Update active sequences
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            # Process observations (search results + feedback)
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # Final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # Process outputs
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, queries, search_complete_flags = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute final predictions (without doing search)
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            # Update stats
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            next_obs_ids = self._process_next_obs(next_obs)

            # Update right side
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        
        # Store metadata for reward computation
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        # Store retrieved_node_indices in a separate attribute (not meta_info)
        # so it survives reorder operations. Will be attached to non_tensor_batch after compose.
        per_example_nodes = [[] for _ in range(len(active_mask))]
        if hasattr(self, '_last_retrieved_node_indices') and self._last_retrieved_node_indices:
            if hasattr(self, '_search_to_question_map'):
                for s_idx, q_idx in self._search_to_question_map.items():
                    if s_idx < len(self._last_retrieved_node_indices):
                        per_example_nodes[q_idx] = self._last_retrieved_node_indices[s_idx]
            n_with_nodes = sum(1 for n in per_example_nodes if n)
            print(f"[Copilot] Stored node_indices for {n_with_nodes}/{len(per_example_nodes)} examples (map size={len(self._search_to_question_map)})")
        else:
            print(f"[Copilot] No node_indices available")

        n_skipped = sum(1 for e in self._retrieved_experiences if not e)
        n_retrieved = len(self._retrieved_experiences) - n_skipped
        print(f"ACTIVE_TRAJ_NUM: {active_num_list} (retrieved={n_retrieved}, skipped={n_skipped})")

        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        # store in non_tensor_batch so it gets reordered with batch
        import numpy as np
        final_output.non_tensor_batch['retrieved_node_indices'] = np.array(per_example_nodes, dtype=object)
        # Store retrieved experiences for Agent (not in Copilot's context)
        if hasattr(self, '_retrieved_experiences'):
            final_output.non_tensor_batch['retrieved_experiences'] = np.array(self._retrieved_experiences, dtype=object)
        return final_output

    def _example_level_pad(self, tensor, str_list, queries, end_flags, active_mask):
        """Pad tensors and lists back to full batch size."""
        if active_mask.all():
            return tensor, str_list, queries, end_flags
            
        full_size = active_mask.shape[0]
        active_indices = torch.where(active_mask)[0]
        
        # Pad tensor
        padded_tensor = torch.zeros(
            (full_size, tensor.shape[1]), 
            dtype=tensor.dtype, 
            device=tensor.device
        )
        padded_tensor[active_indices] = tensor
        
        # Pad string list
        padded_str = [""] * full_size
        for i, idx in enumerate(active_indices):
            padded_str[idx.item()] = str_list[i]
            
        # Pad queries
        padded_queries = [""] * full_size
        for i, idx in enumerate(active_indices):
            padded_queries[idx.item()] = queries[i]
            
        # Pad end flags
        padded_end_flags = [False] * full_size
        for i, idx in enumerate(active_indices):
            padded_end_flags[idx.item()] = end_flags[i]
            
        return padded_tensor, padded_str, padded_queries, padded_end_flags

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> Tuple[List[str], List[bool], List[int], List[int]]:
        """
        Execute copilot actions (graph retrieval) and build the feedback.
        """
        cur_actions, contents, search_complete_flags, important_doc_ids = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        # Track conversation history for each active example
        if not hasattr(self, 'conversation_histories'):
            self.conversation_histories = [""] * len(active_mask)
            
        
        # Only include ACTIVE samples in search queries
        # Only rebuild _search_to_question_map when do_search=True
        if do_search:
            search_queries = []
            self._search_to_question_map = {}
            s_idx = 0
            for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
                if action == 'search' and active:
                    search_queries.append(contents[i])
                    self._search_to_question_map[s_idx] = i
                    s_idx += 1
            n_search = len(search_queries)

            if search_queries:
                search_results = self.batch_search(search_queries)
                assert len(search_results) == n_search
                print(f"[Copilot] Search: {n_search} active queries out of {sum(1 for a in cur_actions if a=='search')} total search actions")
            elif any(active_mask):
                # Force naive cosine search for all active sequences
                print(f"[Copilot] No active search actions, forcing naive for {sum(active_mask)} active examples")
                search_queries = []
                self._search_to_question_map = {}
                s_idx = 0
                for idx in range(len(cur_actions)):
                    if active_mask[idx]:
                        cur_actions[idx] = 'search'
                        contents[idx] = 'R:50 W:50'
                        search_queries.append(contents[idx])
                        self._search_to_question_map[s_idx] = idx
                        s_idx += 1
                n_search = len(search_queries)
                search_results = self.batch_search(search_queries)
            else:
                n_search = 0
                search_results = []
        else:
            # do_search=False: don't rebuild map, don't search
            n_search = 0
            search_results = []

        # Store retrieved experiences per example (for Agent, not Copilot)
        if not hasattr(self, '_retrieved_experiences'):
            self._retrieved_experiences = [""] * len(active_mask)

        search_result_counter = 0
        for i, (action, search_complete, active, doc_ids) in enumerate(zip(cur_actions, search_complete_flags, active_mask, important_doc_ids)):
            if not active and do_search:
                next_obs.append('')
                dones.append(True)
                valid_action.append(0)
                is_search.append(0)
            elif not active and not do_search:
                next_obs.append('')
                dones.append(True)
                valid_action.append(1)
                is_search.append(0)
            else:
                if action == 'search_complete' or not do_search:
                    next_obs.append('')
                    dones.append(True)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    if do_search:
                        search_result = search_results[search_result_counter].strip()
                        search_result_counter += 1

                        # Add important document IDs to conversation history
                        if doc_ids:
                            self.conversation_histories[i] += f"\nImportant documents: {doc_ids}\n"
                        
                        self.conversation_histories[i] += f"\nQuery: {contents[i]}\nSearch results: {search_result}"
                        
                        # Store experience text for Agent, don't feed back to Copilot
                        self._retrieved_experiences[i] = search_result
                        next_obs.append('')
                        dones.append(True)  # Copilot is done after generating R/W
                        valid_action.append(1)
                        is_search.append(1)
                    else:
                        next_obs.append('')
                        dones.append(True)
                        valid_action.append(1)
                        is_search.append(0)
                else:
                    # Invalid action — prompt to retry with R:W format
                    feedback = "\n\nThe information is not enough. Let me adjust retrieval parameters.\n<search>"
                    next_obs.append(feedback)
                    # next_obs.append('')
                    dones.append(False)
                    valid_action.append(0)
                    is_search.append(0)
            
        return next_obs, dones, valid_action, is_search
    
    
    
    def _check_feedback_for_stop(self, feedback: str) -> bool:
        """
        Check if the feedback indicates that we should stop searching.
        
        Args:
            feedback: The feedback from the generator LLM
            
        Returns:
            Boolean indicating whether to stop searching
        """
        # Look for explicit stop indicators in the feedback
        stop_patterns = [
            r'Stop Search:\s*Yes',
            r'stop searching',
            r'no need to search further',
            r'sufficient information',
            r'already have all the information needed',
            r'already have the answer',
            r'can answer the question',
            r'enough information to answer'
        ]
        
        for pattern in stop_patterns:
            if re.search(pattern, feedback, re.IGNORECASE):
                return True
                
        return False
    
        
    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str], List[bool], List[List[int]]]:
        """
        Process predictions to extract actions and content.
        Supports both <search>R:query</search> and legacy <query>JSON</query> formats.
        Returns:
            Tuple of (actions, contents, search_complete_flags, important_doc_ids)
        """
        actions = []
        contents = []
        search_complete_flags = []
        important_doc_ids = []

        for prediction in predictions:
            if isinstance(prediction, str):
                search_complete_match = re.search(r'<search_complete>(.*?)</search_complete>', prediction, re.DOTALL)
                search_match = re.search(r'<search>(.*?)</search>', prediction, re.DOTALL)
                query_match = re.search(r'<query>(.*?)</query>', prediction, re.DOTALL)
                important_info_match = re.search(r'<important_info>(.*?)</important_info>', prediction, re.DOTALL)

                # Parse important document IDs
                doc_ids = []
                if important_info_match:
                    try:
                        import json
                        doc_ids = json.loads(important_info_match.group(1).strip())
                        if not isinstance(doc_ids, list):
                            doc_ids = []
                    except:
                        doc_ids = []
                important_doc_ids.append(doc_ids)

                # Extract R:W parameters from <search>R:W</search> or <search>none</search>
                if search_match:
                    raw_text = search_match.group(1).strip()
                    if raw_text.lower() in ('none', 'skip', 'no'):
                        # Copilot decided to skip retrieval
                        content = ""
                        action = "search_complete"
                    else:
                        rw_match = re.match(r'^([\d.]+)\s*:\s*([\d.]+)$', raw_text)
                        if rw_match:
                            try:
                                R_val = max(0, min(int(float(rw_match.group(1))), 100))
                                W_val = max(0, min(int(float(rw_match.group(2))), 100))
                                content = f"R:{R_val} W:{W_val}"
                            except (ValueError, OverflowError):
                                content = "R:50 W:50"
                        else:
                            # Unparseable content, use default instead of R:0 W:0
                            content = "R:50 W:50"
                        action = "search"
                elif query_match:
                    # Legacy <query> format — treat as search with default R/W
                    content = "R:50 W:50"
                    action = "search"
                else:
                    # No search tag at all — still search with default
                    content = "R:50 W:50"
                    action = "search"

                # Check for search completion flag
                search_complete = False
                if search_complete_match:
                    complete_text = search_complete_match.group(1).strip().lower()
                    search_complete = complete_text in ("true", "yes", "1", "y")
                    if search_complete:
                        content = ""
                        action = "search_complete"

                actions.append(action)
                contents.append(content)
                search_complete_flags.append(search_complete)
            else:
                actions.append(None)
                contents.append('')
                search_complete_flags.append(False)
                important_doc_ids.append([])

        return actions, contents, search_complete_flags, important_doc_ids
    

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> DataProto:
        """Compose final output for the search copilot."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and info mask
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Also stores retrieved node_indices in self._last_retrieved_node_indices
        for bandit reward updates.
        """
        try:
            response = self._batch_search(queries)
            results = response['result']
            # Store node_indices for bandit reward update
            self._last_retrieved_node_indices = response.get('node_indices', [])
        except Exception as e:
            print(f"Error in batch_search: {e}, queries: {queries}")
            self._last_retrieved_node_indices = []
            return ["Error"] * len(queries)

        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        """Call the search API.

        queries: list of "R:XX W:YY" strings from _postprocess_responses.
        The actual semantic queries come from self.original_questions (task descriptions).
        R and W are retrieval control parameters passed separately to the graph server.
        """
        import re as _re
        R_values = []
        W_values = []
        semantic_queries = []

        # Build a mapping: search_idx → original_question_idx
        # queries come from contents where action == 'search', so we need
        # to map back to the original question indices
        search_idx = 0
        for q in queries:
            r_match = _re.search(r'R:(\d+)', q)
            w_match = _re.search(r'W:(\d+)', q)
            R_val = int(r_match.group(1)) if r_match else 0
            W_val = int(w_match.group(1)) if w_match else 0
            R_values.append(R_val)
            W_values.append(W_val)

            # Use original task description as the semantic query
            # Fall back to the raw query if original_questions not available
            if hasattr(self, 'original_questions') and hasattr(self, '_search_to_question_map'):
                q_idx = self._search_to_question_map.get(search_idx, search_idx)
                if q_idx < len(self.original_questions) and self.original_questions[q_idx]:
                    semantic_queries.append(self.original_questions[q_idx])
                else:
                    semantic_queries.append(q)  # fallback
            elif hasattr(self, 'original_questions'):
                # Best effort: use search_idx to index original_questions
                if search_idx < len(self.original_questions) and self.original_questions[search_idx]:
                    semantic_queries.append(self.original_questions[search_idx])
                else:
                    semantic_queries.append(q)
            else:
                semantic_queries.append(q)
            search_idx += 1

        payload = {
            "queries": semantic_queries,
            "R_values": R_values,
            "W_values": W_values,
            "topk": self.config.topk,
            "return_scores": True
        }

        import random
        if random.random() < 0.2:
            print(f"[Copilot] Search: R={R_values[:3]} W={W_values[:3]} queries={[q[:50] for q in semantic_queries[:3]]}")

        return requests.post(self.config.search_url, json=payload, timeout=300).json()

    def _passages2string(self, retrieval_result):
        """Format retrieval results into a string.
        Each experience separated by \n---\n for downstream splitting.
        """
        parts = []
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item.get('document', doc_item)
            if isinstance(content, dict):
                text = content.get('text', '')
            else:
                text = str(content)
            parts.append(text.strip())

        return "\n---\n".join(parts)

    # ================================================================
    # ALFWorld Step 0: Multi-turn Skill Search for Embodied Tasks
    # ================================================================

    def run_alfworld_step0(self, gen_batch, initial_input_ids: torch.Tensor) -> DataProto:
        """
        Run multi-turn skill search (Step 0) for ALFWorld.

        Same as run_llm_loop but designed for embodied tasks:
        - Copilot searches experience graph for task strategies
        - Returns retrieved experiences (to be used by frozen Agent)
        - No environment interaction happens here

        The final output includes 'retrieved_experiences' in meta_info
        for the Agent to use during environment interaction.
        """
        # Reuses the copilot retrieval loop
        output = self.run_llm_loop(gen_batch, initial_input_ids)

        # Extract the important experiences from the search trajectory
        # These will be passed to the frozen Agent for environment interaction
        batch_size = initial_input_ids.shape[0]
        retrieved_experiences = []

        if hasattr(output, 'batch') and 'responses' in output.batch:
            # Only decode responses (not prompts) to avoid extracting example docs from prompt
            response_sequences = self.tokenizer.batch_decode(
                output.batch['responses'],
                skip_special_tokens=True
            )
            for seq in response_sequences:
                experiences = self._extract_important_experiences(seq)
                retrieved_experiences.append(experiences)
        else:
            retrieved_experiences = [""] * batch_size

        # Store in meta_info for the Agent to use
        output.meta_info['retrieved_experiences'] = retrieved_experiences
        return output

    def _extract_important_experiences(self, sequence_str: str) -> str:
        """
        Extract important experiences from a Copilot search trajectory.

        Parses <information> blocks and <important_info> selections,
        returns formatted experience text for the Agent.
        """
        import re as _re

        info_blocks = []
        important_infos = []

        # Find all information blocks
        for match in _re.finditer(r'<information>(.*?)</information>', sequence_str, _re.DOTALL):
            info_blocks.append({
                'position': match.start(),
                'content': match.group(1),
                'processed': False,
            })

        # Find all important_info tags
        for match in _re.finditer(r'<important_info>(.*?)</important_info>', sequence_str, _re.DOTALL):
            try:
                numbers = _re.findall(r'\d+', match.group(1))
                ids = [int(n) for n in numbers if 1 <= int(n) <= 8]
                ids = list(dict.fromkeys(ids))  # deduplicate
            except:
                ids = []
            important_infos.append({
                'position': match.start(),
                'important_ids': ids,
            })

        # Match important_info to closest preceding information block
        all_experiences = []
        seen = set()

        for imp in important_infos:
            closest = None
            for block in info_blocks:
                if not block['processed'] and block['position'] < imp['position']:
                    closest = block

            if closest:
                closest['processed'] = True
                # Parse Doc N(Title: ...) ... format
                doc_pattern = _re.compile(
                    r'Doc\s*(\d+)\s*\(\s*Title\s*:\s*([^)]+)\)\s*(.*?)(?=Doc\s*\d+|$)',
                    _re.DOTALL | _re.IGNORECASE
                )
                for doc_match in doc_pattern.finditer(closest['content']):
                    doc_id = int(doc_match.group(1))
                    text = doc_match.group(3).strip()
                    if doc_id in imp['important_ids'] and text not in seen:
                        seen.add(text)
                        all_experiences.append(text)

        # Also include docs from unprocessed blocks (initial retrieval)
        for block in info_blocks:
            if not block['processed']:
                doc_pattern = _re.compile(
                    r'Doc\s*(\d+)\s*\(\s*Title\s*:\s*([^)]+)\)\s*(.*?)(?=Doc\s*\d+|$)',
                    _re.DOTALL | _re.IGNORECASE
                )
                for doc_match in doc_pattern.finditer(block['content']):
                    text = doc_match.group(3).strip()
                    if text not in seen:
                        seen.add(text)
                        all_experiences.append(text)

        if not all_experiences:
            return ""

        return "\n---\n".join(all_experiences)
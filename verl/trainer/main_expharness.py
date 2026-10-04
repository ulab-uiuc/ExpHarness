"""
ExpHarness PPO entry point (ALFWorld / AppWorld / static ExpSuite tasks).

- Retrieval copilot (e.g. Qwen2.5-3B-Instruct, trained with PPO): reads the task and
  outputs retrieval controls <search>R:W</search> for the experience graph.
- Executor (frozen LLM behind an API): solves the task with and without the retrieved
  experiences.
- Reward: r = (s_with - s_without) + eta * s_with   (utility + generation, eta = 1)
- Co-evolution: the same reward updates the utility of the retrieved graph nodes, and
  new trajectories are summarized into skills / lessons and inserted into the graph.

The environment is selected per batch from `data_source` (alfworld / appworld* / static).
"""

from verl import DataProto
import torch
import re
import os
import json
import time
import requests
import numpy as np
import threading
import random
from queue import Queue

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from expharness.rewards.utility_reward import (
    compute_alfworld_reward,
    ALFWorldZeroshotCache,
    compute_reward_metrics,
)

USE_UTILITY_SCORE = True
USE_GENERATION_SCORE = True


# ============================================================================
# Experience Graph Async Updater
# ============================================================================

class ExperienceGraphUpdater:
    """Background worker that summarizes finished trajectories into experiences
    (skills from successes, lessons from failures) and inserts them into the graph."""

    def __init__(self, graph_url, max_experiences_per_batch=10, enabled=True, **kwargs):
        self.graph_url = graph_url
        self.max_experiences_per_batch = max_experiences_per_batch
        self.enabled = enabled

        if self.enabled:
            self.queue = Queue()
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def submit(self, trajectories):
        if self.enabled:
            self.queue.put(trajectories)

    def _summarize_alfworld(self, task, trajectory_str, won):
        """Summarize an ALFWorld trajectory into a reusable experience."""
        if won:
            prompt = f"""An ALFWorld household task was completed successfully.

Task: {task}
Full trajectory (action → observation):
{trajectory_str}

Extract a reusable skill (3-5 sentences). Include:
1. The general task category (e.g. pick_and_place, heat_then_place, clean_then_place, cool_then_place, examine_in_light, pick_two)
2. The concrete step-by-step strategy that worked (e.g. "go to countertop to find the object, then pick it up, then go to sinkbasin to clean it")
3. Common locations where target objects are found (e.g. "soapbar is usually on countertop, bathtubbasin, or shelf")

Be specific and actionable. Use actual object/location types (countertop, sinkbasin, microwave) not abstract placeholders.
Output format: SKILL: [your skill text]"""
        else:
            prompt = f"""An ALFWorld household task failed after all steps.

Task: {task}
Full trajectory (action → observation):
{trajectory_str}

Extract a reusable lesson (3-5 sentences). Include:
1. The general task category
2. What specific mistake was made (e.g. "kept visiting wrong locations", "forgot to clean before placing")
3. What the agent should have done differently

Be specific and actionable. Use actual object/location types not abstract placeholders.
Output format: SKILL: [your lesson text]"""

        full_prompt = "You are an expert at analyzing household robot trajectories. Extract specific, actionable lessons that would help an agent succeed at similar tasks in the future.\n\n" + prompt
        from expharness.utils.llm_api import summarizer_call
        response = summarizer_call(full_prompt, max_tokens=300)
        match = re.search(r'SKILL:\s*(.*)', response, re.DOTALL)
        skill_text = match.group(1).strip() if match else response.strip()
        prefix = "[SUCCESS]" if won else "[FAILURE]"
        return f"{prefix} {skill_text}"

    def _summarize_appworld(self, task, trajectory_str, score):
        """Summarize an AppWorld trajectory into a structured skill."""
        if score >= 0.8:
            outcome = f"SUCCESSFUL (score: {score:.2f})"
        elif score >= 0.3:
            outcome = f"PARTIALLY SUCCESSFUL (score: {score:.2f})"
        else:
            outcome = f"FAILED (score: {score:.2f})"

        prompt = f"""Analyze this {outcome} AppWorld code generation episode.

Task: {task}
Trajectory:
{trajectory_str}

Extract ONE concise, actionable skill. Focus on SPECIFIC API details, not generic advice.

Format: [Short Title] Specific principle. When: trigger condition.

Good examples (specific, with actual API names/fields):
- [Spotify Login Fields] apis.spotify.login() requires email and password as keyword arguments; get them via apis.supervisor.show_account_passwords(). When: Authenticating with Spotify.
- [Venmo Transactions Return Dict] apis.venmo.list_transactions() returns a dict with key "transactions", not a list; iterate result["transactions"]. When: Processing Venmo transaction data.
- [File System Read Path] Use apis.file_system.read_file(file_path=...) with absolute paths starting from "~/"; relative paths fail silently. When: Reading files in AppWorld.
- [Spotify Pagination Pattern] apis.spotify.search_songs(query=..., page_index=N) returns empty list when N exceeds total pages; loop until empty. When: Searching or listing all Spotify items.

BAD examples (too generic, do NOT generate these):
- "Always inspect API docs first" — this is already in the system prompt
- "Verify API method names" — too vague, no specific detail
- "Print results after each step" — obvious, not helpful

Rules: Include SPECIFIC app names, API method names, parameter names, or return value structures from the trajectory. Output ONLY the skill line."""

        from expharness.utils.llm_api import summarizer_call
        response = summarizer_call(prompt, max_tokens=200)
        skill_text = (response or "").strip()
        skill_text = re.sub(r'^[\-\*]\s*', '', skill_text)
        if not skill_text or len(skill_text) < 20:
            skill_text = "[Inspect API First] Always check API documentation before calling unfamiliar endpoints. When: Starting any new app interaction."
        if score >= 0.8:
            prefix = "[SUCCESS]"
        elif score >= 0.3:
            prefix = "[PARTIAL]"
        else:
            prefix = "[FAILURE]"
        return f"{prefix} {skill_text}"

    def _summarize_qa(self, question, response, domain, correct):
        """Summarize a static-task (QA / math / code) result into a short reusable skill."""
        label = "correctly" if correct else "incorrectly"
        prompt = f"""A {domain} question was answered {label}.

Question: {question[:300]}
Response: {response[:300]}

Summarize the reasoning into ONE short skill (1-2 sentences, under 100 words).
Format: "[problem type]: [key technique or tip]"
Do NOT include the specific numbers or answer from this question.

Output format: SKILL: [your skill text]"""

        full_prompt = "You are an expert at analyzing reasoning patterns.\n\n" + prompt
        from expharness.utils.llm_api import summarizer_call
        resp = summarizer_call(full_prompt, max_tokens=150)
        match = re.search(r'SKILL:\s*(.*)', resp, re.DOTALL)
        skill_text = match.group(1).strip() if match else resp.strip()
        # Enforce max length — prevent verbose responses from polluting graph
        skill_text = skill_text[:200]
        prefix = "[SUCCESS]" if correct else "[FAILURE]"
        return f"{prefix} {skill_text}"

    def _worker(self):
        while True:
            try:
                trajectories = self.queue.get()
                if not trajectories:
                    continue

                env_name = trajectories[0].get('env_name', 'alfworld') if trajectories else 'alfworld'
                is_scored = (env_name == 'appworld')

                if env_name == 'qa':
                    good = [t for t in trajectories if t.get('score', 0) > 0]
                    bad = [t for t in trajectories if t.get('score', 0) == 0]
                elif is_scored:
                    good = [t for t in trajectories if t.get('score', 0) >= 0.3]
                    bad = [t for t in trajectories if t.get('score', 0) < 0.3]
                else:
                    good = [t for t in trajectories if t.get('won', False)]
                    bad = [t for t in trajectories if not t.get('won', False)]

                selected = []
                n_good = min(len(good), self.max_experiences_per_batch // 2)
                if n_good > 0: selected.extend(random.sample(good, n_good))
                n_bad = min(len(bad), self.max_experiences_per_batch - n_good)
                if n_bad > 0: selected.extend(random.sample(bad, n_bad))

                experiences = []
                for traj in selected:
                    if env_name == 'qa':
                        exp = self._summarize_qa(
                            traj.get('question', ''), traj.get('response', ''),
                            traj.get('domain', 'qa'), traj.get('score', 0) > 0)
                        if exp: experiences.append(exp)
                        continue
                    steps = traj.get('trajectory', [])[:15 if is_scored else 50]
                    if env_name == 'appworld':
                        # AppWorld: code + output format
                        traj_str = "\n".join([
                            f"Step {s.get('step',i)+1} CODE:\n{s.get('action', s.get('code',''))}\nStep {s.get('step',i)+1} OUTPUT:\n{s.get('obs', s.get('output',''))[:200]}"
                            for i, s in enumerate(steps)
                        ])
                    else:
                        traj_str = "\n".join([
                            f"Step {s.get('step',i)+1}: ACTION: {s.get('action', s.get('code',''))} → OBS: {s.get('obs', s.get('output',''))[:150]}"
                            for i, s in enumerate(steps)
                        ])
                    if env_name == 'appworld':
                        exp = self._summarize_appworld(
                            traj.get('task', ''), traj_str, traj.get('score', 0))
                    else:
                        exp = self._summarize_alfworld(
                            traj.get('task', ''), traj_str, traj.get('won', False))
                    if exp: experiences.append(exp)

                if experiences:
                    try:
                        resp = requests.post(f"{self.graph_url}/add_experience",
                                             json={"experiences": experiences}, timeout=30)
                        result = resp.json()
                        print(f"[GraphUpdater] Added {result.get('added', 0)} experiences ({env_name})")
                    except Exception as e:
                        print(f"[GraphUpdater] Failed: {e}")
            except Exception as e:
                print(f"[GraphUpdater] Worker error: {e}")


# ============================================================================
# Reward Manager
# ============================================================================

class ExpHarnessRewardManager:
    """
    Utility-grounded reward for the retrieval copilot.

    For each task:
    1. The copilot picks (R, W) -> experiences retrieved from the graph
    2. Frozen executor runs WITH the experiences    -> s_with
    3. Frozen executor runs WITHOUT experiences     -> s_without (cached when possible)
    4. Reward = (s_with - s_without) + s_with, written to the copilot's last token
    5. The same reward updates the utility of the retrieved graph nodes
    """

    def __init__(self, tokenizer, num_examine=0, zeroshot_cache_path="data/alfworld/zeroshot_cache.json",
                 graph_updater=None, val_only=False, output_context_dir=None, graph_url=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.zeroshot_cache = ALFWorldZeroshotCache(zeroshot_cache_path)
        self.graph_updater = graph_updater
        self.val_only = val_only
        self._graph_url = graph_url or os.environ.get('GRAPH_URL', 'http://127.0.0.1:8001')
        self.output_context_dir = output_context_dir or "outputs/copilot_sequences"
        os.makedirs(self.output_context_dir, exist_ok=True)
        # Cache AppWorld no-memory baseline per task_id to avoid re-running it every step.
        self._appworld_baseline_cache = {}
        baseline_file = os.environ.get("APPWORLD_BASELINE_CACHE", "data/appworld/baseline_without_exp.json")
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
                for task_id, result in baseline_data.items():
                    self._appworld_baseline_cache[task_id] = result
                print(f"[AppWorld] Loaded {len(self._appworld_baseline_cache)} baseline scores from {baseline_file}")
            except Exception as e:
                print(f"[AppWorld] Failed to load baseline cache: {e}")
        self._appworld_reward_calls = 0
        # Evaluation statistics (val_only): per data_source success rate / steps
        self._eval_stats = {}

    def _record_eval(self, source, success_with, success_without, steps_with=0, steps_without=0):
        if not self.val_only:
            return
        st = self._eval_stats.setdefault(source, {
            'n': 0, 'success_with': 0, 'success_without': 0, 'steps_with': 0, 'steps_without': 0})
        st['n'] += 1
        st['success_with'] += int(bool(success_with))
        st['success_without'] += int(bool(success_without))
        st['steps_with'] += steps_with or 0
        st['steps_without'] += steps_without or 0

    def __call__(self, data: DataProto):
        """
        Compatible with ray_trainer's reward_fn(data) interface.

        Flow:
        1. Decode Copilot's search trajectories
        2. Extract retrieved experiences from each trajectory
        3. Run Agent (Gemini API) to generate plans WITH and WITHOUT experiences
        4. Judge both plans → compute utility + generation reward
        5. Assign reward to last valid token of Copilot's response
        """
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # Auto-detect environment type
        first_source = data[0].non_tensor_batch.get('data_source', 'alfworld')
        if first_source.startswith('appworld'):
            return self._compute_appworld_rewards(data)
        elif first_source != 'alfworld':
            return self._compute_qa_rewards(data)
        return self._compute_alfworld_rewards(data)

    def _compute_alfworld_rewards(self, data: DataProto):
        """ALFWorld: run real TextWorld episodes with / without experiences."""

        from expharness.envs.alfworld_agent import evaluate_batch, set_api_keys

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        batch_size = len(data)

        # Step 1: Decode all sequences and extract tasks + experiences
        tasks = []
        game_files = []
        experiences_list = []
        valid_response_lengths = []

        for i in range(batch_size):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_lengths.append(int(valid_response_length.item()))

            # Only decode response (not prompt) to avoid extracting example docs from prompt
            response_str = self.tokenizer.decode(valid_response_ids)

            # Extract task and gamefile from ground_truth
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            task = ground_truth.get('task', '') if isinstance(ground_truth, dict) else ''
            gamefile = ground_truth.get('gamefile', '') if isinstance(ground_truth, dict) else ''
            tasks.append(task)
            game_files.append(gamefile)

            # Get experiences from non_tensor_batch (stored by the copilot rollout, not in Copilot's context)
            retrieved_exps = data.non_tensor_batch.get('retrieved_experiences', None)
            if retrieved_exps is not None and i < len(retrieved_exps) and retrieved_exps[i]:
                experiences = str(retrieved_exps[i])
            else:
                # Fallback: try extracting from Copilot response (legacy path)
                experiences = self._extract_experiences(response_str)
            experiences_list.append(experiences)
            if i < 3:  # Debug: show first 3
                print(f"[DEBUG] Sample {i}: task={task[:40]}, exp_len={len(experiences) if experiences else 0}, exp_preview={str(experiences)[:100] if experiences else 'NONE'}")
                if not experiences:
                    print(f"[DEBUG] Response: {response_str[:200]}")

        # Step 2: Batch evaluate with Agent (real ALFWorld env, ProcessPoolExecutor)
        # Use game_files from ground_truth so Copilot's experiences match the actual game
        # For any missing/invalid gamefiles, sample a random replacement
        max_workers = int(os.environ.get('ALFWORLD_NUM_WORKERS', '10'))
        from expharness.envs.alfworld_agent import get_solvable_game_files, resolve_gamefile
        all_games = get_solvable_game_files()
        n_replaced = 0
        for i in range(batch_size):
            game_files[i] = resolve_gamefile(game_files[i])
            if not game_files[i] or not os.path.exists(game_files[i]):
                _, replacement = random.choice(all_games)
                game_files[i] = replacement
                n_replaced += 1
        if n_replaced > 0:
            print(f"[ALFWorld Reward] WARNING: {n_replaced}/{batch_size} gamefiles missing, replaced with random games")
        # Baseline: no experience at all (zero-shot Gemini)
        # Cache zero-shot results per gamefile since they don't change
        naive_scores = {}
        need_naive_eval = []
        for i in range(batch_size):
            cached = self.zeroshot_cache.get(game_files[i])
            if cached is not None:
                naive_scores[i] = cached
            else:
                need_naive_eval.append(i)

        n_cached = len(naive_scores)
        n_to_eval = len(need_naive_eval)
        print(f"[ALFWorld Reward] Running {batch_size} WITH-exp + {n_to_eval} NO-exp episodes "
              f"({n_cached} cached), {max_workers} workers...")

        from expharness.envs.alfworld_agent import _run_episode_standalone
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results_with = [None] * batch_size
        results_without = [None] * batch_size
        default = {"won": False, "score": 0.0, "steps": 0, "trajectory": [], "task": ""}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            worker_id = 0
            for i in range(batch_size):
                # WITH exp: always run
                f_with = executor.submit(_run_episode_standalone,
                    (game_files[i], experiences_list[i], 50, worker_id))
                futures[f_with] = (i, "with")
                worker_id += 1

                # NO exp (zero-shot): only run if not cached
                if i in need_naive_eval:
                    f_naive = executor.submit(_run_episode_standalone,
                        (game_files[i], None, 50, worker_id))
                    futures[f_naive] = (i, "naive")
                    worker_id += 1

            for future in as_completed(futures):
                idx, mode = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = default.copy()
                if "error" in result:
                    result = default.copy()

                if mode == "with":
                    results_with[idx] = result
                else:
                    results_without[idx] = result
                    # Cache the naive result
                    if game_files[idx]:
                        self.zeroshot_cache.set(game_files[idx], result['score'])

        # Fill cached naive scores
        for i, score in naive_scores.items():
            results_without[i] = {"won": score >= 1.0, "score": score, "steps": 0, "trajectory": []}

        # Step 3: Compute rewards
        trajectories_for_graph = []
        for i in range(batch_size):
            score_with = results_with[i]['score']
            score_without = results_without[i]['score']

            reward = compute_alfworld_reward(
                score_with, score_without,
                use_utility=USE_UTILITY_SCORE,
                use_generation=USE_GENERATION_SCORE,
            )

            # Format penalty: if Copilot produced no experiences at all, penalize
            has_experiences = bool(experiences_list[i] and experiences_list[i].strip())
            if not has_experiences:
                reward -= 0.5

            # Assign to last valid token
            vrl = valid_response_lengths[i]
            if vrl > 0:
                reward_tensor[i, vrl - 1] = reward

            self._record_eval('alfworld', results_with[i].get('won', False), results_without[i].get('won', False),
                              results_with[i].get('steps', 0), results_without[i].get('steps', 0))

            # Print occasionally
            if random.random() < 0.15:
                print(f"[ALFWorld] Task: {tasks[i][:60]}")
                print(f"  WITH exp: won={results_with[i]['won']}, steps={results_with[i]['steps']}, score={score_with:.2f}")
                print(f"  NAIVE:    won={results_without[i]['won']}, score={score_without:.2f}")
                print(f"  utility={score_with - score_without:.2f}, reward={reward:.2f}")

            # Collect for graph update (with actual trajectory)
            if not self.val_only and self.graph_updater:
                trajectories_for_graph.append({
                    'task': tasks[i],
                    'won': results_with[i].get('won', False),
                    'trajectory': results_with[i].get('trajectory', []),
                    'score': score_with,
                    'plan': results_with[i].get('plan', ''),
                })

        # Submit to graph updater
        if trajectories_for_graph and self.graph_updater:
            self.graph_updater.submit(trajectories_for_graph)

        # Update bandit rewards on graph nodes
        # Node indices come from Copilot's meta_info (set by expharness/copilot/generation.py),
        # NOT from episode results (which don't have this info)
        if not self.val_only:
            try:
                graph_url = self._graph_url
                # Read from non_tensor_batch (survives reorder), not meta_info
                per_example_nodes_raw = data.non_tensor_batch.get('retrieved_node_indices', None)
                if per_example_nodes_raw is None:
                    per_example_nodes = [[] for _ in range(batch_size)]
                else:
                    # Convert numpy array of objects → list of lists
                    per_example_nodes = [list(n) if hasattr(n, '__len__') else [] for n in per_example_nodes_raw]

                episode_node_indices = []
                episode_rewards = []
                for i in range(batch_size):
                    node_ids = per_example_nodes[i] if i < len(per_example_nodes) else []
                    if len(node_ids) > 0:
                        score_with = results_with[i]['score']
                        score_without = results_without[i]['score']
                        r = compute_alfworld_reward(score_with, score_without,
                                                     use_utility=USE_UTILITY_SCORE,
                                                     use_generation=USE_GENERATION_SCORE)
                        episode_node_indices.append(list(node_ids))
                        episode_rewards.append(r)
                n_with_nodes = sum(1 for n in per_example_nodes if len(n) > 0)
                print(f"[Graph] {n_with_nodes}/{batch_size} examples retrieved nodes, {len(episode_node_indices)} utility updates")
                if episode_node_indices:
                    import requests as _req
                    resp = _req.post(f"{graph_url}/update_rewards",
                              json={"node_indices": episode_node_indices, "rewards": episode_rewards},
                              timeout=10)
                    print(f"[Graph] Utility update sent: {len(episode_node_indices)} episodes, rewards={[f'{r:.2f}' for r in episode_rewards[:5]]}")
            except Exception as e:
                print(f"[WARN] Graph utility update failed: {e}")

        # Log summary
        n_won_with = sum(1 for r in results_with if r.get('won', False))
        n_won_without = sum(1 for r in results_without if r.get('won', False))
        mean_reward = reward_tensor.sum().item() / max(batch_size, 1)
        avg_steps_with = sum(r.get('steps', 0) for r in results_with) / max(batch_size, 1)
        avg_steps_without = sum(r.get('steps', 0) for r in results_without) / max(batch_size, 1)
        print(f"[ALFWorld Reward] Done. won_with={n_won_with}/{batch_size}, "
              f"won_without={n_won_without}/{batch_size}, mean_reward={mean_reward:.3f}, "
              f"steps_with={avg_steps_with:.1f}, steps_without={avg_steps_without:.1f}")
        return reward_tensor

    def _compute_appworld_rewards(self, data: DataProto):
        """Compute rewards for AppWorld tasks via the external AppWorld episode servers
        (HTTP, so the AppWorld env can live in its own Python environment)."""
        import requests as http_requests

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        batch_size = len(data)

        task_ids = []
        experiences_list = []
        valid_response_lengths = []

        for i in range(batch_size):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_lengths.append(int(valid_response_length.item()))

            gt_raw = data_item.non_tensor_batch.get('reward_model', '{}')
            if isinstance(gt_raw, str):
                try:
                    gt = json.loads(gt_raw)
                except:
                    gt = {}
            elif isinstance(gt_raw, dict):
                gt = gt_raw
            else:
                gt = {}
            gt_inner = gt.get('ground_truth', gt)
            if isinstance(gt_inner, str):
                try:
                    gt_inner = json.loads(gt_inner)
                except:
                    gt_inner = {}
            task_id = gt_inner.get('task_id', f'unknown_{i}')
            task_ids.append(task_id)

            retrieved_exps = data.non_tensor_batch.get('retrieved_experiences', None)
            if retrieved_exps is not None and i < len(retrieved_exps) and retrieved_exps[i]:
                experiences = str(retrieved_exps[i])
            else:
                experiences = None
            experiences_list.append(experiences)

        max_steps = int(os.environ.get('APPWORLD_MAX_STEPS', '20'))
        # Support multiple server ports: "http://127.0.0.1:8006" or "http://127.0.0.1:8006,8007,8008,8009"
        appworld_url_raw = os.environ.get('APPWORLD_SERVER_URL', 'http://127.0.0.1:8006')

        def _expand_appworld_urls(raw):
            parts = [p.strip() for p in str(raw).split(',') if p.strip()]
            if not parts:
                return ['http://127.0.0.1:8006']
            if len(parts) > 1 and '://' in parts[0] and all('://' not in p for p in parts[1:]):
                base = parts[0].rsplit(':', 1)[0]
                return [parts[0]] + [f"{base}:{p}" for p in parts[1:]]
            return parts

        appworld_urls = _expand_appworld_urls(appworld_url_raw)

        self._appworld_reward_calls += 1
        baseline_refresh_every = int(os.environ.get("APPWORLD_BASELINE_REFRESH_EVERY", "0"))

        results_with = [None] * batch_size
        results_without = [None] * batch_size
        default = {"success": False, "score": 0.0, "steps": 0, "trajectory": [], "instruction": ""}

        healthy_urls = []
        for url in appworld_urls:
            try:
                resp = http_requests.get(f"{url}/stats", timeout=2)
                if resp.status_code == 200 and resp.json().get("mode") == "single-threaded":
                    healthy_urls.append(url)
                else:
                    print(f"[AppWorld] Ignoring non-AppWorld server at {url}: {resp.text[:120]}")
            except Exception as e:
                print(f"[AppWorld] Server unavailable at {url}: {e}")
        if not healthy_urls:
            print("[AppWorld] No healthy AppWorld episode servers; returning zero rewards.")
            return reward_tensor
        if len(healthy_urls) != len(appworld_urls):
            print(f"[AppWorld] Using healthy servers only: {healthy_urls}")
        appworld_urls = healthy_urls

        # Each AppWorld child is a single-threaded HTTPServer. Keep at most one
        # in-flight request per child to avoid client-side pileups that look like hangs.
        # Use baseline cache for without-exp to avoid redundant runs.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cache_hits = 0
        requests_to_send = []  # (index, mode, payload)
        for i in range(batch_size):
            requests_to_send.append((i, "with", {
                "task_id": task_ids[i],
                "experiences": experiences_list[i],
                "max_steps": max_steps,
            }))
            cached = self._appworld_baseline_cache.get(task_ids[i])
            need_refresh = (baseline_refresh_every > 0 and self._appworld_reward_calls % baseline_refresh_every == 0)
            if cached is not None and not need_refresh:
                results_without[i] = cached
                cache_hits += 1
            else:
                requests_to_send.append((i, "without", {
                    "task_id": task_ids[i],
                    "experiences": None,
                    "max_steps": max_steps,
                }))

        request_timeout = float(os.environ.get('APPWORLD_REQUEST_TIMEOUT', '150'))
        jobs_by_url = [[] for _ in appworld_urls]
        for seq, req in enumerate(requests_to_send):
            jobs_by_url[seq % len(appworld_urls)].append(req)

        def _send_queue(url, jobs):
            outputs = []
            for idx, mode, payload in jobs:
                try:
                    resp = http_requests.post(
                        f"{url}/run_episode",
                        json=payload, timeout=request_timeout)
                    resp.raise_for_status()
                    outputs.append((idx, mode, resp.json()))
                except Exception as e:
                    print(f"[AppWorld] request failed url={url} task={payload.get('task_id')} mode={mode}: {e}")
                    outputs.append((idx, mode, default.copy()))
            return outputs

        num_parallel = min(
            max(1, int(os.environ.get('APPWORLD_POOL_WORKERS', str(len(appworld_urls))))),
            len(appworld_urls),
            max(1, len(requests_to_send)),
        )
        with ThreadPoolExecutor(max_workers=num_parallel) as pool:
            futures = {
                pool.submit(_send_queue, url, jobs): url
                for url, jobs in zip(appworld_urls, jobs_by_url)
                if jobs
            }
            for future in as_completed(futures):
                for idx, mode, result in future.result():
                    if not result:
                        result = default.copy()
                    if mode == "with":
                        results_with[idx] = result
                    else:
                        results_without[idx] = result
                        self._appworld_baseline_cache[task_ids[idx]] = result

        for i in range(batch_size):
            if results_with[i] is None:
                results_with[i] = default.copy()
            if results_without[i] is None:
                results_without[i] = default.copy()

        if cache_hits > 0:
            print(f"[AppWorld] baseline cache hit {cache_hits}/{batch_size}")

        # Utility-grounded reward
        for i in range(batch_size):
            score_with = results_with[i].get('score', 0.0)
            score_without = results_without[i].get('score', 0.0)

            utility = score_with - score_without
            generation = score_with
            if USE_UTILITY_SCORE and USE_GENERATION_SCORE:
                reward = utility + generation
            elif USE_UTILITY_SCORE:
                reward = utility
            else:
                reward = generation

            vrl = valid_response_lengths[i]
            if vrl > 0:
                reward_tensor[i, vrl - 1] = reward

            self._record_eval(data[i].non_tensor_batch.get('data_source', 'appworld'),
                              results_with[i].get('success', False), results_without[i].get('success', False),
                              results_with[i].get('steps', 0), results_without[i].get('steps', 0))

            if random.random() < 0.15:
                print(f"[AppWorld] task={task_ids[i]}, with={score_with:.2f}, "
                      f"without={score_without:.2f}, reward={reward:.2f}")

        n_success_with = sum(1 for r in results_with if r.get('success', False))
        n_success_without = sum(1 for r in results_without if r.get('success', False))
        mean_reward = reward_tensor.sum().item() / max(batch_size, 1)
        print(f"[AppWorld Reward] Done. success_with={n_success_with}/{batch_size}, "
              f"success_without={n_success_without}/{batch_size}, mean_reward={mean_reward:.3f}")

        # Update utilities of the retrieved graph nodes with the same reward (as for ALFWorld / static)
        if not self.val_only:
            try:
                per_example_nodes_raw = data.non_tensor_batch.get('retrieved_node_indices', None)
                if per_example_nodes_raw is not None:
                    per_example_nodes = [list(n) if hasattr(n, '__len__') else [] for n in per_example_nodes_raw]
                    episode_node_indices = []
                    episode_rewards = []
                    for i in range(batch_size):
                        node_ids = per_example_nodes[i] if i < len(per_example_nodes) else []
                        if len(node_ids) > 0:
                            s_with = results_with[i].get('score', 0.0)
                            s_without = results_without[i].get('score', 0.0)
                            episode_node_indices.append(list(node_ids))
                            episode_rewards.append(compute_alfworld_reward(
                                s_with, s_without,
                                use_utility=USE_UTILITY_SCORE, use_generation=USE_GENERATION_SCORE))
                    if episode_node_indices:
                        requests.post(f"{self._graph_url}/update_rewards",
                                      json={"node_indices": episode_node_indices, "rewards": episode_rewards},
                                      timeout=10)
                        print(f"[Graph] Utility update sent: {len(episode_node_indices)} AppWorld episodes")
            except Exception as e:
                print(f"[WARN] Graph utility update failed: {e}")

        # Dynamic graph update
        if not self.val_only and self.graph_updater:
            trajectories_for_graph = []
            for i in range(batch_size):
                rw = results_with[i]
                trajectories_for_graph.append({
                    'task': rw.get('instruction', ''),
                    'trajectory': [{'action': s.get('code', ''), 'obs': s.get('output', ''), 'step': s.get('step', 0)}
                                   for s in rw.get('trajectory', [])],
                    'won': rw.get('success', False),
                    'score': rw.get('score', 0.0),
                    'env_name': 'appworld',
                })
            self.graph_updater.submit(trajectories_for_graph)

        return reward_tensor

    def _compute_qa_rewards(self, data: DataProto):
        """Static ExpSuite tasks (QA / math / code): single-turn executor calls."""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        batch_size = len(data)

        questions = []
        data_sources = []
        ground_truths = []
        extra_infos = []
        experiences_list = []
        valid_response_lengths = []

        for i in range(batch_size):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_lengths.append(int(valid_response_length.item()))

            # Get metadata
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            gt_raw = data_item.non_tensor_batch.get('reward_model', '{}')
            extra_info = data_item.non_tensor_batch.get('extra_info', '{}')

            # Extract question from prompt (<question> tags)
            prompt_str = self.tokenizer.decode(prompt_ids[prompt_ids != self.tokenizer.pad_token_id])
            q_match = re.search(r'<question>(.*?)</question>', prompt_str, re.DOTALL)
            question = q_match.group(1).strip() if q_match else prompt_str[-500:]

            questions.append(question)
            data_sources.append(data_source)
            ground_truths.append(gt_raw)
            extra_infos.append(extra_info)

            # Get experiences
            retrieved_exps = data.non_tensor_batch.get('retrieved_experiences', None)
            if retrieved_exps is not None and i < len(retrieved_exps) and retrieved_exps[i]:
                experiences = str(retrieved_exps[i])
            else:
                experiences = None
            experiences_list.append(experiences)

            if i < 3:
                exp_len = len(experiences) if experiences else 0
                print(f"[DEBUG] QA Sample {i}: source={data_source}, exp_len={exp_len}")

        # Run WITH-exp and WITHOUT-exp via API
        from expharness.envs.static_agent import _run_qa_standalone
        from concurrent.futures import ProcessPoolExecutor, as_completed

        max_workers = int(os.environ.get('QA_NUM_WORKERS', '16'))
        results_with = [None] * batch_size
        results_without = [None] * batch_size
        default = {"score": 0.0, "response": "", "domain": ""}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            worker_id = 0
            for i in range(batch_size):
                f_with = executor.submit(_run_qa_standalone, (
                    questions[i], experiences_list[i], data_sources[i],
                    ground_truths[i], extra_infos[i], worker_id
                ))
                futures[f_with] = (i, "with")
                worker_id += 1

                f_without = executor.submit(_run_qa_standalone, (
                    questions[i], None, data_sources[i],
                    ground_truths[i], extra_infos[i], worker_id
                ))
                futures[f_without] = (i, "without")
                worker_id += 1

            for future in as_completed(futures):
                idx, mode = futures[future]
                try:
                    result = future.result()
                except:
                    result = default.copy()
                if mode == "with":
                    results_with[idx] = result
                else:
                    results_without[idx] = result

        # Compute rewards
        qa_results_for_graph = []
        for i in range(batch_size):
            score_with = results_with[i]['score']
            score_without = results_without[i]['score']

            utility = score_with - score_without
            generation = score_with
            if USE_UTILITY_SCORE and USE_GENERATION_SCORE:
                reward = utility + generation
            elif USE_UTILITY_SCORE:
                reward = utility
            else:
                reward = generation

            vrl = valid_response_lengths[i]
            if vrl > 0:
                reward_tensor[i, vrl - 1] = reward

            self._record_eval(data_sources[i], score_with > 0, score_without > 0)

            if random.random() < 0.1:
                print(f"[QA] source={data_sources[i]}, with={score_with:.0f}, "
                      f"without={score_without:.0f}, reward={reward:.2f}")

            if not self.val_only and self.graph_updater:
                qa_results_for_graph.append({
                    'question': questions[i][:300],
                    'response': results_with[i].get('response', ''),
                    'domain': results_with[i].get('domain', 'qa'),
                    'score': score_with,
                    'env_name': 'qa',
                })

        if qa_results_for_graph and self.graph_updater:
            self.graph_updater.submit(qa_results_for_graph)

        # Update bandit rewards
        if not self.val_only:
            try:
                graph_url = self._graph_url
                per_example_nodes_raw = data.non_tensor_batch.get('retrieved_node_indices', None)
                if per_example_nodes_raw is not None:
                    per_example_nodes = [list(n) if hasattr(n, '__len__') else [] for n in per_example_nodes_raw]
                    episode_node_indices = []
                    episode_rewards = []
                    for i in range(batch_size):
                        node_ids = per_example_nodes[i] if i < len(per_example_nodes) else []
                        if len(node_ids) > 0:
                            r = (results_with[i]['score'] - results_without[i]['score']) + results_with[i]['score']
                            episode_node_indices.append(list(node_ids))
                            episode_rewards.append(r)
                    if episode_node_indices:
                        import requests as _req
                        _req.post(f"{graph_url}/update_rewards",
                                  json={"node_indices": episode_node_indices, "rewards": episode_rewards},
                                  timeout=10)
            except Exception as e:
                print(f"[WARN] Graph utility update failed: {e}")

        n_correct_with = sum(1 for r in results_with if r['score'] > 0)
        n_correct_without = sum(1 for r in results_without if r['score'] > 0)
        mean_reward = reward_tensor.sum().item() / max(batch_size, 1)
        # Per-source summary
        from collections import defaultdict
        src_stats = defaultdict(lambda: [0, 0, 0])  # [with_correct, without_correct, count]
        for i in range(batch_size):
            src = data_sources[i]
            src_stats[src][0] += int(results_with[i]['score'] > 0)
            src_stats[src][1] += int(results_without[i]['score'] > 0)
            src_stats[src][2] += 1
        src_str = " | ".join(f"{s}:{v[0]}/{v[1]}/{v[2]}" for s, v in sorted(src_stats.items()))
        print(f"[QA Reward] Done. correct_with={n_correct_with}/{batch_size}, "
              f"correct_without={n_correct_without}/{batch_size}, mean_reward={mean_reward:.3f} [{src_str}]")
        return reward_tensor

    def _extract_experiences(self, sequences_str: str) -> str:
        """Extract important experiences from Copilot's search trajectory."""
        info_blocks = []
        important_infos = []

        for match in re.finditer(r'<information>(.*?)</information>', sequences_str, re.DOTALL):
            info_blocks.append({'position': match.start(), 'content': match.group(1), 'processed': False})

        for match in re.finditer(r'<important_info>(.*?)</important_info>', sequences_str, re.DOTALL):
            try:
                numbers = re.findall(r'\d+', match.group(1))
                ids = [int(n) for n in numbers if 1 <= int(n) <= 8]
                ids = list(dict.fromkeys(ids))
            except:
                ids = []
            important_infos.append({'position': match.start(), 'important_ids': ids})

        all_experiences = []
        seen = set()

        # Match important_info to closest preceding info block
        for imp in important_infos:
            closest = None
            for block in info_blocks:
                if not block['processed'] and block['position'] < imp['position']:
                    closest = block
            if closest:
                closest['processed'] = True
                doc_pattern = re.compile(r'Doc\s*(\d+)\s*\(\s*Title\s*:\s*([^)]+)\)\s*(.*?)(?=Doc\s*\d+|$)', re.DOTALL | re.IGNORECASE)
                for doc_match in doc_pattern.finditer(closest['content']):
                    doc_id = int(doc_match.group(1))
                    text = doc_match.group(3).strip()
                    if doc_id in imp['important_ids'] and text not in seen:
                        seen.add(text)
                        all_experiences.append(text)

        # Include unprocessed blocks (initial retrieval)
        for block in info_blocks:
            if not block['processed']:
                doc_pattern = re.compile(r'Doc\s*(\d+)\s*\(\s*Title\s*:\s*([^)]+)\)\s*(.*?)(?=Doc\s*\d+|$)', re.DOTALL | re.IGNORECASE)
                for doc_match in doc_pattern.finditer(block['content']):
                    text = doc_match.group(3).strip()
                    if text not in seen:
                        seen.add(text)
                        all_experiences.append(text)

        return "\n---\n".join(all_experiences) if all_experiences else ""

    def save_all_output_sequences(self):
        """In evaluation mode, print and dump the per-source success rate / steps."""
        if not self.val_only or not self._eval_stats:
            return
        summary = {}
        print(f"\n[Eval] {'source':24s} {'w/ exp':>8s} {'w/o exp':>8s} {'steps w/':>9s} {'n':>5s}")
        for src, st in sorted(self._eval_stats.items()):
            n = max(st['n'], 1)
            summary[src] = {
                'n': st['n'],
                'success_with': st['success_with'] / n,
                'success_without': st['success_without'] / n,
                'avg_steps_with': st['steps_with'] / n,
                'avg_steps_without': st['steps_without'] / n,
            }
            print(f"[Eval] {src:24s} {summary[src]['success_with']:>8.3f} {summary[src]['success_without']:>8.3f} "
                  f"{summary[src]['avg_steps_with']:>9.1f} {st['n']:>5d}")
        with open(os.path.join(self.output_context_dir, 'eval_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)



# ============================================================================
# Main
# ============================================================================

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(
            address=None,
            _temp_dir=os.environ.get("RAY_TMPDIR"),
            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
        )
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # Define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    from expharness.utils.llm_api import AGENT_API, get_model_name
    env_name = getattr(getattr(config, 'env', None), 'name', 'alfworld')
    print(f"[ExpHarness] env={env_name}, executor={AGENT_API}:{get_model_name()}")
    if env_name == 'alfworld':
        from expharness.envs.alfworld_agent import get_solvable_game_files
        eval_split = getattr(getattr(config, 'eval', None), 'split', 'train')
        games = get_solvable_game_files(split=eval_split)
        print(f"[ALFWorld] Found {len(games)} solvable games (split={eval_split}).")

    # Create experience graph updater
    graph_updater = None
    graph_update_cfg = getattr(config, 'graph_update', None)
    graph_url = getattr(graph_update_cfg, 'graph_url', None) if graph_update_cfg else None
    if graph_url is None:
        graph_url = config.retriever.url.rsplit('/retrieve', 1)[0]
    if graph_update_cfg and getattr(graph_update_cfg, 'enable', False):
        graph_updater = ExperienceGraphUpdater(
            graph_url=graph_url,
            max_experiences_per_batch=getattr(graph_update_cfg, 'max_per_batch', 10),
        )

    reward_fn = ExpHarnessRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        graph_updater=graph_updater,
        output_context_dir=config.output_context_dir,
        graph_url=graph_url,
    )
    val_reward_fn = ExpHarnessRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        val_only=True,
        output_context_dir=config.output_context_dir,
        graph_url=graph_url,
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        use_generation_score=USE_GENERATION_SCORE,
        use_utility_score=USE_UTILITY_SCORE,
    )
    trainer.init_workers()
    trainer.fit()

    # Save experience graph
    if graph_update_cfg and getattr(graph_update_cfg, 'enable', False):
        graph_url = getattr(graph_update_cfg, 'graph_url', 'http://127.0.0.1:8001')
        save_path = os.path.join(config.trainer.default_local_dir, 'final_graph.json')
        try:
            import requests as _req
            resp = _req.post(f"{graph_url}/save", json={"path": save_path}, timeout=30)
            print(f"[Graph] Saved final graph to {save_path}: {resp.json()}")
        except Exception as e:
            print(f"[Graph] Failed to save final graph: {e}")

    # Save caches and output sequences
    reward_fn.zeroshot_cache.save()
    reward_fn.save_all_output_sequences()
    val_reward_fn.save_all_output_sequences()


if __name__ == '__main__':
    main()

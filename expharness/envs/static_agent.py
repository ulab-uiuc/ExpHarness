"""
Static-task executor for ExpHarness (QA / math reasoning / code generation).

Single turn: question + retrieved experiences -> frozen executor answer -> scored
against the ground truth with the domain-specific checker in
`expharness.rewards.static_reward`.
"""

import os
import re
import json
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ExpSuite static benchmarks, grouped into three domains


# Domain detection from data_source
DOMAIN_MAP = {
    'gsm8k': 'math',
    'gsm_symbolic_main': 'math',
    'math': 'math',
    'arc_c': 'qa',
    'commonsenseqa': 'qa',
    'mmlu': 'qa',
    'obqa': 'qa',
    'gpqa': 'qa',
    'humaneval_plus': 'coding',
    'mbpp_plus': 'coding',
}


def _build_qa_prompt(question: str, experiences: Optional[str] = None) -> str:
    """Build prompt for QA agent with optional retrieved experiences."""
    lines = []

    if experiences:
        lines.append("Here are some relevant reasoning examples that may help:\n")
        for idx, exp in enumerate(experiences.split("\n---\n")[:10]):
            exp = exp.strip()
            if exp:
                lines.append(f"{idx+1}. {exp}")
        lines.append("")

    lines.append(question)
    return "\n".join(lines)


def _run_qa_standalone(args_tuple) -> Dict:
    """
    Run a single QA episode in an independent process.
    Args: (question, experiences, data_source, ground_truth, extra_info)
    or:   (question, experiences, data_source, ground_truth, extra_info, worker_id)

    Returns: dict with 'score', 'response', 'domain'
    """
    import os, sys
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if len(args_tuple) == 6:
        question, experiences, data_source, ground_truth, extra_info, worker_id = args_tuple
        from expharness.utils.llm_api import bind_worker
        bind_worker(worker_id)
    else:
        question, experiences, data_source, ground_truth, extra_info = args_tuple

    from expharness.utils.llm_api import llm_call

    domain = DOMAIN_MAP.get(data_source, 'qa')

    # Build prompt with experiences
    prompt = _build_qa_prompt(question, experiences)

    # Call LLM
    max_tokens = 1024 if domain == 'coding' else 512
    response = llm_call(prompt, max_tokens=max_tokens, temperature=0.0)

    # Compute reward
    from expharness.rewards.static_reward import (
        math_reward, qa_reward, code_reward
    )

    # Parse ground_truth
    if isinstance(ground_truth, str):
        try:
            gt_dict = json.loads(ground_truth)
        except:
            gt_dict = {"ground_truth": ground_truth}
    elif isinstance(ground_truth, dict):
        gt_dict = ground_truth
    else:
        gt_dict = {"ground_truth": str(ground_truth)}

    # Parse extra_info for coding test cases
    if isinstance(extra_info, str):
        try:
            extra_dict = json.loads(extra_info)
        except:
            extra_dict = {}
    elif isinstance(extra_info, dict):
        extra_dict = extra_info
    else:
        extra_dict = {}

    gt = gt_dict.get('ground_truth', '')
    # For math dataset, ground_truth might be "math_reward" placeholder — use extra_info['answer'] instead
    if gt == 'math_reward' or not gt:
        gt = str(extra_dict.get('answer', extra_dict.get('original_answer', '')))

    if domain == 'math':
        score = math_reward(response, gt)
    elif domain == 'qa':
        score = qa_reward(response, gt)
    elif domain == 'coding':
        import numpy as np
        test_cases = extra_dict.get('test_list', [])
        # Convert numpy array to list if needed
        if isinstance(test_cases, np.ndarray):
            test_cases = test_cases.tolist()
        if not test_cases:
            test_cases = gt_dict.get('test_list', [])
            if isinstance(test_cases, np.ndarray):
                test_cases = test_cases.tolist()
        score = code_reward(response, test_cases) if test_cases else 0.0
    else:
        score = 0.0

    return {
        'score': score,
        'response': response[:500],
        'domain': domain,
        'data_source': data_source,
    }


def evaluate_batch(questions, experiences_list, data_sources, ground_truths, extra_infos,
                   max_workers=16):
    """
    Evaluate a batch of QA questions with and without experiences.

    Returns: (results_with, results_without)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results_with = [None] * len(questions)
    results_without = [None] * len(questions)
    default = {"score": 0.0, "response": "", "domain": ""}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        worker_id = 0

        for i in range(len(questions)):
            # WITH experiences
            f_with = executor.submit(_run_qa_standalone, (
                questions[i], experiences_list[i], data_sources[i],
                ground_truths[i], extra_infos[i], worker_id
            ))
            futures[f_with] = (i, "with")
            worker_id += 1

            # WITHOUT experiences (no experience baseline)
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
            except Exception as e:
                result = default.copy()

            if mode == "with":
                results_with[idx] = result
            else:
                results_without[idx] = result

    return results_with, results_without

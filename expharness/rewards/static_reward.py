"""
Answer checkers for the ExpSuite static benchmarks.

- math_reward:  GSM8K, GSM-Symbolic, MATH (final answer after `####` or in \\boxed{})
- qa_reward:    MMLU, CommonsenseQA, OBQA, ARC-C, GPQA (multiple-choice letter)
- code_reward:  HumanEval+, MBPP+ (generated code executed against unit tests)

All checkers return a binary score in {0.0, 1.0}.
"""

import re
import sys
import signal
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Answer Extraction (Unified #### format)
# ============================================================================

def extract_answer_after_hashtag(text: str) -> Optional[str]:
    """Extract answer after #### marker.

    Handles edge cases like:
    - '#### 30.1 ####' -> '30.1'
    - '#### 9\\n#####' -> '9'
    - Multiple #### markers -> finds first valid answer
    """
    if '####' not in text:
        return None

    # Use regex to find answer after #### (not followed immediately by # or newline)
    # This handles cases like "#### answer ####" correctly
    import re

    # Try to find #### followed by non-empty content (not # or whitespace-only)
    match = re.search(r'####\s*([^#\n][^\n]*?)(?:\s*####|\s*$)', text)
    if match:
        answer = match.group(1).strip().rstrip('.')
        if answer:
            return answer

    # Fallback: split and find first non-empty part
    parts = text.split('####')
    for part in parts[1:]:  # Skip first part (before any ####)
        answer = part.strip().split('\n')[0].strip()
        # Skip if it's just # symbols or empty
        if answer and not answer.startswith('#'):
            answer = answer.rstrip('.')
            return answer

    return None


# ============================================================================
# MATH Reward
# ============================================================================

def normalize_math_answer(answer: str) -> str:
    """Normalize mathematical answer for comparison."""
    if not answer:
        return ""

    # Remove LaTeX formatting
    answer = answer.replace('\\', '')
    answer = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\(?:frac)\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)

    # Remove spaces and convert to lowercase
    answer = answer.replace(' ', '').lower()

    # Remove common units and formatting
    answer = re.sub(r'(?:dollars?|cents?|\$|%|degrees?|°)', '', answer)

    # Try to extract number
    number_match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer)
    if number_match:
        return number_match.group().replace(',', '')

    return answer


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format."""
    # Match \boxed{...} with nested braces
    match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if match:
        return match.group(1).strip()
    # Fallback: try simpler pattern
    match = re.search(r'\\boxed\{(.+?)\}', text)
    if match:
        return match.group(1).strip()
    return None


def math_reward(response: str, ground_truth: str) -> float:
    """
    Compute reward for MATH problems.

    Args:
        response: Model response
        ground_truth: Ground truth answer (may include \boxed{})

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Extract predicted answer - try #### first, then \boxed{}
    pred = extract_answer_after_hashtag(response)
    if pred is None:
        # Fallback: try to extract from \boxed{} in response
        pred = extract_boxed_answer(response)
    if pred is None:
        return 0.0

    # Extract ground truth (handle \boxed{} format)
    gt = ground_truth
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', gt)
    if boxed_match:
        gt = boxed_match.group(1)

    # Normalize and compare
    pred_norm = normalize_math_answer(pred)
    gt_norm = normalize_math_answer(gt)

    return 1.0 if pred_norm == gt_norm else 0.0


# ============================================================================
# QA (ARC-Challenge, CommonsenseQA) Reward
# ============================================================================

def qa_reward(response: str, ground_truth: str) -> float:
    """
    Compute reward for QA problems (multiple choice).

    Supports both ARC-Challenge (A-D) and CommonsenseQA (A-E).
    Uses EXACT same patterns as GNN training (commonsenseqa_reward in train_gnn_from_cache.py).

    Args:
        response: Model response
        ground_truth: Ground truth answer (A, B, C, D, or E)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    gt = ground_truth.strip().upper()
    if not gt or gt not in 'ABCDE':
        return 0.0

    response_upper = response.upper()

    # Pattern 1: "The answer is X" or "correct answer is X"
    match = re.search(r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*([A-E])', response_upper)
    if match and match.group(1) == gt:
        return 1.0

    # Pattern 2: "#### X" format (primary format)
    match = re.search(r'####\s*([A-E])', response_upper)
    if match and match.group(1) == gt:
        return 1.0

    # Pattern 3: Letter at start like "A." or "A)" (for direct answers)
    match = re.search(r'^([A-E])\s*[.):]\s', response_upper.strip())
    if match and match.group(1) == gt:
        return 1.0

    # Pattern 4: "I choose X" or "I select X" or "I pick X"
    match = re.search(r'(?:I\s+)?(?:CHOOSE|SELECT|PICK)\s+([A-E])', response_upper)
    if match and match.group(1) == gt:
        return 1.0

    # No more patterns - must match exactly with GNN training function
    # Pattern 5 was removed from GNN training due to false positives
    return 0.0


# ============================================================================
# Code (MBPP) Reward
# ============================================================================

@contextmanager
def timeout(seconds):
    """Context manager for timeout using SIGALRM.

    WARNING: This doesn't work well in Ray workers. Use _safe_exec_with_timeout instead.
    """
    def signal_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================================
# Safe Code Execution with Multiprocessing (Ray-compatible)
# ============================================================================

def _run_code_in_process(code: str, fn_name: str, inputs, expected, result_queue):
    """Run code in isolated process. Used by multiprocessing."""
    import sys
    import io

    # Set recursion limit in subprocess
    sys.setrecursionlimit(500)

    try:
        if fn_name:
            # Function call mode
            exec_globals = {}
            exec(code, exec_globals)

            func = exec_globals.get(fn_name)
            if func is None:
                # Try to find first callable
                for name, obj in exec_globals.items():
                    if callable(obj) and not name.startswith('_'):
                        func = obj
                        break

            if func is None:
                result_queue.put(0)
                return

            # Call function
            if isinstance(inputs, list):
                result = func(*inputs)
            else:
                result = func(inputs)

            # Compare result
            exp = expected[0] if isinstance(expected, list) and len(expected) == 1 else expected
            if result is not None and (str(result).strip() == str(exp).strip() or result == exp):
                result_queue.put(1)
            else:
                result_queue.put(0)
        else:
            # stdin/stdout mode
            old_stdin = sys.stdin
            old_stdout = sys.stdout

            try:
                if isinstance(inputs, list):
                    inp_str = '\n'.join(str(line) for line in inputs)
                else:
                    inp_str = str(inputs) if inputs is not None else ''

                sys.stdin = io.StringIO(inp_str)
                captured = io.StringIO()
                sys.stdout = captured

                exec(code, {'__builtins__': __builtins__})

                sys.stdin = old_stdin
                sys.stdout = old_stdout

                actual = captured.getvalue().strip()

                if isinstance(expected, list):
                    exp_str = '\n'.join(str(line) for line in expected)
                else:
                    exp_str = str(expected) if expected is not None else ''
                exp_str = exp_str.strip()

                if actual == exp_str:
                    result_queue.put(1)
                else:
                    result_queue.put(0)
            finally:
                sys.stdin = old_stdin
                sys.stdout = old_stdout
    except:
        result_queue.put(0)


def _safe_exec_with_timeout(code: str, fn_name: str, inputs, expected, timeout_seconds: float = 2.0) -> int:
    """Execute code in isolated process with timeout. Returns 1 if passed, 0 otherwise.

    This is Ray-compatible as it uses multiprocessing instead of signals.
    """
    import multiprocessing

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_code_in_process,
        args=(code, fn_name, inputs, expected, result_queue)
    )
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
        return 0

    try:
        return result_queue.get_nowait()
    except:
        return 0


def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from response.

    Improved extraction logic inspired by HumanEval:
    - Handle various markdown formats
    - Extract complete function definitions
    - Support both #### marker and raw code
    """
    # Find code after ####
    if '####' in response:
        after_hash = response.split('####')[-1]

        # Check if it's wrapped in code blocks (handle both ```python and ```)
        code_match = re.search(r'```(?:python)?\s*(.*?)```', after_hash, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            # Remove any leading/trailing newlines but keep indentation
            return code

        # Check for def statement - extract until the function ends
        # Match from 'def' to the last line with proper indentation
        def_match = re.search(r'(def\s+\w+\s*\([^)]*\)\s*:.*?)(?=\ndef\s|\n[^\s\n]|\Z)', after_hash, re.DOTALL)
        if def_match:
            return def_match.group(1).strip()

        # Return everything after #### if it looks like code
        after_hash = after_hash.strip()
        if after_hash.startswith('def ') or 'return' in after_hash:
            # Try to extract just the function
            lines = after_hash.split('\n')
            code_lines = []
            in_function = False
            for line in lines:
                if line.strip().startswith('def '):
                    in_function = True
                if in_function:
                    code_lines.append(line)
                    # Check if we've exited the function (non-indented non-empty line after starting)
                    if len(code_lines) > 1 and line and not line[0].isspace() and not line.strip().startswith('def '):
                        code_lines.pop()  # Remove the non-function line
                        break
            if code_lines:
                return '\n'.join(code_lines)
            return after_hash

    # Fallback: find code blocks in entire response
    code_match = re.search(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Fallback: find def statements with improved extraction
    def_match = re.search(r'(def\s+\w+\s*\([^)]*\)\s*:.*?)(?=\ndef\s|\n[^\s\n]|\Z)', response, re.DOTALL)
    if def_match:
        return def_match.group(1).strip()

    return None


def _run_code_with_tests(code: str, test_cases: list, result_queue):
    """Run code with test cases in isolated process. Used by multiprocessing."""
    import sys
    sys.setrecursionlimit(500)

    try:
        exec_globals = {}
        # Execute the code definition
        exec(code, exec_globals)

        # Run all test cases
        for test in test_cases:
            try:
                exec(test, exec_globals)
            except AssertionError:
                result_queue.put(0)
                return
            except Exception:
                result_queue.put(0)
                return

        # All tests passed
        result_queue.put(1)
    except Exception:
        result_queue.put(0)


def code_reward(response: str, test_cases: list, timeout_seconds: int = 5) -> float:
    """
    Compute reward for coding problems by running test cases (assert format).

    Uses multiprocessing for timeout (Ray-compatible, unlike signal.alarm).

    Args:
        response: Model response containing code
        test_cases: List of test case strings (assertions)
        timeout_seconds: Timeout for code execution

    Returns:
        1.0 if all tests pass, 0.0 otherwise
    """
    import multiprocessing

    # Extract code
    code = extract_code_from_response(response)
    if code is None:
        return 0.0

    # Run code execution in subprocess with timeout
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_code_with_tests,
        args=(code, test_cases, result_queue)
    )
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
        logger.debug("Code execution timed out")
        return 0.0

    try:
        result = result_queue.get_nowait()
        return float(result)
    except Exception:
        return 0.0

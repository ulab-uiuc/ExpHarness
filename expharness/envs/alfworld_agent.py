"""
ALFWorld executor for ExpHarness.

Runs the frozen executor LLM in the real ALFWorld (TextWorld) environment, optionally
conditioned on experiences retrieved from the experience graph. Episodes run in a
ProcessPoolExecutor; each process registers its own TextWorld game.

Requires the official `alfworld` package and its data (`alfworld-download`); set
ALFWORLD_DATA if the data is not in the default location.
"""

import os
import sys
import re
import time
import types
import json
import uuid
import threading
import yaml
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from expharness.utils.llm_api import llm_call

# ============================================================================
# ALFWorld setup
# ============================================================================

_alfworld_setup_done = False


def _setup_alfworld():
    global _alfworld_setup_done
    if _alfworld_setup_done:
        return
    os.environ.setdefault('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
    import alfworld  # noqa: F401  (fail early if the package is missing)
    _alfworld_setup_done = True


def resolve_gamefile(gamefile: str) -> str:
    """Resolve a gamefile stored relative to the ALFWorld data root
    (e.g. 'json_2.1.1/valid_seen/.../game.tw-pddl') to an absolute path under
    $ALFWORLD_DATA. Absolute paths that exist are returned unchanged."""
    if not gamefile or os.path.exists(gamefile):
        return gamefile
    marker = "json_2.1.1"
    if marker in gamefile:
        root = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
        candidate = os.path.join(root, gamefile[gamefile.index(marker):])
        if os.path.exists(candidate):
            return candidate
    return gamefile


def set_api_keys(keys):
    """Legacy compatibility — keys are read from the environment (see llm_api)."""
    pass


# ============================================================================
# Collect solvable game files
# ============================================================================

_cached_game_files = {}


def get_solvable_game_files(split="train") -> List[Tuple[str, str]]:
    """Get list of (task_type, gamefile) for solvable games. Cached per split."""
    global _cached_game_files
    if split in _cached_game_files:
        return _cached_game_files[split]

    _setup_alfworld()
    from alfworld.info import ALFWORLD_DATA

    TASK_TYPES = {
        "pick_and_place_simple", "look_at_obj_in_light",
        "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
    }

    if split == "train":
        data_path = os.path.join(ALFWORLD_DATA, "json_2.1.1", "train")
    elif split == "eval_in_distribution":
        data_path = os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_seen")
    else:
        data_path = os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_unseen")

    results = []
    for root, _, files in os.walk(data_path):
        if "traj_data.json" not in files:
            continue
        if "movable" in root or "Sliced" in root:
            continue
        game_path = os.path.join(root, "game.tw-pddl")
        if not os.path.exists(game_path):
            continue
        try:
            with open(game_path) as f:
                gd = json.load(f)
            if not gd.get("solvable", False):
                continue
            with open(os.path.join(root, "traj_data.json")) as f:
                td = json.load(f)
            if td.get("task_type") not in TASK_TYPES:
                continue
            results.append((td["task_type"], game_path))
        except:
            continue

    _cached_game_files[split] = results
    return results


# ============================================================================
# Helpers
# ============================================================================

def _unwrap_single(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        first = value[0]
        if first is None:
            return default
        if isinstance(first, (list, tuple)):
            return first[0] if first else default
        return first
    return value


def _extract_objective(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"Your task is to:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().splitlines()[0].strip().rstrip(".")


def _extract_admissible(info) -> List[str]:
    if isinstance(info, list) and info:
        info = info[0]
    if isinstance(info, dict):
        commands = info.get("admissible_commands")
        if isinstance(commands, list) and commands:
            if isinstance(commands[0], list):
                commands = commands[0]
            return [c for c in commands if c != 'help']
    return []


def _parse_action_response(response: str, admissible: List[str]) -> Tuple[str, bool]:
    text = (response or "").strip()
    if not text:
        return (admissible[0] if admissible else "look"), False
    text = re.sub(r'^\s*action\s*[:=\-]\s*', '', text, flags=re.IGNORECASE)
    line = text.splitlines()[0].strip().strip('"').strip("'").strip("`")
    lowered = text.lower()
    if admissible:
        for cmd in admissible:
            if line.lower() == str(cmd).lower():
                return cmd, True
        for cmd in admissible:
            if str(cmd).lower() in lowered:
                return cmd, True
        return admissible[0], False
    return line if line else "look", False


def _build_action_prompt(objective, trajectory_text, inventory, admissible, experiences=None):
    lines = [
        "You are controlling a text-based ALFWorld environment.",
        "Your job: choose the NEXT action as ONE text command.",
        "Output ONLY the command string, with no extra text.",
        "You MUST choose an action from the admissible actions list and copy it EXACTLY.",
    ]
    if objective:
        lines += ["", f"Goal: {objective}"]
    if experiences:
        lines += ["", "Retrieved procedural tips:"]
        if isinstance(experiences, str):
            for idx, exp in enumerate(experiences.split("\n---\n")[:10]):
                lines.append(f"{idx+1}. {exp.strip()}")
        elif isinstance(experiences, list):
            for idx, exp in enumerate(experiences[:10]):
                lines.append(f"{idx+1}. {exp}")
    # Truncate trajectory to last 20 lines to avoid exceeding API token limits
    traj_lines = trajectory_text.strip().splitlines() if trajectory_text else []
    if len(traj_lines) > 20:
        traj_lines = ["... (earlier steps omitted) ..."] + traj_lines[-20:]
    lines += ["", "Interaction history so far:", "\n".join(traj_lines) if traj_lines else "(empty)"]
    if inventory and inventory.strip() and inventory.strip().lower() not in {"none", "null", "(empty)"}:
        lines += ["", "Inventory:", inventory.strip()]
    if admissible:
        lines += ["", "Admissible actions (choose exactly ONE and copy it verbatim):"]
        for cmd in admissible:
            lines.append(f"- {cmd}")
        lines += ["", "Now output exactly one line: the chosen action (must match one item above)."]
    return "\n".join(lines)


# ============================================================================
# Run single episode (standalone, for ProcessPoolExecutor)
# ============================================================================

def _run_episode_standalone(args_tuple) -> Dict:
    """
    Run a single ALFWorld episode in an independent process.
    Args: (gamefile, experiences, max_steps) or (gamefile, experiences, max_steps, worker_id)
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Don't load GPU in worker processes

    if len(args_tuple) == 4:
        gamefile, experiences, max_steps, worker_id = args_tuple
        # Pin this worker to one API key to spread rate limits
        from expharness.utils.llm_api import bind_worker
        bind_worker(worker_id)
    else:
        gamefile, experiences, max_steps = args_tuple

    _setup_alfworld()
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    request_infos = textworld.EnvInfos(
        feedback=True, description=True, inventory=True,
        admissible_commands=True, objective=True, extras=["gamefile"]
    )
    wrappers = [AlfredDemangler(), AlfredInfos]
    env_id = textworld.gym.register_games(
        [gamefile], request_infos, batch_size=1, auto_reset=False,
        max_episode_steps=max_steps, asynchronous=False,
        name=f"alfworld-{uuid.uuid4().hex}", wrappers=wrappers
    )
    env = textworld.gym.make(env_id)

    try:
        # Reset with timeout
        reset_result = {}
        def _do_reset():
            try:
                reset_result["value"] = env.reset()
            except Exception as exc:
                reset_result["error"] = exc

        t = threading.Thread(target=_do_reset, daemon=True)
        t.start()
        t.join(120.0)
        if t.is_alive():
            return {"error": "reset timeout", "won": False, "score": 0.0, "steps": 0, "trajectory": []}
        if "error" in reset_result:
            raise reset_result["error"]

        obs_batch, info_batch = reset_result["value"]
        obs = _unwrap_single(obs_batch, "")
        info = _unwrap_single(info_batch, {})
        objective = _extract_objective(str(obs))

        trajectory_lines = [str(obs).strip()] if obs else []
        won = False
        valid_actions = 0
        history = []

        for step_idx in range(1, max_steps + 1):
            admissible = _extract_admissible(info)
            inventory = ""
            if isinstance(info, dict):
                inv = info.get("inventory") or info.get("inv") or ""
                if isinstance(inv, list):
                    inv = inv[0] if inv else ""
                inventory = str(inv or "")

            # Truncate trajectory to last 10 steps to keep prompt short
            max_history_lines = 20  # ~10 steps × 2 lines (ACTION + OBS)
            truncated_trajectory = trajectory_lines[-max_history_lines:] if len(trajectory_lines) > max_history_lines else trajectory_lines
            prompt = _build_action_prompt(
                objective=objective,
                trajectory_text="\n".join(truncated_trajectory),
                inventory=inventory,
                admissible=admissible,
                experiences=experiences,
            )

            response = llm_call(prompt, max_tokens=32)
            action, valid = _parse_action_response(response, admissible)
            if valid:
                valid_actions += 1

            obs_batch, scores, dones, infos = env.step([action])
            obs = _unwrap_single(obs_batch, "")
            info = _unwrap_single(infos, {})
            score = _unwrap_single(scores, 0.0)
            done = bool(_unwrap_single(dones, False))

            trajectory_lines.append(f"ACTION: {action}")
            trajectory_lines.append(f"OBSERVATION: {str(obs).strip()}")

            history.append({"step": step_idx, "action": action, "obs": str(obs)[:200], "valid": valid})

            won_info = info.get("won", False)
            if isinstance(won_info, (list, tuple)):
                won_info = won_info[0] if won_info else False
            won = bool(won_info) or (done and float(score or 0) >= 1.0)
            if done:
                break

        goal_progress = 0.0
        if isinstance(info, dict):
            gp = info.get("goal_condition_success_rate", 0.0)
            if isinstance(gp, (list, tuple)):
                gp = gp[0] if gp else 0.0
            goal_progress = float(gp or 0.0)

        return {
            "won": won,
            "score": 1.0 if won else goal_progress,
            "steps": len(history),
            "valid_ratio": valid_actions / max(len(history), 1),
            "task": objective,
            "trajectory": history,
        }

    except Exception as e:
        return {"error": str(e), "won": False, "score": 0.0, "steps": 0, "trajectory": []}
    finally:
        try:
            env.close()
        except:
            pass


# ============================================================================
# Batch evaluation (ProcessPoolExecutor)
# ============================================================================

def evaluate_batch(
    tasks: List[str],
    experiences_list: List[Optional[str]],
    max_steps: int = 50,
    max_workers: int = 10,
    game_files: List[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Evaluate tasks WITH and WITHOUT experiences using ProcessPoolExecutor.
    Each episode runs in its own process with its own env.

    Args:
        tasks: task descriptions (for logging)
        experiences_list: experiences from Copilot
        max_steps: max steps per episode
        max_workers: parallel processes
        game_files: specific game files to use (if None, randomly sampled)

    Returns:
        (results_with_exp, results_without_exp)
    """
    import random

    batch_size = len(tasks)

    # Get game files if not provided
    # IMPORTANT: with/without must use the SAME game for fair utility comparison
    if game_files is None:
        all_games = get_solvable_game_files()
        sampled = random.sample(all_games, min(batch_size, len(all_games)))
        game_files_with = [gf for _, gf in sampled[:batch_size]]
    else:
        game_files_with = game_files
    game_files_without = game_files_with  # Same games for fair comparison

    # Build args for with/without
    args_with = [(game_files_with[i], experiences_list[i], max_steps) for i in range(batch_size)]
    args_without = [(game_files_without[i], None, max_steps) for i in range(batch_size)]

    results_with = [None] * batch_size
    results_without = [None] * batch_size
    default = {"won": False, "score": 0.0, "steps": 0, "trajectory": [], "task": ""}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(batch_size):
            f_with = executor.submit(_run_episode_standalone, args_with[i])
            futures[f_with] = (i, "with")
            f_without = executor.submit(_run_episode_standalone, args_without[i])
            futures[f_without] = (i, "without")

        for future in as_completed(futures):
            idx, mode = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"[ALFWorldAgent] Episode error: {e}")
                result = default.copy()

            if "error" in result:
                result = default.copy()

            if mode == "with":
                results_with[idx] = result
            else:
                results_without[idx] = result

    for i in range(batch_size):
        if results_with[i] is None:
            results_with[i] = default.copy()
        if results_without[i] is None:
            results_without[i] = default.copy()

    return results_with, results_without

"""
Prepare ALFWorld copilot training data as parquet files for ExpHarness.

Each row contains:
- prompt: Step 0 search prompt (task + initial obs + search instructions)
- data_source: "alfworld"
- reward_model: ground_truth with task info
- extra_info: gamefile, task_type, etc.

The actual environment interaction happens live during training.
Runs each gamefile through the real ALFWorld env to get true initial observations.
"""

import argparse
import json
import os
import re
import sys
import types
import uuid
import threading
import requests

import datasets
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from expharness.envs.alfworld_prompts import STEP0_SEARCH_TEMPLATE, STEP0_SEARCH_EXAMPLE


# ============================================================================
# ALFWorld env setup
# ============================================================================

def _setup_alfworld():
    os.environ.setdefault('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
    import alfworld  # noqa: F401


def get_real_initial_state(gamefile, max_steps=50):
    """Get the real initial observation and admissible actions from ALFWorld env."""
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
        name=f"alfworld-prep-{uuid.uuid4().hex}", wrappers=wrappers
    )
    env = textworld.gym.make(env_id)

    try:
        obs_batch, info_batch = env.reset()
        obs = obs_batch[0] if isinstance(obs_batch, (list, tuple)) else str(obs_batch)
        info = info_batch[0] if isinstance(info_batch, (list, tuple)) else info_batch

        # Extract objective (natural language task description)
        objective = ""
        match = re.search(r"Your task is to:\s*(.+)", str(obs), re.IGNORECASE | re.DOTALL)
        if match:
            objective = match.group(1).strip().splitlines()[0].strip().rstrip(".")

        # Extract admissible actions
        admissible = []
        if isinstance(info, dict):
            commands = info.get("admissible_commands")
            if isinstance(commands, list) and commands:
                if isinstance(commands[0], list):
                    commands = commands[0]
                admissible = [c for c in commands if c != 'help']

        return {
            "observation": str(obs).strip(),
            "objective": objective,
            "admissible": admissible,
        }
    except Exception as e:
        print(f"Warning: Could not get initial state for {gamefile}: {e}")
        return None
    finally:
        try:
            env.close()
        except:
            pass


def build_step0_prompt(task_description, initial_obs=None, admissible_actions=None):
    """Build the Step 0 retrieval parameter prompt for Copilot."""
    prompt = STEP0_SEARCH_TEMPLATE.format(
        task_description=task_description,
    )
    prompt += STEP0_SEARCH_EXAMPLE
    prompt += f"\n<question>\n{task_description}\n</question>\n"
    return prompt


def detect_task_type(gamefile):
    """Detect ALFWorld task type from gamefile path."""
    task_types = [
        "pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
        "pick_cool_then_place", "look_at_obj_in_light", "pick_two_obj_and_place",
    ]
    for tt in task_types:
        if tt in gamefile:
            return tt
    return "unknown"


def collect_solvable_games(split="train"):
    """Collect solvable game files from ALFWorld data directory."""
    _setup_alfworld()
    from alfworld.info import ALFWORLD_DATA

    TASK_TYPES = {
        "pick_and_place_simple", "look_at_obj_in_light",
        "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
    }

    if split == "train":
        data_path = os.path.join(ALFWORLD_DATA, "json_2.1.1", "train")
    elif split == "valid_seen":
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
            results.append({
                "task_type": td["task_type"],
                "gamefile": game_path,
            })
        except:
            continue

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld_data", default=None,
                        help="Path to ALFWorld game files JSON (gamefile list)")
    parser.add_argument("--output_dir", default="data/alfworld")
    parser.add_argument("--skip_env_init", action="store_true",
                        help="Skip loading real env states (use fallback obs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect game files
    if args.alfworld_data and os.path.exists(args.alfworld_data):
        with open(args.alfworld_data) as f:
            gamefiles = json.load(f)
    else:
        print("Collecting solvable games from ALFWorld data directory...")
        train_games = collect_solvable_games("train")
        val_seen_games = collect_solvable_games("valid_seen")
        val_unseen_games = collect_solvable_games("valid_unseen")
        print(f"Found {len(train_games)} train, {len(val_seen_games)} valid_seen, {len(val_unseen_games)} valid_unseen")

        # Tag with split
        for g in train_games:
            g["split"] = "train"
        for g in val_seen_games:
            g["split"] = "val"
        for g in val_unseen_games:
            g["split"] = "val"
        gamefiles = train_games + val_seen_games + val_unseen_games

    print(f"Total tasks: {len(gamefiles)}")

    # Fallback observation for when env loading fails
    FALLBACK_OBS = "You are in the middle of a room. Looking quickly around you, you see various furniture and objects."
    FALLBACK_ADMISSIBLE = ["go to counter 1", "go to shelf 1", "go to fridge 1", "go to cabinet 1", "go to table 1", "look"]

    # Build dataset
    all_data = []
    n_real_obs = 0
    n_fallback = 0

    for i, item in enumerate(gamefiles):
        if isinstance(item, str):
            gamefile = item
            task_type = detect_task_type(item)
            split = "train"
        else:
            gamefile = item.get("gamefile", "")
            # Fix: use `or` to handle empty string task_type
            task_type = item.get("task_type") or detect_task_type(gamefile)
            split = item.get("split", "train" if i < len(gamefiles) * 0.9 else "val")

        # Get real initial state from ALFWorld env
        initial_obs = FALLBACK_OBS
        admissible = FALLBACK_ADMISSIBLE
        task_description = task_type  # fallback

        if not args.skip_env_init and gamefile and os.path.exists(gamefile):
            state = get_real_initial_state(gamefile)
            if state:
                initial_obs = state["observation"]
                if state["admissible"]:
                    admissible = state["admissible"]
                if state["objective"]:
                    task_description = state["objective"]
                n_real_obs += 1
            else:
                n_fallback += 1
        else:
            n_fallback += 1

        # If task_description is still just a type, try to parse from gamefile path
        if task_description == task_type or not task_description:
            # Convert ID like "look_at_obj_in_light-Pillow-None-DeskLamp-317" to readable form
            basename = os.path.basename(os.path.dirname(os.path.dirname(gamefile))) if gamefile else ""
            task_description = basename if basename else task_type

        prompt = build_step0_prompt(task_description, initial_obs, admissible)

        data = {
            "data_source": "alfworld",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "embodied-reasoning",
            "reward_model": {
                "style": "env",
                "ground_truth": {
                    "task": task_description,
                    "task_type": task_type,
                    # stored relative to $ALFWORLD_DATA so the parquet is portable
                    "gamefile": gamefile[gamefile.index("json_2.1.1"):] if "json_2.1.1" in gamefile else gamefile,
                }
            },
            "extra_info": {
                "split": split,
                "index": i,
                "task_type": task_type,
            }
        }
        all_data.append(data)

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(gamefiles)} (real_obs={n_real_obs}, fallback={n_fallback})")

    print(f"Done. real_obs={n_real_obs}, fallback={n_fallback}")

    # Split and save
    train_data = [d for d in all_data if d["extra_info"]["split"] == "train"]
    val_data = [d for d in all_data if d["extra_info"]["split"] == "val"]

    train_ds = datasets.Dataset.from_list(train_data)
    val_ds = datasets.Dataset.from_list(val_data)

    train_ds.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(args.output_dir, "val.parquet"))

    print(f"Saved {len(train_data)} train, {len(val_data)} val to {args.output_dir}")


if __name__ == "__main__":
    main()

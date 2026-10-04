"""
Build the initial ALFWorld experience graph for ExpHarness.

1. Run N zero-shot episodes with the frozen executor (no experiences)
2. Summarize successful trajectories into skills and failed ones into lessons
3. Insert them into an experience graph (dedup + kNN edges)
4. Offline scoring: re-run the same games with / without retrieved experiences and
   initialize each node's utility with r = (s_with - s_without) + s_with

The executor / summarizer backend is configured through AGENT_API (see expharness/utils/llm_api.py).
"""

import argparse
import json
import os
import re
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concurrent.futures import ProcessPoolExecutor, as_completed
from expharness.envs.alfworld_agent import _run_episode_standalone, get_solvable_game_files
from expharness.utils.llm_api import summarizer_call


def summarize_trajectory(task, trajectory, won):
    """Summarize an ALFWorld trajectory into a reusable skill / lesson."""
    steps = trajectory if isinstance(trajectory, list) else []
    traj_str = "\n".join([
        f"Step {i+1}: ACTION: {s.get('action', '')} -> OBS: {s.get('obs', '')[:150]}"
        for i, s in enumerate(steps[:30])
    ])

    if won:
        prompt = f"""An ALFWorld household task was completed successfully.

Task: {task}
Full trajectory (action -> observation):
{traj_str}

Extract a reusable skill (3-5 sentences). Include:
1. The general task category (e.g. pick_and_place, heat_then_place, clean_then_place, cool_then_place, examine_in_light, pick_two)
2. The concrete step-by-step strategy that worked (e.g. "go to countertop to find the object, then pick it up, then go to sinkbasin to clean it")
3. Common locations where target objects are found (e.g. "soapbar is usually on countertop, bathtubbasin, or shelf")

Be specific and actionable. Use actual object/location types (countertop, sinkbasin, microwave) not abstract placeholders.
Output format: SKILL: [your skill text]"""
    else:
        prompt = f"""An ALFWorld household task failed after all steps.

Task: {task}
Full trajectory (action -> observation):
{traj_str}

Extract a reusable lesson (3-5 sentences). Include:
1. The general task category
2. What specific mistake was made (e.g. "kept visiting wrong locations", "forgot to clean before placing")
3. What the agent should have done differently

Be specific and actionable. Use actual object/location types not abstract placeholders.
Output format: SKILL: [your lesson text]"""

    full_prompt = "You are an expert at analyzing household robot trajectories. Extract specific, actionable lessons.\n\n" + prompt
    response = summarizer_call(full_prompt, max_tokens=300)
    match = re.search(r'SKILL:\s*(.*)', response, re.DOTALL)
    skill_text = match.group(1).strip() if match else response.strip()
    prefix = "[SUCCESS]" if won else "[FAILURE]"
    return f"{prefix} {skill_text}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--output_graph", default="data/alfworld/cold_start_graph.json")
    parser.add_argument("--device", default="cuda", help="Device for the Contriever encoder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Get solvable game files
    all_games = get_solvable_game_files()
    print(f"Found {len(all_games)} solvable games")

    # Sample games for cold start
    sampled = random.sample(all_games, min(args.num_episodes, len(all_games)))
    print(f"Running {len(sampled)} zero-shot episodes (no experiences)...")

    # Run episodes in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for i, (task, gamefile) in enumerate(sampled):
            f = executor.submit(_run_episode_standalone, (gamefile, None, args.max_steps))
            futures[f] = (i, task, gamefile)

        for future in as_completed(futures):
            i, task, gamefile = futures[future]
            try:
                result = future.result()
                result['task'] = task
                results.append(result)
                if (len(results)) % 20 == 0:
                    n_won = sum(1 for r in results if r.get('won', False))
                    print(f"  Progress: {len(results)}/{len(sampled)} episodes, {n_won} won ({n_won/len(results)*100:.0f}%)")
            except Exception as e:
                print(f"  Episode {i} failed: {e}")

    n_won = sum(1 for r in results if r.get('won', False))
    n_lost = len(results) - n_won
    print(f"\nDone: {len(results)} episodes, {n_won} won ({n_won/len(results)*100:.1f}%), {n_lost} lost")

    # Summarize trajectories into skills
    print(f"\nSummarizing trajectories into skills / lessons...")
    # Take all successes + sample of failures
    successes = [r for r in results if r.get('won', False)]
    failures = [r for r in results if not r.get('won', False)]
    # Keep all successes, sample up to same number of failures
    selected_failures = random.sample(failures, min(len(successes), len(failures)))
    to_summarize = successes + selected_failures
    random.shuffle(to_summarize)

    skills = []
    skill_rewards = []  # Track whether skill came from success or failure
    for i, result in enumerate(to_summarize):
        try:
            skill = summarize_trajectory(
                result['task'],
                result.get('trajectory', []),
                result.get('won', False)
            )
            if skill and len(skill) > 20:
                skills.append(skill)
                # Success skills get reward 1.0, failure lessons get 0.2
                skill_rewards.append(1.0 if result.get('won', False) else 0.2)
                if (i + 1) % 20 == 0:
                    print(f"  Summarized {i+1}/{len(to_summarize)}, got {len(skills)} skills")
        except Exception as e:
            print(f"  Summarize failed for episode {i}: {e}")
        # Rate limit
        time.sleep(0.2)

    print(f"\nGenerated {len(skills)} skills from {len(to_summarize)} episodes")

    # Build graph with dedup (initial rewards = optimistic 0.5 for now)
    print("\nBuilding experience graph...")
    from expharness.graph.experience_graph_server import ContrieverRetriever, ExperienceGraph

    retriever = ContrieverRetriever(device=args.device)
    graph = ExperienceGraph(retriever=retriever, max_nodes=2000)
    graph.add_nodes(skills, dedup_threshold=0.90)
    print(f"Graph built: {graph.get_stats()}")

    # ================================================================
    # Phase 2: Offline evaluation — score each skill node
    # Re-run the SAME gamefiles with retrieved experiences vs without
    # to compute real utility + generation reward for each node
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Offline scoring ({len(sampled)} episodes with experience retrieval)")
    print(f"{'='*60}")

    # Collect per-node rewards: node_id -> list of rewards
    from collections import defaultdict
    node_rewards = defaultdict(list)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        batch_size = 16
        for batch_start in range(0, len(sampled), batch_size):
            batch_games = sampled[batch_start:batch_start + batch_size]

            # Retrieve experiences for each game
            batch_experiences = []
            batch_node_indices = []
            for task, gamefile in batch_games:
                texts, node_ids = graph.retrieve(task, R=50, top_k=10, W=60)
                exp_text = "\n---\n".join(texts) if texts else None
                batch_experiences.append(exp_text)
                batch_node_indices.append(node_ids)

            # Run WITH-exp and WITHOUT-exp episodes in parallel
            futures_with = {}
            futures_without = {}
            for i, (task, gamefile) in enumerate(batch_games):
                fw = executor.submit(_run_episode_standalone, (gamefile, batch_experiences[i], args.max_steps))
                futures_with[fw] = i
                fwo = executor.submit(_run_episode_standalone, (gamefile, None, args.max_steps))
                futures_without[fwo] = i

            results_with = [None] * len(batch_games)
            results_without = [None] * len(batch_games)
            default_result = {"won": False, "score": 0.0}

            for future in as_completed({**futures_with, **futures_without}):
                if future in futures_with:
                    idx = futures_with[future]
                    try:
                        results_with[idx] = future.result()
                    except:
                        results_with[idx] = default_result
                else:
                    idx = futures_without[future]
                    try:
                        results_without[idx] = future.result()
                    except:
                        results_without[idx] = default_result

            # Compute rewards and assign to nodes
            for i in range(len(batch_games)):
                rw = results_with[i] or default_result
                rwo = results_without[i] or default_result
                score_with = rw.get('score', 0.0)
                score_without = rwo.get('score', 0.0)
                # Utility-grounded reward: utility + generation
                reward = (score_with - score_without) + score_with
                for node_id in batch_node_indices[i]:
                    node_rewards[node_id].append(reward)

            n_done = min(batch_start + batch_size, len(sampled))
            if n_done % 50 == 0 or n_done == len(sampled):
                print(f"  Scored {n_done}/{len(sampled)} episodes")

    # Update graph bandit scores with real rewards
    print(f"\nUpdating {len(node_rewards)} nodes with offline scores...")
    for node_id, rewards in node_rewards.items():
        if node_id < graph.num_nodes and rewards:
            avg = sum(rewards) / len(rewards)
            graph.avg_rewards[node_id] = avg
            graph.retrieval_counts[node_id] = len(rewards)

    # Nodes never retrieved keep optimistic init (0.5)
    n_scored = sum(1 for nid in range(graph.num_nodes) if nid in node_rewards)
    n_unscored = graph.num_nodes - n_scored
    print(f"  Scored: {n_scored} nodes, Unscored (kept 0.5): {n_unscored} nodes")

    # Print score distribution
    scored_rewards = [sum(r)/len(r) for r in node_rewards.values() if r]
    if scored_rewards:
        print(f"  Reward distribution: min={min(scored_rewards):.2f}, "
              f"mean={sum(scored_rewards)/len(scored_rewards):.2f}, "
              f"max={max(scored_rewards):.2f}")

    graph.save(args.output_graph)
    print(f"\nGraph saved to {args.output_graph}: {graph.get_stats()}")


if __name__ == "__main__":
    main()

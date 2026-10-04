"""
Build the initial experience graph for the static ExpSuite tasks (QA / reasoning / coding).

Phase 1: run the frozen executor zero-shot on training questions -> summarize into skills / lessons
Phase 2: re-run with retrieved experiences -> initialize node utilities offline
"""

import argparse
import json
import os
import re
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from expharness.envs.static_agent import _run_qa_standalone, DOMAIN_MAP
from expharness.utils.llm_api import summarizer_call


def summarize_qa(question, response, domain, correct):
    """Summarize a QA result into a reusable reasoning pattern."""
    prefix = "[SUCCESS]" if correct else "[FAILURE]"
    label = "correctly" if correct else "incorrectly"
    prompt = f"""A {domain} question was answered {label}.

Question: {question[:300]}
Response: {response[:300]}

Summarize the reasoning into ONE short skill (1-2 sentences, under 100 words).
Format: "[problem type]: [key technique or tip]"
Do NOT include the specific numbers or answer from this question.

Output format: SKILL: [your skill text]"""

    full_prompt = "You are an expert at analyzing reasoning patterns.\n\n" + prompt
    resp = summarizer_call(full_prompt, max_tokens=150)
    match = re.search(r'SKILL:\s*(.*)', resp, re.DOTALL)
    skill_text = match.group(1).strip() if match else resp.strip()
    return f"{prefix} {skill_text}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/static/all/train.parquet")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--output_graph", default="data/static/all/cold_start_graph.json")
    parser.add_argument("--device", default="cuda", help="Device for the Contriever encoder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load training data
    df = pd.read_parquet(args.train_data)
    print(f"Loaded {len(df)} training samples")

    # Sample
    indices = random.sample(range(len(df)), min(args.num_samples, len(df)))
    samples = df.iloc[indices]
    print(f"Sampled {len(samples)} for cold start")

    # Phase 1: Run zero-shot
    print(f"\n{'='*60}")
    print(f"Phase 1: Zero-shot evaluation ({len(samples)} questions)")
    print(f"{'='*60}")

    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for i, (_, row) in enumerate(samples.iterrows()):
            prompt = row['prompt']
            if isinstance(prompt, list):
                question = prompt[-1]['content'] if prompt else ''
            else:
                question = str(prompt)

            data_source = row.get('data_source', 'unknown')
            ground_truth = row.get('reward_model', '{}')
            extra_info = row.get('extra_info', '{}')

            f = executor.submit(_run_qa_standalone,
                (question, None, data_source, ground_truth, extra_info, i))
            futures[f] = (i, question, data_source)

        for future in as_completed(futures):
            i, question, data_source = futures[future]
            try:
                result = future.result()
                result['question'] = question[:500]
                results.append(result)
            except Exception as e:
                print(f"  Episode {i} failed: {e}")

            if len(results) % 50 == 0 and len(results) > 0:
                n_correct = sum(1 for r in results if r.get('score', 0) > 0)
                print(f"  Progress: {len(results)}/{len(samples)}, "
                      f"{n_correct} correct ({n_correct/len(results)*100:.0f}%)")

    n_correct = sum(1 for r in results if r.get('score', 0) > 0)
    print(f"\nDone: {len(results)} questions, {n_correct} correct "
          f"({n_correct/len(results)*100:.1f}%)")

    # Summarize into skills
    print(f"\nSummarizing into skills...")
    correct_results = [r for r in results if r.get('score', 0) > 0]
    incorrect_results = [r for r in results if r.get('score', 0) == 0]
    selected = correct_results + random.sample(
        incorrect_results, min(len(correct_results), len(incorrect_results)))
    random.shuffle(selected)

    skills = []
    skill_rewards = []
    for i, r in enumerate(selected):
        try:
            skill = summarize_qa(
                r.get('question', ''),
                r.get('response', ''),
                r.get('domain', 'qa'),
                r.get('score', 0) > 0,
            )
            if skill and len(skill) > 20:
                skills.append(skill)
                skill_rewards.append(1.0 if r.get('score', 0) > 0 else 0.2)
            if (i + 1) % 20 == 0:
                print(f"  Summarized {i+1}/{len(selected)}, got {len(skills)} skills")
        except Exception as e:
            print(f"  Summarize failed: {e}")
        time.sleep(0.1)

    print(f"\nGenerated {len(skills)} skills")

    # Build graph
    print("\nBuilding experience graph...")
    from expharness.graph.experience_graph_server import ContrieverRetriever, ExperienceGraph

    retriever = ContrieverRetriever(device=args.device)
    graph = ExperienceGraph(retriever=retriever, max_nodes=2000)
    graph.add_nodes(skills, dedup_threshold=0.90, initial_rewards=skill_rewards)
    print(f"Graph built: {graph.get_stats()}")

    # Phase 2: Offline scoring
    print(f"\n{'='*60}")
    print(f"Phase 2: Offline scoring ({len(samples)} questions with experiences)")
    print(f"{'='*60}")

    from collections import defaultdict
    node_rewards = defaultdict(list)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        batch_size = 16
        for batch_start in range(0, len(samples), batch_size):
            batch = list(samples.iterrows())[batch_start:batch_start + batch_size]

            # Retrieve experiences
            batch_experiences = []
            batch_node_indices = []
            for _, row in batch:
                prompt = row['prompt']
                if isinstance(prompt, list):
                    q = prompt[-1]['content'] if prompt else ''
                else:
                    q = str(prompt)
                texts, node_ids = graph.retrieve(q[:200], R=50, top_k=10, W=60)
                exp_text = "\n---\n".join(texts) if texts else None
                batch_experiences.append(exp_text)
                batch_node_indices.append(node_ids)

            # Run with and without
            futures_with = {}
            futures_without = {}
            for j, ((_, row), exp) in enumerate(zip(batch, batch_experiences)):
                prompt = row['prompt']
                q = prompt[-1]['content'] if isinstance(prompt, list) and prompt else str(prompt)
                ds = row.get('data_source', 'unknown')
                gt = row.get('reward_model', '{}')
                ei = row.get('extra_info', '{}')

                fw = executor.submit(_run_qa_standalone, (q, exp, ds, gt, ei, j))
                futures_with[fw] = j
                fwo = executor.submit(_run_qa_standalone, (q, None, ds, gt, ei, j + batch_size))
                futures_without[fwo] = j

            results_w = [None] * len(batch)
            results_wo = [None] * len(batch)
            for future in as_completed({**futures_with, **futures_without}):
                if future in futures_with:
                    idx = futures_with[future]
                    try: results_w[idx] = future.result()
                    except: results_w[idx] = {"score": 0.0}
                else:
                    idx = futures_without[future]
                    try: results_wo[idx] = future.result()
                    except: results_wo[idx] = {"score": 0.0}

            for j in range(len(batch)):
                rw = results_w[j] or {"score": 0.0}
                rwo = results_wo[j] or {"score": 0.0}
                reward = (rw['score'] - rwo['score']) + rw['score']
                for nid in batch_node_indices[j]:
                    node_rewards[nid].append(reward)

            n_done = min(batch_start + batch_size, len(samples))
            if n_done % 100 == 0 or n_done == len(samples):
                print(f"  Scored {n_done}/{len(samples)}")

    # Update scores
    print(f"\nUpdating {len(node_rewards)} nodes with offline scores...")
    for nid, rewards in node_rewards.items():
        if nid < graph.num_nodes and rewards:
            graph.avg_rewards[nid] = sum(rewards) / len(rewards)
            graph.retrieval_counts[nid] = len(rewards)

    graph.save(args.output_graph)
    print(f"\nGraph saved to {args.output_graph}: {graph.get_stats()}")


if __name__ == "__main__":
    main()

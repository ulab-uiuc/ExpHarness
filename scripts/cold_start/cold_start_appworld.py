"""
Build the initial AppWorld experience graph for ExpHarness.

1. Run all AppWorld `train` tasks zero-shot through the AppWorld episode server
2. Summarize each trajectory into a structured skill / lesson
3. Build the experience graph (dedup + kNN edges)

Start the episode server first:  bash run/appworld/start_servers.sh
"""

import json
import os
import re
import random
import sys
import time
import argparse
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def expand_server_urls(raw_url):
    parts = [p.strip() for p in str(raw_url).split(",") if p.strip()]
    if not parts:
        return ["http://127.0.0.1:8006"]
    if len(parts) > 1 and "://" in parts[0] and all("://" not in p for p in parts[1:]):
        base = parts[0].rsplit(":", 1)[0]
        return [parts[0]] + [f"{base}:{p}" for p in parts[1:]]
    return parts


def summarize_appworld(task, trajectory, score):
    """Summarize a single AppWorld episode into a structured skill."""
    from expharness.utils.llm_api import summarizer_call

    steps = trajectory if isinstance(trajectory, list) else []
    traj_str = "\n".join([
        f"Step {s.get('step',i+1)} CODE:\n{s.get('code','')}\nStep {s.get('step',i+1)} OUTPUT:\n{s.get('output','')[:200]}"
        for i, s in enumerate(steps[:10])
    ])

    if score >= 0.8:
        outcome = f"SUCCESSFUL (score: {score:.2f})"
    elif score >= 0.3:
        outcome = f"PARTIALLY SUCCESSFUL (score: {score:.2f})"
    else:
        outcome = f"FAILED (score: {score:.2f})"

    prompt = f"""Analyze this {outcome} AppWorld code generation episode.

Task: {task[:300]}
Trajectory:
{traj_str}

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

    try:
        resp = summarizer_call(prompt, max_tokens=200)
    except Exception:
        resp = ""

    skill_text = (resp or "").strip()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--server_url", default="http://127.0.0.1:8006")
    parser.add_argument("--output_episodes", default="outputs/appworld_cold_start/episodes.json")
    parser.add_argument("--output_skills", default="outputs/appworld_cold_start/skills.json")
    parser.add_argument("--output_graph", default="data/appworld/cold_start_graph.json")
    parser.add_argument("--contriever_path", default="facebook/contriever")
    parser.add_argument("--retriever_device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    import expharness.utils.llm_api as _llm
    _llm.GEMINI_MODEL = os.environ.get("GEMINI_MODEL", _llm.GEMINI_MODEL)
    _llm.AGENT_API = os.environ.get("AGENT_API", _llm.AGENT_API)

    server_urls = expand_server_urls(args.server_url)
    healthy_urls = []
    for url in server_urls:
        try:
            resp = http_requests.get(f"{url}/stats", timeout=5)
            if resp.status_code == 200 and resp.json().get("mode") == "single-threaded":
                healthy_urls.append(url)
                print(f"AppWorld server: {url} {resp.json()}")
            else:
                print(f"Skipping non-AppWorld server at {url}: {resp.text[:120]}")
        except Exception as e:
            print(f"AppWorld server unavailable at {url}: {e}")
    if not healthy_urls:
        print(f"ERROR: AppWorld server not available at {args.server_url}")
        return
    server_urls = healthy_urls

    # Get train task IDs
    if not os.environ.get("APPWORLD_ROOT"):
        raise SystemExit("Please export APPWORLD_ROOT=<path to your AppWorld root (contains data/)>")
    from appworld import load_task_ids
    train_ids = load_task_ids("train")
    print(f"Train tasks: {len(train_ids)}")

    # ================================================================
    # Phase 1: Run all training tasks zero-shot
    # ================================================================
    print(f"\nPhase 1: Running {len(train_ids)} episodes zero-shot...")

    # Send in batches of 8, with one sequential queue per single-threaded server.
    all_results = []
    batch_size = 8
    request_timeout = float(os.environ.get("APPWORLD_REQUEST_TIMEOUT", "150"))
    for start in range(0, len(train_ids), batch_size):
        batch_ids = train_ids[start:start + batch_size]
        episodes_req = [{"task_id": tid, "max_steps": args.max_steps} for tid in batch_ids]

        jobs_by_url = [[] for _ in server_urls]
        for j, ep in enumerate(episodes_req):
            jobs_by_url[j % len(server_urls)].append(ep)

        def run_queue(url, jobs):
            queue_results = []
            for ep in jobs:
                try:
                    resp = http_requests.post(f"{url}/run_episode", json=ep, timeout=request_timeout)
                    resp.raise_for_status()
                    queue_results.append(resp.json())
                except Exception as e:
                    print(f"  Episode {ep['task_id']} failed via {url}: {e}")
                    queue_results.append({"task_id": ep["task_id"], "success": False, "score": 0.0,
                                          "steps": 0, "trajectory": [], "instruction": ""})
            return queue_results

        with ThreadPoolExecutor(max_workers=len(server_urls)) as pool:
            futures = [
                pool.submit(run_queue, url, jobs)
                for url, jobs in zip(server_urls, jobs_by_url)
                if jobs
            ]
            for future in as_completed(futures):
                all_results.extend(future.result())

        done = min(start + batch_size, len(train_ids))
        n_success = sum(1 for r in all_results if r.get("success"))
        avg_score = sum(r.get("score", 0) for r in all_results) / len(all_results) if all_results else 0
        print(f"  {done}/{len(train_ids)}: success={n_success}, avg_score={avg_score:.3f}", flush=True)

    # Save episodes
    os.makedirs(os.path.dirname(args.output_episodes) or ".", exist_ok=True)
    with open(args.output_episodes, "w") as f:
        json.dump(all_results, f)
    n_success = sum(1 for r in all_results if r.get("success"))
    avg_score = sum(r.get("score", 0) for r in all_results) / len(all_results)
    print(f"\nPhase 1 done: {len(all_results)} episodes, {n_success} success, avg_score={avg_score:.3f}")

    # ================================================================
    # Phase 2: Summarize each trajectory into a skill
    # ================================================================
    print(f"\nPhase 2: Summarizing into skills...")

    successes = [r for r in all_results if r.get("score", 0) >= 0.8]
    partials = [r for r in all_results if 0.3 <= r.get("score", 0) < 0.8]
    failures = [r for r in all_results if r.get("score", 0) < 0.3]
    n_good = len(successes) + len(partials)
    selected_failures = random.sample(failures, min(len(failures), max(n_good, 20)))
    to_summarize = successes + partials + selected_failures
    random.shuffle(to_summarize)
    print(f"  Success: {len(successes)}, Partial: {len(partials)}, Failure(sampled): {len(selected_failures)}")
    print(f"  Summarizing {len(to_summarize)} episodes...")

    skills = []
    for i, r in enumerate(to_summarize):
        try:
            skill = summarize_appworld(
                r.get("instruction", ""),
                r.get("trajectory", []),
                r.get("score", 0))
            if skill and len(skill) > 20:
                skills.append(skill)
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(to_summarize)}, got {len(skills)} skills", flush=True)
        except Exception as e:
            print(f"    Summarize failed: {e}")
        time.sleep(0.3)

    os.makedirs(os.path.dirname(args.output_skills) or ".", exist_ok=True)
    with open(args.output_skills, "w") as f:
        json.dump(skills, f)
    print(f"  Generated {len(skills)} skills")
    lengths = [len(s) for s in skills]
    if lengths:
        print(f"  Avg: {sum(lengths)/len(lengths):.0f} chars, Max: {max(lengths)}, Min: {min(lengths)}")

    # ================================================================
    # Phase 3: Build experience graph
    # ================================================================
    print(f"\nPhase 3: Building graph...")
    from expharness.graph.experience_graph_server import ContrieverRetriever, ExperienceGraph
    retriever = ContrieverRetriever(model_name=args.contriever_path, device=args.retriever_device)
    graph = ExperienceGraph(retriever=retriever, max_nodes=2000)
    graph.add_nodes(skills, dedup_threshold=0.92)
    graph.save(args.output_graph)
    print(f"  Graph: {graph.get_stats()}")

    # Show samples
    print(f"\nSample skills:")
    for s in skills[:5]:
        print(f"  {s[:150]}")


if __name__ == "__main__":
    main()

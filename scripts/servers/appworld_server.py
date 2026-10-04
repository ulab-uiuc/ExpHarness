"""
AppWorld episode server for ExpHarness — multiple single-threaded http.server instances.
Each instance listens on its own port and is pinned to its own executor API key.
Runs in the AppWorld Python environment (separate from the training environment).
The HTTP process stays single-threaded; each episode runs in a child process
so AppWorld/IPython/SQLite hangs cannot permanently occupy a server port.

Usage:
  # Launch 4 servers on ports 8006-8009:
  APPWORLD_ROOT=/path/to/appworld python scripts/servers/appworld_server.py --port 8006 --num-servers 4

  # Single server on port 8006:
  python scripts/servers/appworld_server.py --port 8006 --num-servers 1
"""

import os
import sys
import json
import re
import argparse
import time
import shutil
import multiprocessing as mp
import queue as queue_module
from http.server import HTTPServer, BaseHTTPRequestHandler

# Set AppWorld root BEFORE any import
APPWORLD_ROOT = os.environ.get("APPWORLD_ROOT")
if not APPWORLD_ROOT:
    sys.exit("Please export APPWORLD_ROOT=<path to your AppWorld root (contains data/)>")
if "IPYTHONDIR" not in os.environ:
    os.environ["IPYTHONDIR"] = "/tmp/.ipython_expharness"
os.makedirs(os.environ["IPYTHONDIR"], exist_ok=True)

EXPHARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if EXPHARNESS_ROOT not in sys.path:
    sys.path.insert(0, EXPHARNESS_ROOT)

# Executor LLM client
import expharness.utils.llm_api as _llm
llm_call = _llm.llm_call

LLM_TEMPERATURE = float(os.environ.get("APPWORLD_LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.environ.get("APPWORLD_LLM_MAX_TOKENS", "512"))
EPISODE_TIMEOUT_SECONDS = int(os.environ.get("APPWORLD_EPISODE_TIMEOUT", "60"))
EPISODE_PROCESS = os.environ.get("APPWORLD_EPISODE_PROCESS", "1") != "0"
KEEP_OUTPUTS = os.environ.get("APPWORLD_KEEP_OUTPUTS", "0") == "1"
PROMPT_CODE_MAX_CHARS = int(os.environ.get("APPWORLD_PROMPT_CODE_CHARS", "1200"))
PROMPT_OUTPUT_MAX_CHARS = int(os.environ.get("APPWORLD_PROMPT_OUTPUT_CHARS", "2000"))
STEP_OUTPUT_MAX_CHARS = int(os.environ.get("APPWORLD_STEP_OUTPUT_CHARS", "4000"))
if hasattr(time, "clock_gettime") and hasattr(time, "CLOCK_MONOTONIC"):
    def real_monotonic():
        return time.clock_gettime(time.CLOCK_MONOTONIC)
else:
    _REAL_MONOTONIC = time.monotonic

    def real_monotonic():
        return _REAL_MONOTONIC()


# ================================================================
# Prompt building
# ================================================================

def _build_task_header(instruction, supervisor, allowed_apps, app_descriptions, retrieved_memories):
    name_parts = [supervisor.get("first_name"), supervisor.get("last_name")]
    supervisor_name = " ".join([p for p in name_parts if p]).strip()
    supervisor_email = supervisor.get("email") or ""
    supervisor_phone = supervisor.get("phone_number") or ""
    descriptions = [{"name": n, "description": app_descriptions.get(n, "")} for n in allowed_apps]
    lines = [
        "You are an AI assistant acting on behalf of your supervisor in AppWorld.",
        "Solve the task by writing Python code that will be executed in the current AppWorld environment.",
        "Act autonomously and do not ask for clarification.",
        "In each turn, output exactly one Python code block and nothing else.",
        "Use short code snippets that make concrete progress.",
        "Write only one small code chunk per turn and make sure it works before making irreversible changes.",
        "Variables persist across turns, so you may reuse earlier variables.",
        "Never invent or guess values; retrieve real values through the available APIs.",
        "Never leave placeholders like usernames, ids, tokens, or addresses in the code.",
        "Use the supervisor app for personal information, credentials, addresses, and payment cards when needed.",
        "When a result may be needed later, store it in a variable.",
        "Always print the important intermediate results that later steps depend on.",
        "For example, after login, print the returned token or relevant keys; after list/search APIs, print the useful ids, titles, counts, or selected objects.",
        "Do not print unnecessarily large outputs; print only concise, useful fields.",
        "If you need API details, inspect them first with these helpers:",
        'print(apis.api_docs.show_app_descriptions())',
        'print(apis.api_docs.show_api_descriptions(app_name="spotify"))',
        'print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))',
        "Always inspect API documentation before calling an unfamiliar API.",
        "For paginated APIs, keep iterating page_index until all results are processed.",
        "Use only the provided app APIs and the Python standard library; do not use OS/system operations or third-party app packages.",
        "When the task is finished, call `apis.supervisor.complete_task()`.",
        "If the task requires a final textual answer, call `apis.supervisor.complete_task(answer=...)`.",
        "If you cannot complete the task after reasonable attempts, call `apis.supervisor.complete_task(status=\"fail\")`.",
        "If you provide an answer, keep it minimal: only the direct value requested.",
    ]
    lines.extend(["", f"Task: {instruction.strip()}"])
    if supervisor_name:
        lines.extend(["", "Supervisor:", f"- name: {supervisor_name}", f"- email: {supervisor_email}", f"- phone: {supervisor_phone}"])
    if allowed_apps:
        lines.extend(["", "Allowed apps:", ", ".join(allowed_apps)])
    if descriptions:
        lines.extend(["", "App descriptions:", json.dumps(descriptions, indent=1, ensure_ascii=False)])
    if retrieved_memories:
        lines.append("")
        lines.append("Retrieved memories (optional hints):")
        for idx, m in enumerate(retrieved_memories):
            lines.append(f"{idx+1}. {str(m).strip()}")
    return "\n".join(lines).strip()


def _build_turn_prompt(task_header, recent_steps):
    lines = [task_header]
    if recent_steps:
        lines.extend(["", "Recent execution history:"])
        for step in recent_steps:
            code = _clip_text(step.get("code") or "", PROMPT_CODE_MAX_CHARS)
            output = _clip_text(step.get("output") or "", PROMPT_OUTPUT_MAX_CHARS)
            lines.extend(["", f"Step {step['step']} code:", f"```python\n{code}\n```",
                          f"Step {step['step']} output:", f"```\n{output}\n```"])
    lines.extend(["", "Write the next Python code snippet.", "Return only one Python code block."])
    return "\n".join(lines).strip()


def _extract_python_code(text):
    content = (text or "").strip()
    if not content:
        return ""
    m = re.search(r"```python\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```python\s*(.*)$", content, flags=re.IGNORECASE | re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", content, flags=re.DOTALL)
    if m: return m.group(1).strip()
    return content


def _clip_text(text, max_chars):
    text = str(text or "").rstrip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    return f"{text[:max_chars]}\n...(truncated {remaining} chars)"


def _supervisor_to_dict(supervisor):
    if supervisor is None: return {}
    if isinstance(supervisor, dict): return supervisor
    return {k: getattr(supervisor, k, "") for k in ["first_name", "last_name", "email", "phone_number"]}


# ================================================================
# Episode execution
# ================================================================

def _run_episode(task_id, experiences, max_steps, history_window=5):
    if not EPISODE_PROCESS:
        return _run_episode_inner(task_id, experiences, max_steps, history_window)

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_episode_worker,
        args=(result_queue, task_id, experiences, max_steps, history_window),
        daemon=True,
    )
    proc.start()
    proc.join(EPISODE_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        print(f"[EP] {task_id}: HARD TIMEOUT {EPISODE_TIMEOUT_SECONDS}s", flush=True)
        return _default_episode_result(task_id, f"hard_timeout_{EPISODE_TIMEOUT_SECONDS}s")
    try:
        return result_queue.get_nowait()
    except queue_module.Empty:
        return _default_episode_result(task_id, f"worker_exit_{proc.exitcode}")
    finally:
        result_queue.close()
        result_queue.join_thread()

def _run_episode_worker(result_queue, task_id, experiences, max_steps, history_window):
    try:
        result = _run_episode_inner(task_id, experiences, max_steps, history_window)
    except BaseException as e:
        print(f"[EP] {task_id}: WORKER ERROR {e}", flush=True)
        result = _default_episode_result(task_id, str(e))
    result_queue.put(result)

def _default_episode_result(task_id, error):
    return {"task_id": task_id, "success": False, "score": 0.0,
            "pass_percentage": 0.0, "steps": 0, "trajectory": [],
            "instruction": "", "error": error}

def _run_episode_inner(task_id, experiences, max_steps, history_window=5):
    from appworld import AppWorld
    _patch_appworld_requester()

    retrieved_memories = []
    if experiences:
        if isinstance(experiences, list):
            retrieved_memories = experiences
        elif isinstance(experiences, str) and "\n---\n" in experiences:
            retrieved_memories = [e.strip() for e in experiences.split("\n---\n") if e.strip()]
        elif isinstance(experiences, str):
            retrieved_memories = [experiences]

    experiment_name = f"expharness_{task_id}_{os.getpid()}_{int(time.time() * 1000) % 100000}"
    output_directory = None

    try:
        steps = []
        episode_start = real_monotonic()
        episode_timeout = max(1, EPISODE_TIMEOUT_SECONDS - 5)

        with AppWorld(task_id=task_id, experiment_name=experiment_name,
                      load_ground_truth=True, ground_truth_mode="minimal",
                      max_interactions=int(max_steps), timeout_seconds=30,
                      raise_on_failure=False) as world:
            output_directory = world.base_output_directory
            task = world.task
            instruction = str(task.instruction or "").strip()
            supervisor = _supervisor_to_dict(getattr(task, "supervisor", None))
            allowed_apps = list(getattr(task, "allowed_apps", []) or [])
            app_descriptions = dict(getattr(task, "app_descriptions", {}) or {})

            task_header = _build_task_header(instruction, supervisor, allowed_apps,
                                            app_descriptions, retrieved_memories)

            consecutive_empty = 0
            for step_idx in range(1, int(max_steps) + 1):
                if real_monotonic() - episode_start > episode_timeout:
                    break
                recent = steps[-max(1, int(history_window)):]
                prompt = _build_turn_prompt(task_header, recent)
                response = llm_call(prompt, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
                code = _extract_python_code(str(response or ""))
                if not code:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        break
                    steps.append({"step": step_idx, "code": "", "output": "(no code)"})
                    continue
                consecutive_empty = 0
                output = _clip_text(str(world.execute(code)), STEP_OUTPUT_MAX_CHARS)
                steps.append({"step": step_idx, "code": code, "output": output})
                if world.task_completed():
                    break

            tracker = world.evaluate(suppress_errors=True)
            success = bool(tracker.success)
            pass_pct = float(tracker.pass_percentage)

        elapsed = real_monotonic() - episode_start
        print(f"[EP] {task_id}: score={pass_pct/100:.2f} steps={len(steps)} time={elapsed:.0f}s", flush=True)
        return {
            "task_id": task_id, "success": success,
            "score": pass_pct / 100.0, "pass_percentage": pass_pct,
            "steps": len(steps),
            "trajectory": [{"step": s["step"], "code": s["code"][:200], "output": s["output"][:200]} for s in steps],
            "instruction": instruction[:200],
        }
    except Exception as e:
        elapsed = real_monotonic() - episode_start
        print(f"[EP] {task_id}: ERROR {e} time={elapsed:.0f}s", flush=True)
        return {
            "task_id": task_id, "success": False, "score": 0.0,
            "pass_percentage": 0.0, "steps": 0, "trajectory": [],
            "instruction": "", "error": str(e),
        }
    finally:
        if output_directory and not KEEP_OUTPUTS:
            shutil.rmtree(output_directory, ignore_errors=True)


def _patch_appworld_requester():
    if os.environ.get("APPWORLD_TESTCLIENT_ENTER", "0") == "1":
        return
    import appworld.requester as requester_module

    Requester = requester_module.Requester
    if getattr(Requester, "_expharness_testclient_patch", False):
        return

    def _get_client(self):
        klass = self.__class__
        if self.apps not in klass.clients:
            klass.clients[self.apps] = requester_module.TestClient(
                requester_module.build_main_app(list(self.apps))
            )
        return klass.clients[self.apps]

    def _close_client(client):
        if hasattr(client, "exit_stack"):
            client.__exit__(None, None, None)

    def close(self):
        if self.client:
            _close_client(self.client)
            self.clients.pop(self.apps, None)
        self.request_tracker.reset()
        if self.mcp:
            self.mcp.disconnect()
        if not self.time_freezer_or_id:
            return
        time_freezer_or_id = self.time_freezer_or_id
        if not self.remote_apis_url:
            requester_module.unset_local_date_and_time(time_freezer_or_id)
        else:
            requester_module.unset_remote_date_and_time(self.remote_apis_url, time_freezer_or_id)
            self.time_freezer_id_to_remote_apis_url.pop(time_freezer_or_id, None)
        if self.date_and_time and self.time_freezer_or_id in self.time_freezers_or_ids:
            self.time_freezers_or_ids.remove(self.time_freezer_or_id)

    @classmethod
    def close_all(cls):
        for client in cls.clients.values():
            _close_client(client)
        cls.clients = {}
        for mcp in cls.mcps:
            mcp.disconnect()
        for time_freezer_or_id in cls.time_freezers_or_ids:
            if not isinstance(time_freezer_or_id, str):
                requester_module.unset_local_date_and_time(time_freezer_or_id)
            else:
                remote_apis_url = cls.time_freezer_id_to_remote_apis_url.pop(
                    time_freezer_or_id, None
                )
                if remote_apis_url is not None:
                    requester_module.unset_remote_date_and_time(remote_apis_url, time_freezer_or_id)
        cls.time_freezers_or_ids = []
        cls.time_freezer_id_to_remote_apis_url = {}

    Requester._get_client = _get_client
    Requester.close = close
    Requester.close_all = close_all
    Requester._expharness_testclient_patch = True


# ================================================================
# HTTP Handler
# ================================================================

class AppWorldHandler(BaseHTTPRequestHandler):
    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/run_episode":
            body = self._read_json()
            result = _run_episode(body["task_id"], body.get("experiences"),
                                  body.get("max_steps", 20))
            self._send_json(result)
        elif self.path == "/run_batch":
            body = self._read_json()
            results = [_run_episode(ep["task_id"], ep.get("experiences"),
                                    ep.get("max_steps", 20)) for ep in body["episodes"]]
            self._send_json({"results": results})
        else:
            self._send_json({"error": f"unknown path: {self.path}"}, 404)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/stats":
            self._send_json({
                "mode": "single-threaded",
                "episode_process": EPISODE_PROCESS,
                "episode_timeout": EPISODE_TIMEOUT_SECONDS,
                "pid": os.getpid(),
            })
        else:
            self._send_json({"status": "ok"})

    def log_message(self, format, *args):
        pass


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


# ================================================================
# Multi-server launcher
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--num-servers", type=int, default=4,
                        help="Launch N independent servers on consecutive ports (each pinned to its own API key)")
    args = parser.parse_args()

    if args.num_servers == 1:
        # Single server mode
        key_idx = int(os.environ.get("_APPWORLD_KEY_IDX", "0"))
        if _llm.AGENT_API == "gemini" and _llm.GEMINI_API_KEYS:
            _llm.set_gemini_worker_key(_llm.GEMINI_API_KEYS[key_idx % len(_llm.GEMINI_API_KEYS)])
        else:
            _llm.bind_worker(key_idx)
        print(f"[AppWorldServer] port={args.port} key={key_idx} pid={os.getpid()}", flush=True)
        ReusableHTTPServer(("0.0.0.0", args.port), AppWorldHandler).serve_forever()
    else:
        # Multi-server mode: launch N children on consecutive ports
        import subprocess, signal
        children = []
        for i in range(args.num_servers):
            port = args.port + i
            env = dict(os.environ)
            env["_APPWORLD_KEY_IDX"] = str(i)
            p = subprocess.Popen(
                [sys.executable, __file__, "--port", str(port), "--num-servers", "1"],
                env=env,
            )
            children.append((port, p))
            print(f"[AppWorldServer] Launched child {i}: port={port} pid={p.pid} key={i}", flush=True)

        time.sleep(1)
        failed = [(port, p.returncode) for port, p in children if p.poll() is not None]
        if failed:
            print(f"[AppWorldServer] Child startup failed: {failed}", file=sys.stderr, flush=True)
            for _, p in children:
                if p.poll() is None:
                    p.terminate()
            sys.exit(1)

        def _shutdown(sig, frame):
            for _, p in children:
                p.terminate()
            sys.exit(0)
        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        print(f"[AppWorldServer] {args.num_servers} servers on ports {args.port}-{args.port+args.num_servers-1}", flush=True)
        for _, p in children:
            p.wait()

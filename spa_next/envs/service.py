"""Loopback-only WebShop service; run in the legacy simulator environment."""
import argparse
import hashlib
import json
import math
import random
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_action(text):
    matches = re.findall(r"^Action:\s*((?:search|click)\[[^\n]*\])\s*$", text.strip(), re.M)
    if len(matches) != 1:
        raise ValueError("Expected exactly one Action: search[...] or Action: click[...] line")
    return matches[0]


class WebShopFactory:
    def __init__(self):
        from webshop.web_agent_site.envs import WebAgentTextEnv
        from webshop.web_agent_site.utils import DEFAULT_FILE_PATH
        from spa_next.envs.outcome_server import OutcomeServer
        self.env_class = WebAgentTextEnv
        # The legacy server seeds goal SHUFFLING only after random prices/goals
        # have been generated. Seed their construction too, for stable restarts.
        random.seed(0)
        server = OutcomeServer("http://127.0.0.1:3000", DEFAULT_FILE_PATH, human_goals=True)
        self.base = WebAgentTextEnv(observation_mode="text", server=server)
        # Release temporary allocations from loading the 5 GB legacy JSON file.
        import gc
        import sys
        gc.collect()
        if sys.platform.startswith("linux"):
            import ctypes
            libc = ctypes.CDLL(None)
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)

    def __call__(self, task_id, episode_id):
        # A unique prefix isolates sessions even when the same task runs concurrently.
        env = self.env_class(observation_mode="text", server=self.base.server,
                             session_prefix=episode_id + "_")
        unused_session = env.session
        env.reset(int(task_id))  # int selects the fixed goal; str would select a random goal.
        env.server.user_sessions.pop(unused_session, None)
        return env

    def close(self, env):
        env.server.user_sessions.pop(env.session, None)
        env.close()


class Episodes:
    def __init__(self, factory, max_steps=10, capacity=64, ttl=1800):
        self.factory, self.max_steps = factory, max_steps
        self.capacity, self.ttl = capacity, ttl
        self.items = {}
        # The legacy simulator uses shared mutable state; serialize only environment calls.
        self.lock = threading.RLock()

    def dispatch(self, operation, data):
        with self.lock:
            now = time.monotonic()
            for eid, item in list(self.items.items()):
                if now - item["touched"] > self.ttl:
                    self._close(eid)
            if operation == "reset":
                if len(self.items) >= self.capacity:
                    raise RuntimeError("Environment capacity exhausted")
                task_id = int(data["task_id"])
                if int(data.get("env_seed", 0)) != 0:
                    raise ValueError("V0 exposes only deterministic env_seed=0")
                if task_id < 0:
                    raise ValueError("Negative task ID")
                eid = uuid.uuid4().hex
                env = self.factory(task_id, eid)
                observation = str(env.observation)
                state_id = hashlib.sha256(observation.encode()).hexdigest()
                self.items[eid] = dict(env=env, step=0, done=False, touched=now)
                return dict(episode_id=eid, observation=observation,
                            initial_state_id=state_id, task_id=task_id,
                            env_seed=0, max_steps=self.max_steps)
            eid = data["episode_id"]
            if operation == "close":
                self._close(eid)
                return {"closed": True}
            if operation != "step":
                raise ValueError("Unknown operation")
            item = self.items[eid]
            item["touched"] = now
            if item["done"] or int(data["step_id"]) != item["step"]:
                raise ValueError("Stale/duplicate step or episode already ended; do not retry actions")
            error_type = None
            try:
                action = parse_action(data["text"])
            except ValueError:
                action = None
                obs, reward, done = "Invalid format. Use Action: search[...] or Action: click[...].", 0., False
                error_type = "invalid_format"
            else:
                try:
                    obs, reward, done, _ = item["env"].step(action)
                except AssertionError:
                    obs, reward, done = "Invalid action.", 0., False
                    error_type = "invalid_action"
                # All other simulator failures propagate as infrastructure errors.
            item["step"] += 1
            reward = float(reward)
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError("WebShop returned a non-finite or out-of-range reward")
            truncated = not done and item["step"] >= self.max_steps
            item["done"] = bool(done or truncated)
            return dict(observation=str(obs), parsed_action=action, terminated=bool(done),
                        truncated=truncated, success=bool(done and reward >= 1.0 - 1e-6),
                        outcome_reward=reward if done else 0., error_type=error_type,
                        step_id=item["step"] - 1)

    def _close(self, eid):
        item = self.items.pop(eid, None)
        if item:
            self.factory.close(item["env"])


def serve(episodes, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply(200 if self.path == "/health" else 404,
                       {"service": "spa-webshop-v0", "max_steps": episodes.max_steps,
                        "task_count": len(episodes.factory.base.server.goals)})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                self.reply(200, episodes.dispatch(self.path.lstrip("/"), data))
            except (ValueError, KeyError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:
                self.log_error("Simulator failure: %r", exc)
                self.reply(500, {"error": "Simulator infrastructure failure; abort this rollout batch"})

        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class Server(ThreadingHTTPServer):
        # GRPO starts an entire group batch at once. The stdlib default backlog
        # of five can reset connections before a handler receives the request.
        request_queue_size = 128

    Server(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    serve(Episodes(WebShopFactory(), args.max_steps), args.port)

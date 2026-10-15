"""Replay saved actions against legacy and outcome-only servers sharing one catalog."""
import argparse
import json
from pathlib import Path
from uuid import uuid4


def check(factory, trajectories):
    from webshop.web_agent_site.envs.web_agent_text_env import SimServer
    legacy = SimServer.__new__(SimServer)
    legacy.__dict__.update(factory.base.server.__dict__)
    legacy.user_sessions, legacy.sub_reward = {}, []
    rows = [json.loads(p.read_text()) for p in Path(trajectories).glob("*.json")]
    rows = sorted((r for r in rows if r["termination_reason"] == "terminated"),
                  key=lambda r: len(r["steps"]), reverse=True)
    selected, seen = [], set()
    for row in rows:
        if row["task_id"] not in seen:
            selected.append(row)
            seen.add(row["task_id"])
        if len(selected) == 4:
            break
    if len(selected) < 4:
        raise ValueError("Need four distinct saved terminal tasks for the parity check")
    turns = 0
    for row in selected:
        old = factory.env_class(observation_mode="text", server=legacy, session_prefix=uuid4().hex)
        old.reset(int(row["task_id"]))
        new = factory(int(row["task_id"]), uuid4().hex)
        try:
            assert old.observation == new.observation
            for step in row["steps"]:
                action = step.get("parsed_action")
                if not step["complete"] or action is None:
                    raise ValueError("Parity replay requires complete valid actions")
                old_obs, old_reward, old_done, _ = old.step(action)
                new_obs, new_reward, new_done, _ = new.step(action)
                assert old_obs == new_obs, (row["task_id"], action, "observation")
                assert old_done == new_done, (row["task_id"], action, "done")
                if old_done:
                    assert old_reward == new_reward, (row["task_id"], "terminal reward")
                turns += 1
            assert new_done
        finally:
            factory.close(new)
            legacy.user_sessions.clear()
    print(json.dumps(dict(parity="passed", tasks=len(selected), actions=turns,
                          checks="identical observations, termination, terminal reward")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="outputs/v0/trajectories")
    parser.add_argument("--serve-after", action="store_true")
    args = parser.parse_args()
    from spa_next.envs.service import WebShopFactory, Episodes, serve
    factory = WebShopFactory()
    check(factory, args.trajectories)
    if args.serve_after:
        serve(Episodes(factory), 8765)

import unittest

from spa_next.agents.tokens import TokenTrace
from spa_next.envs.service import Episodes, parse_action
from spa_next.prepare_data import select_splits


class FakeEnv:
    observation = "fixed goal"

    def step(self, action):
        if action == "click[assert]":
            raise AssertionError()
        if action == "click[crash]":
            raise RuntimeError("infrastructure failure")
        return "new observation", 0.5, True, None


class Factory:
    def __call__(self, task, episode):
        return FakeEnv()

    def close(self, env):
        pass


class Invariants(unittest.TestCase):
    def setUp(self):
        self.episodes = Episodes(Factory(), max_steps=2)
        self.reset = self.episodes.dispatch("reset", {"task_id": 10})

    def step(self, text, step=0):
        return self.episodes.dispatch("step", dict(episode_id=self.reset["episode_id"], text=text, step_id=step))

    def test_done_is_not_success(self):
        result = self.step("Thought: buy it\nAction: click[buy]")
        self.assertTrue(result["terminated"])
        self.assertFalse(result["success"])
        self.assertEqual(result["outcome_reward"], 0.5)

    def test_invalid_action_reward_is_initialized(self):
        self.assertEqual(self.step("Action: click[assert]")["outcome_reward"], 0)

    def test_budget_exhaustion(self):
        self.step("invalid")
        result = self.step("invalid", 1)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["terminated"])
        self.assertEqual(result["outcome_reward"], 0)

    def test_infrastructure_failure_propagates(self):
        with self.assertRaises(RuntimeError):
            self.step("Action: click[crash]")

    def test_actions_are_not_retried(self):
        self.step("invalid")
        with self.assertRaises(ValueError):
            self.step("invalid")

    def test_same_goal_isolated_sessions(self):
        other = self.episodes.dispatch("reset", {"task_id": 10})
        self.assertEqual(other["initial_state_id"], self.reset["initial_state_id"])
        self.assertNotEqual(other["episode_id"], self.reset["episode_id"])

    def test_ambiguous_action_rejected(self):
        with self.assertRaises(ValueError):
            parse_action("Action: click[x]\nAction: click[y]")

    def test_original_tokens_and_observation_mask(self):
        trace = TokenTrace([100, 101], 8)
        self.assertEqual(trace.append([8, 9], action_id=0, train=True, logprobs=[-1., -2.]), [0, 2])
        trace.append([5, 6, 7])
        trace.append([10], action_id=1, train=True, logprobs=[-3.])
        self.assertEqual(trace.context, [100, 101, 8, 9, 5, 6, 7, 10])
        self.assertEqual(trace.mask, [1, 1, 0, 0, 0, 1])
        self.assertEqual(trace.action_ids, [0, 0, -1, -1, -1, 1])

    def test_partial_action_is_not_trainable(self):
        trace = TokenTrace([1], 3)
        trace.append([3, 4])
        self.assertEqual(trace.mask, [0, 0])
        with self.assertRaises(ValueError):
            trace.append([5, 6], action_id=1, train=True, logprobs=[-1., -1.])
        self.assertEqual(trace.response_ids, [3, 4])

    def test_official_test_tasks_are_reserved(self):
        result = select_splits(list(range(13)), {0, 12}, 4, 4)
        self.assertEqual(result["train"], [1, 2, 3, 4])
        self.assertEqual(result["val"], [8, 9, 10, 11])
        self.assertFalse(set(result["train"]) & set(result["val"]))

    def test_insufficient_disjoint_tasks_rejected(self):
        with self.assertRaises(ValueError):
            select_splits(list(range(8)), {0}, 4, 4)


if __name__ == "__main__":
    unittest.main()

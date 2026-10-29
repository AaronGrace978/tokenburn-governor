"""Budget and stopping regression tests: python3 -m unittest -v test_budgets."""
import unittest

from governor import Governor, shannon_entropy


class BudgetTests(unittest.TestCase):
    def test_token_limit_overrides_warmup(self):
        g = Governor(max_tokens=10, min_steps=100)
        self.assertEqual(g.allowance(50), 10)
        self.assertEqual(g.observe(tokens_this_step=10), "STOP")
        self.assertEqual(g.stop_reason, "token_budget")
        self.assertEqual(g.allowance(50), 0)

    def test_cost_limit(self):
        g = Governor(token_price=0.25, max_cost=1)
        self.assertEqual(g.allowance(100), 4)
        self.assertEqual(g.observe(tokens_this_step=4), "STOP")
        self.assertEqual(g.cost_spent, 1)
        self.assertEqual(g.stop_reason, "cost_budget")

    def test_partial_remaining_budget(self):
        g = Governor(token_price=0.25, max_cost=1, max_tokens=3)
        self.assertEqual(g.observe(tokens_this_step=2), "THINK")
        self.assertEqual(g.allowance(100), 1)

    def test_zero_budget(self):
        for kwargs in ({"max_tokens": 0}, {"max_cost": 0}):
            self.assertEqual(Governor(**kwargs).allowance(10), 0)

    def test_subtoken_budget_and_free_tokens(self):
        self.assertEqual(Governor(token_price=1, max_cost=0.5).allowance(10), 0)
        self.assertEqual(Governor(token_price=0, max_cost=1).allowance(10), 10)

    def test_patience_recovers_then_stops(self):
        g = Governor(patience=2)
        for probs in ([0.5, 0.5], [0.5, 0.5], [0.9, 0.1], [0.9, 0.1]):
            self.assertEqual(g.observe(probs, 10), "THINK")
        self.assertEqual(g.observe([0.9, 0.1], 10), "STOP")
        self.assertEqual(g.stop_reason, "stalled_reasoning")
        self.assertEqual(g.observe([1, 0], 10), "STOP")
        self.assertEqual(g.tokens_spent, 60)

    def test_zero_price_stalls(self):
        g = Governor(token_price=0)
        g.observe([0.5, 0.5], 10)
        self.assertEqual(g.observe([0.5, 0.5], 10), "STOP")

    def test_missing_probabilities_reset_entropy_baseline(self):
        g = Governor()
        g.observe([0.5, 0.5], 10)
        g.observe(tokens_this_step=10)
        self.assertEqual(g.observe([0.5, 0.5], 10), "THINK")

    def test_invalid_inputs_do_not_change_accounting(self):
        g = Governor()
        for probs in ([], [0.2, 0.2], [-1, 2], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                g.observe(probs, 10)
        for tokens in (-1, 0, 1.5, True):
            with self.assertRaises(ValueError):
                g.observe(tokens_this_step=tokens)
        self.assertEqual(g.tokens_spent, 0)
        self.assertEqual(shannon_entropy([0, 1]), 0)

    def test_invalid_configuration(self):
        for kwargs in ({"max_tokens": -1}, {"max_cost": float("inf")},
                       {"token_price": -1}, {"patience": 0}, {"min_steps": None}):
            with self.assertRaises(ValueError):
                Governor(**kwargs)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

from adaptive import AdaptiveGovernor


class AdaptiveTests(unittest.TestCase):
    def test_learns_only_verified_first_attempts(self):
        g = AdaptiveGovernor(required_successes=2)
        self.addCleanup(g.close)
        for _ in range(2):
            a = g.start('model:v1:task:v1')
            self.assertEqual(a.next_plan().tokens, 256)
            self.assertEqual(a.report(100, verified=True), 'verified')
        self.assertEqual(g.start('model:v1:task:v1').next_plan().tokens, 64)
        self.assertEqual(g.start('model:v2:task:v1').next_plan().tokens, 256)
        self.assertEqual(g.start('model:v1:task:v1', high_stakes=True).next_plan().tokens, 256)

    def test_failure_revokes_familiarity_and_escalates(self):
        g = AdaptiveGovernor(required_successes=1, max_tokens=512)
        self.addCleanup(g.close)
        a = g.start('key'); a.next_plan(); a.report(10, verified=True)
        a = g.start('key')
        self.assertEqual(a.next_plan().tokens, 64)
        self.assertEqual(a.report(64, verified=False), 'retry')
        self.assertEqual(a.next_plan().tokens, 128)
        a.report(128, verified=True)
        self.assertEqual(g.start('key').next_plan().tokens, 256)

    def test_cumulative_ceiling_and_provider_overshoot(self):
        g = AdaptiveGovernor(initial_tokens=64, max_tokens=100)
        self.addCleanup(g.close)
        a = g.start('key'); a.next_plan(); a.report(64, verified=False)
        self.assertEqual(a.next_plan().tokens, 36)
        self.assertEqual(a.report(50, verified=False), 'budget_exhausted')
        self.assertEqual(a.spent, 114)
        self.assertEqual(a.next_plan().tokens, 0)

    def test_unknown_check_never_retries_or_learns(self):
        g = AdaptiveGovernor(required_successes=1)
        self.addCleanup(g.close)
        a = g.start('key'); a.next_plan()
        self.assertEqual(a.report(20, verified=None), 'verification_required')
        self.assertEqual(a.next_plan().tokens, 0)
        self.assertEqual(g.start('key').next_plan().tokens, 256)

    def test_history_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'history.sqlite')
            g = AdaptiveGovernor(path, required_successes=1)
            a = g.start('key'); a.next_plan(); a.report(10, verified=True); g.close()
            g = AdaptiveGovernor(path, required_successes=1)
            try:
                self.assertEqual(g.start('key').next_plan().tokens, 64)
            finally:
                g.close()

    def test_invalid_transitions_and_reports(self):
        g = AdaptiveGovernor(); self.addCleanup(g.close)
        a = g.start('key')
        with self.assertRaises(RuntimeError): a.report(1, verified=True)
        a.next_plan()
        with self.assertRaises(RuntimeError): a.next_plan()
        for usage, verified in ((-1, True), (True, True), (1, 'yes')):
            with self.assertRaises(ValueError): a.report(usage, verified=verified)
        self.assertEqual(a.spent, 0)
        a.report(0, verified=False)
        self.assertEqual(a.next_plan().tokens, 512)

    def test_zero_usage_retries_remain_bounded(self):
        g = AdaptiveGovernor(max_tokens=256); self.addCleanup(g.close)
        a = g.start('key')
        # Even a provider reporting zero usage must not permit unlimited retries.
        for _ in range(20):
            if a.next_plan().tokens == 0:
                break
            a.report(0, verified=False)
        else:
            self.fail('unbounded zero-usage retries')


if __name__ == '__main__':
    unittest.main()

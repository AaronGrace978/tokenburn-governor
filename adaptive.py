"""Verification-driven reasoning allocation; no model confidence required."""
from dataclasses import dataclass
import sqlite3


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Plan:
    tokens: int
    reason: str


class AdaptiveGovernor:
    """Learn cheap routes from external checks, scoped by exact task identity.

    Keys must include model, prompt/tool versions, task family and relevant
    input constraints. Never use a vague topic as a key. Store metadata only;
    this is not an answer cache. Use a separate instance per concurrent worker.
    """

    def __init__(self, history_path=":memory:", *, initial_tokens=256,
                 familiar_tokens=64, max_tokens=4096, required_successes=3, max_rounds=4):
        for name, value in (("initial_tokens", initial_tokens),
                            ("familiar_tokens", familiar_tokens),
                            ("max_tokens", max_tokens),
                            ("required_successes", required_successes), ("max_rounds", max_rounds)):
            _integer(name, value, 1)
        if not familiar_tokens <= initial_tokens <= max_tokens:
            raise ValueError("require familiar_tokens <= initial_tokens <= max_tokens")
        self.initial_tokens = initial_tokens
        self.familiar_tokens = familiar_tokens
        self.max_tokens = max_tokens
        self.required_successes = required_successes
        self.max_rounds = max_rounds
        self.db = sqlite3.connect(history_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS history "
                        "(task_key TEXT PRIMARY KEY, streak INTEGER NOT NULL)")
        self.db.commit()

    def close(self):
        self.db.close()

    def start(self, task_key, *, high_stakes=False):
        if not isinstance(task_key, str) or not task_key.strip():
            raise ValueError("task_key must be a nonempty versioned identity")
        if type(high_stakes) is not bool:
            raise ValueError("high_stakes must be bool")
        row = self.db.execute("SELECT streak FROM history WHERE task_key=?",
                              (task_key,)).fetchone()
        familiar = row is not None and row[0] >= self.required_successes
        budget = self.familiar_tokens if familiar and not high_stakes else self.initial_tokens
        return Attempt(self, task_key, budget,
                       "verified_familiarity" if familiar and not high_stakes else "initial_probe",
                       high_stakes)

    def _record(self, key, cheap_success):
        with self.db:
            self.db.execute(
                "INSERT INTO history(task_key, streak) VALUES (?, ?) "
                "ON CONFLICT(task_key) DO UPDATE SET streak = "
                "CASE WHEN excluded.streak=1 THEN history.streak+1 ELSE 0 END",
                (key, int(cheap_success)))


class Attempt:
    """Sequential probe, verify, escalate loop with a cumulative token ceiling."""

    def __init__(self, owner, key, budget, reason, high_stakes):
        self.owner, self.key = owner, key
        self.budget, self.reason = budget, reason
        self.high_stakes = high_stakes
        self.spent = 0
        self.rounds = 0
        self.pending = None
        self.done = False
        self.stop_reason = None

    def next_plan(self):
        if self.pending is not None:
            raise RuntimeError("report usage and verification before requesting another plan")
        if self.done:
            return Plan(0, self.stop_reason)
        tokens = min(self.budget, self.owner.max_tokens - self.spent)
        if tokens <= 0:
            self.done, self.stop_reason = True, "budget_exhausted"
            return Plan(0, self.stop_reason)
        self.pending = tokens
        return Plan(tokens, self.reason)

    def report(self, tokens_used, *, verified):
        """verified is True/False from an external check, or None if unavailable.

        Unknown checks stop for review rather than buying blind retries.
        Report actual usage even if the provider exceeded the requested limit.
        """
        if self.pending is None:
            raise RuntimeError("request a plan before reporting its result")
        _integer("tokens_used", tokens_used)
        if verified is not None and type(verified) is not bool:
            raise ValueError("verified must be True, False, or None")
        limit = self.pending
        self.pending = None
        self.spent += tokens_used
        self.rounds += 1
        # An adapter overshoot must never teach the cheap route as successful.
        cheap = (verified is True and self.rounds == 1 and tokens_used <= limit
                 and not self.high_stakes)
        self.owner._record(self.key, cheap)
        if verified is True:
            self.done, self.stop_reason = True, "verified"
        elif self.spent >= self.owner.max_tokens:
            self.done, self.stop_reason = True, "budget_exhausted"
        elif verified is None:
            self.done, self.stop_reason = True, "verification_required"
        elif self.rounds >= self.owner.max_rounds:
            self.done, self.stop_reason = True, "attempt_limit"
        else:
            self.budget *= 2
            self.reason = "failed_check_escalation"
        return self.stop_reason or "retry"

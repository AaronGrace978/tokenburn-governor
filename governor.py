"""TokenBurn Governor — entropy-gated thinking-budget control.

THE MATH (the innovation):
Thinking tokens are billed as OUTPUT (the expensive tier, e.g. $20/1M for
frontier models). Most of that burn is redundant re-derivation. This module
treats thinking as a stochastic process and applies three control laws:

1. ENTROPY GATE (Information Gain per Token):
   Let H_t be the Shannon entropy of the model's candidate-action
   distribution at step t. Thinking is only worth tokens when uncertainty
   is HIGH:
       IGT(t) = (H_prev - H_t) / tokens_spent_t   # info gain per token
   Continue thinking while IGT > tau (a cost-calibrated threshold).
   When IGT collapses (the model is re-deriving what it already knows),
   the governor forces an answer. This is a stopping rule in the same
   family as optimal experiment design / Chernoff stopping.

2. VALUE-OF-INFORMATION PRICING:
   Expected value of one more thinking token:
       VOI_t = p(decision_flips) * answer_value - token_price
   where p(flip) ~ tanh(entropy drop rate). Stop when VOI < 0 — think
   exactly until the marginal value of thought equals the marginal price
   of a token. Economically optimal stopping.

3. DECAY-COMPRESSED THOUGHT CACHE (geometric prior):
   Repeated reasoning over the same context decays as:
       C_n = C0 * r^n,  r in (0,1)  (measured redundancy factor)
   So the budget for a familiar context shrinks geometrically — the agent
   never pays full price twice for the same thought.

Dependency-free Python 3. Run tests: python3 test_governor.py
"""

import math


def shannon_entropy(probs):
    """H = -sum p log p (nats)."""
    return -sum(p * math.log(p) for p in probs if p > 0)


def information_gain_per_token(H_prev, H_now, tokens_spent):
    """IGT = entropy reduction / tokens spent."""
    if tokens_spent <= 0:
        return float("inf")
    return (H_prev - H_now) / tokens_spent


def value_of_information(dH_rate, answer_value, token_price):
    """VOI = p(decision flips) * answer value - token cost.

    p(flip) ~ tanh(dH_rate): if entropy is dropping fast, more thought
    still has a chance of changing the answer; if flat, it won't.
    """
    p_flip = math.tanh(max(0.0, dH_rate))
    return p_flip * answer_value - token_price


def decay_budget(C0, r, n):
    """Geometric thought-cache budget: familiar context => cheaper thought."""
    return C0 * (r ** n)


def tau_threshold(token_price, answer_value):
    """Cost-calibrated stopping threshold.

    tau scales with the price of thinking relative to the value of the
    answer: expensive tokens or cheap answers => stop thinking sooner.
    """
    return token_price / max(answer_value, 1e-9)


class Governor:
    """Runs the stopping rule over a stream of thinking steps."""

    def __init__(self, token_price=20.0 / 1_000_000, answer_value=1.0,
                 tau=None, min_steps=1):
        self.token_price = token_price
        self.answer_value = answer_value
        self.tau = tau if tau is not None else tau_threshold(token_price, answer_value)
        self.min_steps = min_steps
        self.H_prev = None
        self.tokens_spent = 0
        self.steps = 0

    def observe(self, probs, tokens_this_step):
        """Feed one thinking step's candidate distribution + token count.
        Returns 'THINK' or 'STOP'."""
        H = shannon_entropy(probs)
        self.steps += 1
        self.tokens_spent += tokens_this_step
        if self.H_prev is None or self.steps <= self.min_steps:
            self.H_prev = H
            return "THINK"
        igt = information_gain_per_token(self.H_prev, H, tokens_this_step)
        dH_rate = (self.H_prev - H) / max(tokens_this_step, 1)
        voi = value_of_information(dH_rate, self.answer_value, self.token_price)
        self.H_prev = H
        # Stop when info gain per token falls below the cost-calibrated
        # threshold AND further thought is not worth its price.
        if igt < self.tau and voi <= 0:
            return "STOP"
        return "THINK"

    def budget_for(self, C0, r, n):
        """Thought-cache budget for a context seen n times before."""
        return decay_budget(C0, r, n)

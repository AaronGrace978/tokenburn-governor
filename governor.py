"""Provider-neutral controls for reasoning token budgets.

Use allowance() before a request and observe() after receiving usage.
Optional probability distributions enable an entropy-based stopping heuristic.
Entropy reduction does not establish answer correctness or calibrated value.
No network calls or provider cancellation are performed by this module.

Dependency-free Python 3. Run tests: python3 test_governor.py
"""

import math


def shannon_entropy(probs):
    """H = -sum p log p (nats)."""
    probs = tuple(probs)
    if (not probs or any(not math.isfinite(p) or p < 0 for p in probs)
            or not math.isclose(sum(probs), 1.0, abs_tol=1e-9)):
        raise ValueError("probs must be a finite probability distribution summing to 1")
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
                 tau=None, min_steps=1, *, max_tokens=None, max_cost=None,
                 patience=1):
        for name, value in (("token_price", token_price), ("answer_value", answer_value),
                            ("tau", tau), ("max_cost", max_cost)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(name + " must be finite and nonnegative")
        for name, value, minimum in (("min_steps", min_steps, 0),
                                     ("max_tokens", max_tokens, 0),
                                     ("patience", patience, 1)):
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(name + " must be an integer >= " + str(minimum))
        if patience is None or min_steps is None:
            raise ValueError("patience and min_steps must be integers")
        self.token_price = token_price
        self.answer_value = answer_value
        self.tau = tau if tau is not None else tau_threshold(token_price, answer_value)
        self.min_steps = min_steps
        self.H_prev = None
        self.tokens_spent = 0
        self.steps = 0
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.patience = patience
        self.stalled_steps = 0
        self.stop_reason = None

    @property
    def cost_spent(self):
        """Estimated reasoning cost, excluding input and answer tokens."""
        return self.tokens_spent * self.token_price

    def _stop(self, reason):
        self.stop_reason = reason
        return "STOP"

    def _budget_reason(self):
        if self.max_tokens is not None and self.tokens_spent >= self.max_tokens:
            return "token_budget"
        if self.max_cost is not None and self.cost_spent >= self.max_cost:
            return "cost_budget"
        return None

    def allowance(self, requested_tokens):
        """Cap the NEXT request before spending; pass result to provider limits.

        Does not reserve tokens. Report each completed request to observe()
        before requesting another allowance. Zero means do not make the call.
        """
        if type(requested_tokens) is not int or requested_tokens < 0:
            raise ValueError("requested_tokens must be a nonnegative integer")
        if self.stop_reason is not None:
            return 0
        reason = self._budget_reason()
        if reason:
            self._stop(reason)
            return 0
        result = requested_tokens
        if self.max_tokens is not None:
            result = min(result, self.max_tokens - self.tokens_spent)
        if self.max_cost is not None and self.token_price > 0:
            remaining = max(0, math.floor(self.max_cost / self.token_price)
                            - self.tokens_spent)
            result = min(result, remaining)
        return result

    def observe(self, probs=None, tokens_this_step=0):
        """Feed one thinking step's candidate distribution + token count.
        Returns 'THINK' or 'STOP'."""
        if type(tokens_this_step) is not int or tokens_this_step <= 0:
            raise ValueError("tokens_this_step must be a positive integer")
        H = shannon_entropy(probs) if probs is not None else None
        self.steps += 1
        self.tokens_spent += tokens_this_step
        if self.stop_reason is not None:
            return "STOP"
        reason = self._budget_reason()
        if reason:
            return self._stop(reason)
        if H is None:
            self.H_prev = None
            self.stalled_steps = 0
            return "THINK"
        if self.H_prev is None or self.steps <= self.min_steps:
            self.H_prev = H
            return "THINK"
        igt = information_gain_per_token(self.H_prev, H, tokens_this_step)
        dH_rate = (self.H_prev - H) / max(tokens_this_step, 1)
        voi = value_of_information(dH_rate, self.answer_value, self.token_price)
        self.H_prev = H
        # Stop when info gain per token is at or below the cost-calibrated
        # threshold AND further thought is not worth its price.
        if igt <= self.tau and voi <= 0:
            self.stalled_steps += 1
            if self.stalled_steps >= self.patience:
                return self._stop("stalled_reasoning")
        else:
            self.stalled_steps = 0
        return "THINK"

    def budget_for(self, C0, r, n):
        """Thought-cache budget for a context seen n times before."""
        return decay_budget(C0, r, n)

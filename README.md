# TokenBurn Governor 🦖

Provider-neutral controls to limit reasoning spend across model calls.
Dependency-free Python 3.

## Start cheaply, escalate on failed checks

`AdaptiveGovernor` allocates reasoning before a call. It starts with a small
probe, learns a cheaper starting route after three externally verified
first-attempt successes, and doubles the next allowance only after a failed
check. Any failure revokes familiarity. Unknown verification stops for review.

```python
from adaptive import AdaptiveGovernor

controller = AdaptiveGovernor("/tmp/reasoning-history.sqlite",
                              initial_tokens=256, familiar_tokens=64,
                              max_tokens=4096, max_rounds=4)
attempt = controller.start("model-version:prompt-v2:tools-v1:task-family:constraints")
plan = attempt.next_plan()
# Your adapter calls the model with plan.tokens as its reasoning limit.
# Then run an independent check (tests, schema validation, or a task verifier).
result = attempt.report(tokens_used=100, verified=False)
if result == "retry":
    plan = attempt.next_plan()  # larger allowance, within remaining total budget
controller.close()
```

The adapter must carry the task and failed-check feedback into each retry.
Do not make a call when a plan returns zero. A `verified` result ends the attempt;
`verification_required`, `attempt_limit`, and `budget_exhausted` require stopping
or human review. No answer is accepted merely because its budget ran out.
The total ceiling counts actual reasoning usage across retries, including
provider overshoots; the controller cannot undo already billed usage.

History persists in SQLite when a path is supplied (default is in-memory).
It stores identity keys and success streaks, not prompts or answers. Keep it
outside the checkout. Scope keys to model/prompt/tool versions and task
constraints; change the key when these change. Familiarity is a routing prior,
not proof a smaller budget will work. Failures reset it, and every answer still
needs verification. `high_stakes=True` disables the learned cheap route, but
does not replace domain-specific safety checks. Use one controller per worker;
parallel budget reservation and automatic semantic task classification are not
implemented. This controller learns budget routing, not reusable answer content.

This replaces unverified entropy confidence with measurable task outcomes in
the adaptive workflow. It is a testable policy, not a claim of novel research
or demonstrated model savings. Compare total billed cost and verifier pass
rates against fixed-budget baselines on your own tasks.

## Control the next call's budget

```python
from governor import Governor

g = Governor(
    token_price=20.0 / 1_000_000,  # illustrative dollars per reasoning token
    max_tokens=1000,
    max_cost=0.02,              # reasoning-only dollar budget
    patience=2,                # consecutive uninformative steps before STOP
)

limit = g.allowance(500)
if limit > 0:
    # Pass limit to your substrate's supported token budget parameter.
    # Make the call, then report its ACTUAL incremental reasoning usage:
    verdict = g.observe(tokens_this_step=200)
    print(verdict, g.cost_spent, g.stop_reason)
```

Call `allowance()` before each sequential request and skip the call if it returns
zero. It returns the smallest of the requested tokens, remaining token budget,
and affordable whole tokens. It does not reserve tokens: report actual usage
before making another call. Use a separate Governor per request/session budget;
concurrent callers need external synchronization and reservation.

`observe()` accepts positive integer token increments, not cumulative totals.
With no probabilities, only budget limits apply. With candidate probabilities,
`observe(probs=[0.9, 0.1], tokens_this_step=50)` also checks entropy reduction
per token and a value-of-information heuristic. Probabilities must be finite,
nonnegative, and sum to one. Many APIs do not expose suitable probabilities;
do not fabricate them from hidden reasoning or confidence text.

Hard budgets override `min_steps`. STOP remains in effect for that Governor.
`stop_reason` is `token_budget`, `cost_budget`, or `stalled_reasoning`.
`patience` tolerates transient plateaus and resets when progress resumes.
Omitting limits preserves the original unbounded budget behavior. Select limits
for your workload; the library makes no measured cost-saving or quality guarantee.

## Integration limits

This library advises your substrate; it cannot interrupt a provider request by
itself. Apply provider-supported reasoning/output limits or cancellation in your
adapter. Usage reported after a call is already billed: an oversized completed
call can exceed the budget. Reserve input and final-answer costs separately;
`cost_spent` estimates reasoning cost at the configured fixed per-token price.
Provider billing, cached tokens, and combined output caps may differ.

Entropy reduction is a heuristic, not proof of correctness. Its tanh-based
value estimate is not calibrated to a model's probability of improving an answer.
Validate savings and answer quality on representative tasks before deployment.
`decay_budget(C0, r, n)` computes a geometric budget; it does not implement a
persistent thought cache or automatically apply that budget.

## Tests

```sh
PYTHONDONTWRITEBYTECODE=1 python3 test_governor.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_budgets
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_adaptive
```

The original seven tests cover entropy, information gain, value estimates,
stopping, and geometric budgets. Additional tests cover spending boundaries,
usage-only integration, patience, persistent stopping, and invalid inputs.

## License

GPL-3.0 — see [LICENSE](LICENSE).

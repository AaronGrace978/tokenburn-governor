# TokenBurn Governor 🦖

**Entropy-gated thinking budgets — stop frontier models from burning tokens on redundant thought.**

## The problem
Frontier models (Claude Opus, GPT-class) bill *thinking* tokens at the output rate — the most expensive tier (e.g. **$20 per 1M tokens**). Most of that spend is redundant: the model re-derives the same conclusions over and over. At scale, that's death by a thousand re-reasonings.

## The math
This repo treats reasoning as a stochastic process and applies three control laws:

### 1. Entropy Gate — Information Gain per Token (IGT)
Let `H_t` be the Shannon entropy of the model's candidate-action distribution at step `t`.

```
IGT(t) = (H_prev − H_t) / tokens_spent
```

Thinking is only worth tokens while it is *reducing uncertainty*. When IGT falls below a cost-calibrated threshold `tau`, the governor forces an answer. This is a stopping rule in the family of Chernoff stopping / optimal experiment design.

### 2. Value-of-Information Pricing (VOI)
```
VOI = p(decision_flips) × answer_value − token_price
```

`p(flip)` is estimated from the entropy-drop rate (`tanh(dH/dtokens)`). Stop when `VOI < 0` — think exactly until the marginal value of thought equals the marginal price of a token. Economically optimal stopping.

### 3. Decay-Compressed Thought Cache
```
C_n = C₀ · rⁿ,  r ∈ (0,1)
```

Contexts the agent has reasoned about before get geometrically smaller thinking budgets. Never pay full price twice for the same thought.

## Usage
```python
from governor import Governor

g = Governor(token_price=20.0 / 1_000_000, answer_value=1.0)
verdict = g.observe(probs=[0.9, 0.1], tokens_this_step=50)
# 'THINK' or 'STOP'
```

## Tests
```
python3 test_governor.py
```

All seven tests pass: entropy math, IGT, VOI pricing, decay budgets, stopping on redundant thought, continuing while informative, and geometric cache budgets.

## License
GPL-3.0 — see [LICENSE](LICENSE).

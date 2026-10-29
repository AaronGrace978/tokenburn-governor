"""Tests for the TokenBurn Governor. Run: python3 test_governor.py"""

from governor import (shannon_entropy, information_gain_per_token,
                      value_of_information, decay_budget, tau_threshold, Governor)


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_entropy():
    assert approx(shannon_entropy([0.5, 0.5]), math.log(2))
    assert approx(shannon_entropy([1.0]), 0.0)
    print("ok: entropy")


def test_igt():
    # Entropy dropped from 1.0 to 0.5 over 10 tokens => 0.05 nats/token
    assert approx(information_gain_per_token(1.0, 0.5, 10), 0.05)
    assert information_gain_per_token(1.0, 1.0, 10) == 0.0  # no gain
    print("ok: information gain per token")


def test_voi():
    # Flat entropy rate => p_flip ~ 0 => VOI negative (stop thinking)
    assert value_of_information(0.0, 1.0, 1e-5) < 0
    # Fast entropy drop => positive VOI (keep thinking)
    assert value_of_information(2.0, 1.0, 1e-5) > 0
    print("ok: value of information")


def test_decay_budget():
    # Context seen 3 times with r=0.5 => 1/8 of the original budget
    assert approx(decay_budget(1000, 0.5, 3), 125)
    print("ok: decay-compressed thought cache")


def test_governor_stops_on_redundant_thought():
    g = Governor(token_price=20.0 / 1_000_000, answer_value=1.0)
    # Step 1: high uncertainty
    assert g.observe([0.5, 0.5], 50) == "THINK"
    # Step 2: big entropy drop => still thinking is informative
    assert g.observe([0.9, 0.1], 50) == "THINK"
    # Step 3: same distribution as step 2 — zero new information,
    # the model is re-deriving what it already knows => governor stops the burn
    assert g.observe([0.9, 0.1], 50) == "STOP"
    print("ok: governor stops redundant thinking")


def test_governor_keeps_thinking_when_informative():
    g = Governor()
    assert g.observe([0.25, 0.25, 0.25, 0.25], 50) == "THINK"
    assert g.observe([0.4, 0.3, 0.2, 0.1], 50) == "THINK"
    assert g.observe([0.7, 0.2, 0.05, 0.05], 50) == "THINK"
    print("ok: governor keeps thinking while entropy still drops")


def test_budget_for_familiar_context():
    g = Governor()
    assert g.budget_for(2000, 0.6, 0) == 2000
    assert approx(g.budget_for(2000, 0.6, 4), 2000 * 0.6 ** 4)
    print("ok: familiar contexts get geometrically smaller budgets")


if __name__ == "__main__":
    import math
    test_entropy()
    test_igt()
    test_voi()
    test_decay_budget()
    test_governor_stops_on_redundant_thought()
    test_governor_keeps_thinking_when_informative()
    test_budget_for_familiar_context()
    print("\nALL TESTS PASSED")

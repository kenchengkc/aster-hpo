from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")


def test_self_financing_pnl_includes_open_and_liquidation_costs():
    from benchmarks.hedging import hedge_pnl

    paths = np.array([[1.0, 1.1, 1.2]])
    positions = np.array([[0.5, 0.5]])
    pnl, costs, turnover = hedge_pnl(paths, positions, cost_rate=0.01)
    assert pnl[0] == pytest.approx(0.089)
    assert costs[0] == pytest.approx(0.011)
    assert turnover[0] == pytest.approx(1.1)
    zero, costs, _ = hedge_pnl(paths, np.zeros_like(positions), cost_rate=0.01)
    assert zero[0] == 0 and costs[0] == 0


def test_zero_volatility_and_frictionless_delta_sanity():
    from benchmarks.hedging import Market, call_price, delta_positions, hedge_pnl, simulate_paths

    zero = Market(volatility=0)
    paths = simulate_paths(zero, 20, np.random.default_rng(4))
    assert np.all(paths == 1)
    assert call_price(zero) == 0
    market = Market(steps=64, transaction_cost=0)
    paths = simulate_paths(market, 5000, np.random.default_rng(4))
    pnl, _, _ = hedge_pnl(paths, delta_positions(paths, market), cost_rate=0)
    payoff = np.maximum(paths[:, -1] - market.strike, 0)
    error = call_price(market) + pnl - payoff
    unhedged = call_price(market) - payoff
    assert np.mean(error**2) < 0.1 * np.mean(unhedged**2)
    assert call_price(market) == pytest.approx(0.07965567455405798)


def test_policy_cannot_look_at_future_prices():
    from benchmarks.hedging import Market, initialize, positions, simulate_paths

    market = Market(steps=6)
    paths = simulate_paths(market, 3, np.random.default_rng(4))
    parameters = initialize(4, np.random.default_rng(5))
    before = positions(parameters, paths, market)
    paths[:, 3:] *= 1.5
    after = positions(parameters, paths, market)
    np.testing.assert_array_equal(before[:, :3], after[:, :3])
    assert np.all((after >= 0) & (after <= 1))


def test_training_gradient_matches_finite_differences_with_costs():
    from benchmarks.hedging import Market, initialize, loss_and_gradient, simulate_paths

    market = Market(steps=4, transaction_cost=0.01)
    paths = simulate_paths(market, 7, np.random.default_rng(19))
    parameters = initialize(3, np.random.default_rng(3))
    _, gradients = loss_and_gradient(parameters, paths, market, regularization=0.002)
    for name, values in parameters.items():
        for index in np.ndindex(values.shape):
            original = values[index]
            values[index] = original + 1e-6
            upper, _ = loss_and_gradient(parameters, paths, market, regularization=0.002)
            values[index] = original - 1e-6
            lower, _ = loss_and_gradient(parameters, paths, market, regularization=0.002)
            values[index] = original
            assert gradients[name][index] == pytest.approx(
                (upper - lower) / 2e-6, abs=1e-7, rel=1e-4
            )


def test_training_is_independent_of_evaluation_seed_and_learns():
    from aster_hpo import Replication
    from benchmarks.hedging import (
        Market,
        evaluate_policy,
        initialize,
        simulate_paths,
        train_policy,
    )

    market = Market(epochs=64, batch_paths=128, evaluation_paths=1024)
    config = {"width": 8, "learning_rate": 0.03, "regularization": 0.0001}
    replication = Replication.create(1000, config, 0, "pilot", True)
    trained = train_policy(config, replication, market)
    other = train_policy(config, replace(replication, scenario_seed=999), market)
    for key in trained:
        np.testing.assert_array_equal(trained[key], other[key])
    initial = initialize(
        8, np.random.default_rng(np.random.SeedSequence(replication.training_seed).spawn(2)[0])
    )
    paths = simulate_paths(market, 1024, np.random.default_rng(99))
    assert (
        evaluate_policy(trained, paths, market)["mse"]
        < evaluate_policy(initial, paths, market)["mse"]
    )


def test_objective_repeats_and_exports_finite_diagnostics(tmp_path):
    from aster_hpo import Replication
    from benchmarks.hedging import HedgingObjective, Market

    config = {"width": 4, "learning_rate": 0.03, "regularization": 0.0001}
    objective = HedgingObjective(Market(epochs=3, evaluation_paths=32), tmp_path)
    rep = Replication.create(7, config, 0, "audit", True)
    assert objective(config, rep) == objective(config, rep)
    assert np.isfinite(objective(config, rep))
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volatility": -1},
        {"steps": 0},
        {"spot": float("nan")},
        {"epochs": True},
        {"transaction_cost": -0.01},
    ],
)
def test_invalid_market_rejected(kwargs):
    from benchmarks.hedging import Market

    with pytest.raises(ValueError):
        Market(**kwargs)


def test_objective_backend_is_explicit_serializable_and_lazy():
    import pickle

    from benchmarks.hedging import HedgingObjective, Market

    objective = HedgingObjective(Market(epochs=1), backend="cuda")
    assert pickle.loads(pickle.dumps(objective)) == objective
    with pytest.raises(ValueError, match="backend"):
        HedgingObjective(backend="automatic")


def test_shared_config_validation_preserves_training_constraints():
    from benchmarks.hedging import validate_config

    config = {"width": 4, "learning_rate": 0.03, "regularization": 0.0001}
    assert validate_config(config) == (4, 0.03, 0.0001)
    for field, value in (("width", True), ("learning_rate", 0), ("regularization", -1)):
        with pytest.raises(ValueError):
            validate_config({**config, field: value})


def test_requested_cuda_does_not_fall_back_to_cpu_when_unavailable():
    import importlib.util

    from aster_hpo import Replication
    from benchmarks.hedging import HedgingObjective, Market

    if importlib.util.find_spec("cupy") is not None:
        pytest.skip("This check covers the optional dependency missing on CPU hosts")
    config = {"width": 4, "learning_rate": 0.03, "regularization": 0.0001}
    objective = HedgingObjective(Market(epochs=1, batch_paths=2), backend="cuda")
    with pytest.raises(RuntimeError, match="CuPy|CUDA|cupy"):
        objective(config, Replication.create(4, config, 0, "search", True))


def test_eager_cuda_reference_is_explicit_and_serializable():
    import pickle

    from benchmarks.hedging import HedgingObjective

    objective = HedgingObjective(backend="cuda_eager")
    assert pickle.loads(pickle.dumps(objective)).backend == "cuda_eager"
    from benchmarks.hedging_cuda import train_policy_eager

    assert callable(train_policy_eager)

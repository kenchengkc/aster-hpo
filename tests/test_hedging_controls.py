import pytest

np = pytest.importorskip("numpy")


def test_strategies_share_payoff_capital_and_round_trip_fees():
    from benchmarks.hedging import Market, call_price, delta_positions, initialize
    from examples.hedging_controls import compare_hedges

    market = Market(steps=1, transaction_cost=0.01)
    paths = np.array([[1.0, 1.2], [1.0, 0.9]])
    # Zero output weights fix the neural position at half a share on both paths.
    parameters = initialize(2, np.random.default_rng(1))
    parameters["w2"].fill(0)
    rows = compare_hedges(parameters, paths, market)
    shares = {
        "Fixed neural hedge": 0.5,
        "Black-Scholes delta": delta_positions(paths, market)[0, 0],
        "No hedge": 0.0,
    }
    assert rows.keys() == shares.keys()
    for strategy, holding in shares.items():
        turnover = holding * np.array([2.2, 1.9])  # Open at 1 and close at the final price.
        costs = 0.01 * turnover
        error = call_price(market) + holding * np.array([0.2, -0.1]) - costs - np.array([0.2, 0])
        assert rows[strategy] == pytest.approx(
            {
                "mse": np.mean(error**2),
                "mean_transaction_cost": np.mean(costs),
                "mean_turnover": np.mean(turnover),
            }
        )

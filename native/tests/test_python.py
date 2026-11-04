"""Public native boundaries: wrong math, hidden copies, aliasing, and state corruption."""

import importlib.util
from decimal import Decimal, localcontext

import numpy as np
import pytest

from benchmarks.hedging import hedge_pnl as reference


def test_companion_is_installable():
    assert importlib.util.find_spec("aster_hpo_native") is not None


@pytest.fixture
def native():
    import aster_hpo_native

    return aster_hpo_native


def inputs(count=5, intervals=12):
    rng = np.random.default_rng(71)
    return rng.uniform(0.5, 2, (count, intervals + 1)), rng.uniform(-1, 1, (count, intervals))


def test_hand_calculated_outputs_and_ownership(native):
    prices = np.array([[1.0, 1.1, 1.2]])
    holdings = np.array([[0.5, 0.5]])
    prices.flags.writeable = False
    holdings.flags.writeable = False
    actual = native.hedge_pnl(prices, holdings, cost_rate=0.01)
    for array, expected in zip(actual, (0.089, 0.011, 1.1), strict=True):
        assert array[0] == pytest.approx(expected)
        assert array.flags.owndata
        assert array.dtype == np.float64
    del prices, holdings
    assert actual[0][0] == pytest.approx(0.089)


@pytest.mark.parametrize("count,intervals", [(0, 1), (1, 1), (7, 12), (129, 64), (2048, 256)])
@pytest.mark.parametrize("rate", [0.0, 0.01])
def test_identical_inputs_match_numpy_and_preallocated_outputs(native, count, intervals, rate):
    prices, holdings = inputs(count, intervals)
    saved = prices.copy(), holdings.copy()
    expected = reference(prices, holdings, cost_rate=rate)
    actual = native.hedge_pnl(prices, holdings, cost_rate=rate)
    output = tuple(np.empty(count) for _ in range(3))
    native.hedge_pnl_into(prices, holdings, *output, cost_rate=rate)
    for wanted, allocated, preallocated in zip(expected, actual, output, strict=True):
        np.testing.assert_allclose(allocated, wanted, atol=1e-12, rtol=1e-10)
        np.testing.assert_array_equal(allocated, preallocated)
    np.testing.assert_array_equal(prices, saved[0])
    np.testing.assert_array_equal(holdings, saved[1])


def test_cancelling_gains_and_zero_final_pnl(native):
    prices = np.array([[1.0, 2.0] * 128 + [1.0]])
    holdings = np.full((1, 256), 0.5)
    pnl, costs, turnover = native.hedge_pnl(prices, holdings, cost_rate=0)
    assert pnl[0] == 0
    assert costs[0] == 0
    assert turnover[0] == 1
    holdings[0, -1] += 1e-10
    np.testing.assert_allclose(
        native.hedge_pnl(prices, holdings, cost_rate=0)[0],
        reference(prices, holdings, cost_rate=0)[0],
        atol=1e-12,
        rtol=1e-10,
    )


def test_streaming_matches_batch_including_short_positions(native):
    prices, holdings = inputs(3, 64)
    expected = native.hedge_pnl(prices, holdings, cost_rate=0.01)
    ledger = native.Ledger(1, 0.01)
    for index in range(3):
        ledger.reset(prices[index, 0], 0.01)
        for price, holding in zip(prices[index, :-1], holdings[index], strict=True):
            ledger.update(price, holding)
        result = ledger.finish(prices[index, -1])
        assert result.pnl == pytest.approx(expected[0][index], abs=1e-12, rel=1e-10)
        assert result.costs == pytest.approx(expected[1][index])
        assert result.turnover == pytest.approx(expected[2][index])
        with pytest.raises(RuntimeError):
            ledger.finish(prices[index, -1])
        with pytest.raises(RuntimeError):
            ledger.update(1, 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_failed_events_and_reset_preserve_ledger_state(native, bad):
    ledger = native.Ledger(1, 0.01)
    before = ledger.update(1.1, 0.5)
    for operation in (
        lambda: ledger.update(bad, 0),
        lambda: ledger.update(1, bad),
        lambda: ledger.finish(bad),
        lambda: ledger.reset(1, bad),
    ):
        with pytest.raises(ValueError):
            operation()
        result = ledger.result()
        assert (result.pnl, result.costs, result.turnover) == (
            before.pnl,
            before.costs,
            before.turnover,
        )


def test_overflow_is_an_error_without_mutating_ledger(native):
    ledger = native.Ledger(1, 1)
    before = ledger.update(1, 1e308)
    with pytest.raises(OverflowError):
        ledger.update(1e308, 1e308)
    assert ledger.result().pnl == before.pnl
    with pytest.raises(OverflowError):
        native.hedge_pnl(np.array([[1, 1e308]]), np.array([[1e308]]), cost_rate=0)


@pytest.mark.parametrize(
    "case", ["float32", "list", "noncontiguous", "unaligned", "shape", "zero_steps", "nan"]
)
def test_incompatible_inputs_are_rejected_without_conversion(native, case):
    prices, holdings = inputs()
    if case == "float32":
        prices = prices.astype(np.float32)
    elif case == "list":
        prices = prices.tolist()
    elif case == "noncontiguous":
        prices = prices[:, ::-1]
    elif case == "unaligned":
        prices = np.ndarray(
            prices.shape, dtype=np.float64, buffer=bytearray(prices.nbytes + 1), offset=1
        )
    elif case == "shape":
        holdings = holdings[:, :-1]
    elif case == "zero_steps":
        prices, holdings = np.ones((1, 1)), np.empty((1, 0))
    else:
        prices[0, 0] = np.nan
    with pytest.raises((TypeError, ValueError)):
        native.hedge_pnl(prices, holdings, cost_rate=0.01)


@pytest.mark.parametrize(
    "case", ["readonly", "float32", "shape", "overlap_input", "overlap_output"]
)
def test_output_contract_prevents_corruption(native, case):
    prices, holdings = inputs()
    outputs = [np.full(5, 42.0) for _ in range(3)]
    if case == "readonly":
        outputs[0].flags.writeable = False
    elif case == "float32":
        outputs[0] = outputs[0].astype(np.float32)
    elif case == "shape":
        outputs[0] = np.empty(4)
    elif case == "overlap_input":
        outputs[0] = prices.ravel()[:5]
    else:
        outputs[1] = outputs[0]
    saved_prices = prices.copy()
    with pytest.raises((TypeError, ValueError)):
        native.hedge_pnl_into(prices, holdings, *outputs, cost_rate=0.01)
    np.testing.assert_array_equal(prices, saved_prices)


@pytest.mark.parametrize("rate", [-1, np.nan, np.inf, True])
def test_invalid_cost_rate(native, rate):
    with pytest.raises((TypeError, ValueError)):
        native.hedge_pnl(*inputs(), cost_rate=rate)


def test_outside_domain_accounting_against_decimal_oracle(native):
    prices = np.array([[1e8, 1e8 + 1, 1e8 - 1, 1e8 + 0.5]])
    holdings = np.array([[0.25, -0.5, 0.125]])
    with localcontext() as context:
        context.prec = 60
        p = [Decimal.from_float(value) for value in prices[0]]
        h = [Decimal.from_float(value) for value in holdings[0]]
        gain = sum(h[t] * (p[t + 1] - p[t]) for t in range(3))
        turnover = (
            abs(h[0]) * p[0]
            + sum(abs(h[t] - h[t - 1]) * p[t] for t in range(1, 3))
            + abs(h[-1]) * p[-1]
        )
        cost = Decimal.from_float(0.001) * turnover
    actual = native.hedge_pnl(prices, holdings, cost_rate=0.001)
    for result, expected in zip(actual, (gain - cost, cost, turnover), strict=True):
        assert result[0] == pytest.approx(float(expected), rel=1e-14, abs=1e-8)

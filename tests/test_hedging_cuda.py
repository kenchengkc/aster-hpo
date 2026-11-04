"""The optional resident backend is checked on a real CUDA device when present."""

import gc
import importlib
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"width": 4, "learning_rate": 0.03, "regularization": 0.0001}


def test_cuda_module_imports_without_loading_optional_dependency():
    assert importlib.util.find_spec("benchmarks.hedging_cuda") is not None
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.hedging_cuda; assert 'cupy' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_cuda_request_explains_missing_optional_dependency():
    if importlib.util.find_spec("cupy") is not None:
        pytest.skip("CuPy is installed")
    backend = importlib.import_module("benchmarks.hedging_cuda")
    with pytest.raises(RuntimeError, match="CuPy.*CUDA device"):
        backend.synchronize()


def test_shared_kernel_executes_literal_open_and_liquidation_case(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is unavailable")
    source = (ROOT / "cuda/hedge_accounting.cu").read_text()
    assert '#include "accounting_kernel.cuh"' in source
    assert "__global__ void accounting_kernel" not in source
    program = tmp_path / "shared.cpp"
    program.write_text(
        "#include <cmath>\n#define __global__\n"
        "struct Index { unsigned x; };\n"
        "Index blockIdx{0}, blockDim{128}, threadIdx{0};\n"
        '#include "accounting_kernel.cuh"\n'
        "int main() { double s[]{1,1.1,1.2}, h[]{.5,.5}, p[1],c[1],t[1];\n"
        "accounting_kernel(s,h,1,2,.01,p,c,t);\n"
        "return !(fabs(p[0]-.089)<1e-14 && fabs(c[0]-.011)<1e-14 "
        "&& fabs(t[0]-1.1)<1e-14); }\n"
    )
    binary = tmp_path / "shared"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-I", str(ROOT / "cuda/include"), str(program), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    assert subprocess.run([str(binary)], check=False).returncode == 0


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if count == 0:
        pytest.skip("CUDA device unavailable")
    yield cp, importlib.import_module("benchmarks.hedging_cuda")
    # CuPy retains freed arrays in reusable pools. Release those caches before
    # context teardown so memcheck measures live allocations, not retained capacity.
    cp.cuda.get_current_stream().synchronize()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.mark.parametrize("count", [0, 1, 129])
def test_accounting_matches_reference_including_partial_blocks_and_empty(gpu, count):
    from benchmarks.hedging import Market, hedge_pnl, simulate_paths

    cp, backend = gpu
    market = Market(steps=5)
    paths = simulate_paths(market, count, np.random.default_rng(5))
    holdings = np.random.default_rng(6).uniform(-1, 1, (count, market.steps))
    expected = hedge_pnl(paths, holdings, cost_rate=0.01)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        actual = backend.accounting(cp.asarray(paths), cp.asarray(holdings), cost_rate=0.01)
    stream.synchronize()
    for want, got in zip(expected, actual, strict=True):
        np.testing.assert_allclose(cp.asnumpy(got), want, rtol=1e-12, atol=1e-12)


def test_accounting_literal_costs_and_zero_sign(gpu):
    cp, backend = gpu
    outputs = backend.accounting(
        cp.array([[1.0, 1.1, 1.2]]),
        cp.array([[0.5, 0.5]]),
        cost_rate=0.01,
    )
    np.testing.assert_allclose([float(x[0]) for x in outputs], [0.089, 0.011, 1.1])
    zero = backend.accounting(cp.ones((1, 3)), cp.zeros((1, 2)), cost_rate=0.01)
    assert all(float(value[0]) == 0 for value in zero)


@pytest.mark.parametrize("case", ["host", "dtype", "shape", "strided", "nan", "rate"])
def test_accounting_rejects_invalid_arrays(gpu, case):
    cp, backend = gpu
    paths, holdings, rate = cp.ones((3, 4)), cp.zeros((3, 3)), 0.01
    if case == "host":
        paths = np.ones((3, 4))
    elif case == "dtype":
        paths = paths.astype(cp.float32)
    elif case == "shape":
        holdings = cp.zeros((3, 2))
    elif case == "strided":
        paths = cp.ones((3, 8))[:, ::2]
    elif case == "nan":
        paths[0, 0] = cp.nan
    else:
        rate = -0.1
    with pytest.raises((ValueError, TypeError)):
        backend.accounting(paths, holdings, cost_rate=rate)


def test_gradient_and_loss_match_numpy_and_finite_differences(gpu):
    from benchmarks.hedging import Market, initialize, loss_and_gradient, simulate_paths

    cp, backend = gpu
    market = Market(steps=4, transaction_cost=0.01)
    paths = simulate_paths(market, 7, np.random.default_rng(19))
    parameters = initialize(3, np.random.default_rng(3))
    expected_loss, expected = loss_and_gradient(parameters, paths, market, regularization=0.002)
    device_parameters = {name: cp.asarray(value) for name, value in parameters.items()}
    device_paths = cp.asarray(paths)
    loss, gradients = backend.loss_and_gradient(
        device_parameters,
        device_paths,
        market,
        regularization=0.002,
    )
    assert float(loss) == pytest.approx(expected_loss, rel=1e-11, abs=1e-12)
    for name, values in device_parameters.items():
        np.testing.assert_allclose(cp.asnumpy(gradients[name]), expected[name], atol=1e-11)
        for index in np.ndindex(values.shape):
            original = float(values[index])
            values[index] = original + 1e-6
            upper, _ = backend.loss_and_gradient(
                device_parameters,
                device_paths,
                market,
                regularization=0.002,
            )
            values[index] = original - 1e-6
            lower, _ = backend.loss_and_gradient(
                device_parameters,
                device_paths,
                market,
                regularization=0.002,
            )
            values[index] = original
            assert float(gradients[name][index]) == pytest.approx(
                (float(upper) - float(lower)) / 2e-6,
                abs=1e-7,
                rel=1e-4,
            )


@pytest.mark.parametrize("epochs", [1, 8, 64])
def test_training_parameters_and_all_metrics_match_numpy(gpu, epochs):
    from aster_hpo import Replication
    from benchmarks.hedging import Market, evaluate_policy, simulate_paths, train_policy

    cp, backend = gpu
    market = Market(epochs=epochs, batch_paths=32, evaluation_paths=129)
    rep = Replication.create(1000, CONFIG, 0, "search", True)
    expected = train_policy(CONFIG, rep, market)
    actual = backend.train_policy(CONFIG, rep, market)
    for name in expected:
        np.testing.assert_allclose(cp.asnumpy(actual[name]), expected[name], rtol=1e-7, atol=1e-9)
    paths = simulate_paths(
        market,
        market.evaluation_paths,
        np.random.default_rng(rep.scenario_seed),
    )
    want = evaluate_policy(expected, paths, market)
    got = backend.evaluate_policy(actual, cp.asarray(paths), market)
    assert got.keys() == want.keys()
    for name in want:
        assert got[name] == pytest.approx(want[name], rel=1e-6, abs=1e-8)
    assert backend.evaluate(CONFIG, rep, market) == pytest.approx(got, rel=1e-9, abs=1e-11)


def test_training_ignores_scenario_seed_and_policy_ignores_future(gpu):
    from aster_hpo import Replication
    from benchmarks.hedging import Market, simulate_paths

    cp, backend = gpu
    market = Market(epochs=3, steps=6, batch_paths=16, evaluation_paths=32)
    rep = Replication.create(7, CONFIG, 0, "audit", True)
    parameters = backend.train_policy(CONFIG, rep, market)
    other = backend.train_policy(CONFIG, replace(rep, scenario_seed=999), market)
    for name in parameters:
        cp.testing.assert_array_equal(parameters[name], other[name])
    paths = cp.asarray(simulate_paths(market, 3, np.random.default_rng(4)))
    before = backend.positions(parameters, paths, market)
    paths[:, 3:] *= 1.5
    after = backend.positions(parameters, paths, market)
    cp.testing.assert_array_equal(before[:, :3], after[:, :3])
    assert bool(cp.all((after >= 0) & (after <= 1)))


def test_nonfinite_training_is_rejected(gpu):
    from aster_hpo import Replication
    from benchmarks.hedging import Market

    _, backend = gpu
    config = {**CONFIG, "learning_rate": 1e308}
    rep = Replication.create(7, config, 0, "search", True)
    with pytest.raises(FloatingPointError):
        backend.train_policy(config, rep, Market(epochs=3, batch_paths=8))


def test_cuda_objective_runs_in_real_spawned_study_and_audit(gpu, tmp_path):
    from aster_hpo import FullBudget, Study, audit
    from benchmarks.hedging import HedgingObjective, Market

    _, backend = gpu
    objective = HedgingObjective(
        Market(epochs=2, batch_paths=8, evaluation_paths=16),
        backend="cuda",
    )
    result = Study(
        [CONFIG],
        scheduler=FullBudget(2),
        workers=2,
        seed=78,
        storage=tmp_path / "search.sqlite",
        objective_id="test-cuda-resident",
    ).optimize(objective)
    assert result.best_trial is not None
    assert result.metadata["execution_backend"] == "supervised_processes"
    assert len(result.best_trial.observations) == 2
    assert all(o.error is None for o in result.best_trial.observations)
    checked = audit(objective, result.best_trial.config, replications=2, workers=1, seed=78)
    assert len(checked.observations) == 2
    assert all(o.error is None for o in checked.observations)
    backend.synchronize()
    info = backend.device_info()
    assert info["backend"] == "cuda"
    assert info["device_name"]


def test_cuda_frozen_study_runs_resumes_with_stable_device_identity(gpu, tmp_path):
    from benchmarks.common import write_json
    from benchmarks.hedging import Market
    from benchmarks.study import freeze_protocol, run_study

    _, backend = gpu
    market = Market(epochs=1, batch_paths=4, evaluation_paths=16)
    pilot = tmp_path / "pilot"
    write_json(
        pilot / "manifest.json",
        {
            "kind": "financial pilot",
            "seed": 10000,
            "backend": "cuda",
            "market": asdict(market),
        },
    )
    write_json(pilot / "summary.json", {"full_budget": {"audit_mean": 0.01}})
    protocol = tmp_path / "protocol.json"
    freeze_protocol(
        pilot,
        protocol,
        seeds=(0, 1),
        count=2,
        workers=1,
        checkpoints=(2, 4),
        audit_replications=2,
        scaling_workers=(),
        backend="cuda",
    )
    identity = backend.device_info()
    report = run_study(protocol, tmp_path / "study")
    assert all(row["failed_runs"] == 0 for row in report["methods"].values())
    assert backend.device_info() == identity
    assert run_study(protocol, tmp_path / "study", resume=True) == report

"""Independent checks for the optional fused float64 training path."""

import gc
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
NAMES = ("w1", "b1", "w2", "b2")


def test_fused_module_import_has_no_cuda_initialization():
    assert importlib.util.find_spec("benchmarks.hedging_cuda_fused") is not None
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.hedging_cuda_fused; assert 'cupy' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_kernel_arithmetic_runs_against_numpy_on_host(tmp_path):
    """One-thread CUDA emulation exercises every stage without a CUDA dependency."""
    from benchmarks.hedging import Market, loss_and_gradient

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    assert (ROOT / "cuda/include/hedging_training.cuh").is_file()
    program = tmp_path / "training.cpp"
    program.write_text(
        "#include <cmath>\n#include <cstdio>\n"
        "#define __global__\n#define __device__\n#define __forceinline__ inline\n"
        "#define __shared__\n#define __syncthreads() ((void)0)\n"
        "struct Index { unsigned x; }; Index blockIdx{0}, blockDim{1}, threadIdx{0};\n"
        '#include "hedging_training.cuh"\n'
        "int main() { double paths[]{1.,1.1,1.2}, w1[]{.2,-.1}, b1[]{.05},"
        "w2[]{.3}, b2[]{.1}, h[2], hold[2], feature[2], error[1],"
        "gw1[2],gb1[1],gw2[1],gb2[1],loss[1],"
        "m1[2]{},mb1[1]{},m2[1]{},mb2[1]{},"
        "v1[2]{},vb1[1]{},v2[1]{},vb2[1]{}; bool finite=true;\n"
        "for (unsigned i=0;i<2;++i) {blockIdx.x=i; "
        "hedging_forward(paths,w1,b1,w2,b2,1,2,1,1.,.2,feature,h,hold);}\n"
        "blockIdx.x=0; hedging_error(paths,hold,1,2,.01,.07965567455405798,1.,error);\n"
        "for (unsigned i=0;i<6;++i) {blockIdx.x=i; "
        "hedging_gradient(paths,w1,w2,feature,h,hold,error,1,2,1,.01,.002,"
        "gw1,gb1,gw2,gb2,loss);}\n"
        'printf("[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",'
        "loss[0],gw1[0],gw1[1],gb1[0],gw2[0],gb2[0]);\n"
        "blockIdx.x=0; hedging_adam(w1,b1,w2,b2,gw1,gb1,gw2,gb2,"
        "m1,mb1,m2,mb2,v1,vb1,v2,vb2,loss,&finite,1,.03,.1,.001);\n"
        'printf(",%.17g,%.17g,%.17g,%.17g,%.17g,%d]",'
        "w1[0],w1[1],b1[0],w2[0],b2[0],finite); }\n"
    )
    binary = tmp_path / "training"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-ffp-contract=off",
            "-I",
            str(ROOT / "cuda/include"),
            str(program),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    actual = json.loads(subprocess.check_output([str(binary)], text=True))
    parameters = {
        "w1": np.array([[0.2], [-0.1]]),
        "b1": np.array([0.05]),
        "w2": np.array([0.3]),
        "b2": np.array([0.1]),
    }
    loss, gradients = loss_and_gradient(
        parameters,
        np.array([[1.0, 1.1, 1.2]]),
        Market(steps=2, transaction_cost=0.01),
        regularization=0.002,
    )
    norm = np.sqrt(sum(np.sum(value**2) for value in gradients.values()))
    expected_update = []
    for name in NAMES:
        gradient = gradients[name] / max(norm, 1)
        expected_update.extend(
            (parameters[name] - 0.03 * gradient / (np.abs(gradient) + 1e-8)).ravel()
        )
    expected = [
        loss,
        *np.concatenate([gradients[name].ravel() for name in NAMES]),
        *expected_update,
        1,
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    yield cp, importlib.import_module("benchmarks.hedging_cuda_fused")
    cp.cuda.get_current_stream().synchronize()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.mark.parametrize("cost", [0.0, 0.01])
@pytest.mark.parametrize("regularization", [0.0, 0.002])
def test_fused_gradients_loss_and_finite_differences(gpu, cost, regularization):
    from benchmarks.hedging import Market, initialize, loss_and_gradient, simulate_paths

    cp, fused = gpu
    market = Market(steps=4, transaction_cost=cost)
    paths = simulate_paths(market, 7, np.random.default_rng(19))
    parameters = initialize(3, np.random.default_rng(3))
    expected_loss, expected = loss_and_gradient(
        parameters, paths, market, regularization=regularization
    )
    device_parameters = {name: cp.asarray(value) for name, value in parameters.items()}
    device_paths = cp.asarray(paths)
    workspace = fused.TrainingWorkspace(cp, 7, 4, 3)
    loss, gradients = fused.loss_and_gradient(
        device_parameters,
        device_paths,
        market,
        regularization=regularization,
        workspace=workspace,
    )
    assert float(loss) == pytest.approx(expected_loss, rel=1e-11, abs=1e-12)
    saved = {name: cp.asnumpy(value) for name, value in gradients.items()}
    for name, value in device_parameters.items():
        np.testing.assert_allclose(saved[name], expected[name], rtol=1e-11, atol=1e-12)
        for index in np.ndindex(value.shape):
            original = float(value[index])
            value[index] = original + 1e-6
            upper = float(
                workspace.compute(device_parameters, device_paths, market, regularization)[0]
            )
            value[index] = original - 1e-6
            lower = float(
                workspace.compute(device_parameters, device_paths, market, regularization)[0]
            )
            value[index] = original
            assert saved[name][index] == pytest.approx((upper - lower) / 2e-6, rel=1e-4, abs=1e-7)


@pytest.mark.parametrize("width,scale", [(3, 0.01), (3, 5.0), (67, 5.0), (3, 1e200)])
def test_fused_optimizer_matches_numpy_clipping_and_reuses_moments(gpu, width, scale):
    from benchmarks.hedging import initialize

    cp, fused = gpu
    rng = np.random.default_rng(5)
    original = initialize(width, rng)
    gradients = {name: rng.normal(size=value.shape) * scale for name, value in original.items()}
    parameters = {name: cp.asarray(value) for name, value in original.items()}
    device_gradients = {name: cp.asarray(value) for name, value in gradients.items()}
    first = {name: cp.zeros_like(value) for name, value in parameters.items()}
    second = {name: cp.zeros_like(value) for name, value in parameters.items()}
    pointers = {name: (first[name].data.ptr, second[name].data.ptr) for name in NAMES}
    finite = cp.ones((), dtype=cp.bool_)
    reference_first = {name: np.zeros_like(value) for name, value in original.items()}
    reference_second = {name: np.zeros_like(value) for name, value in original.items()}
    for epoch in (1, 2):
        with np.errstate(over="ignore"):
            norm = np.sqrt(sum(np.sum(value**2) for value in gradients.values()))
        for name, raw in gradients.items():
            gradient = raw / max(norm, 1)
            reference_first[name] = 0.9 * reference_first[name] + 0.1 * gradient
            reference_second[name] = 0.999 * reference_second[name] + 0.001 * gradient**2
            original[name] -= (
                0.03
                * (reference_first[name] / (1 - 0.9**epoch))
                / (np.sqrt(reference_second[name] / (1 - 0.999**epoch)) + 1e-8)
            )
        fused.update(
            parameters,
            device_gradients,
            first,
            second,
            cp.array(0.25),
            finite,
            learning_rate=0.03,
            epoch=epoch,
        )
        assert bool(finite)
        for name in NAMES:
            np.testing.assert_allclose(
                cp.asnumpy(parameters[name]), original[name], rtol=1e-12, atol=1e-13
            )
            assert (first[name].data.ptr, second[name].data.ptr) == pointers[name]


def test_workspace_nondefault_stream_repeated_and_changed_inputs(gpu):
    from benchmarks.hedging import Market, initialize, loss_and_gradient, simulate_paths

    cp, fused = gpu
    market = Market(steps=5)
    workspace = fused.TrainingWorkspace(cp, 129, 5, 4)
    stream = cp.cuda.Stream(non_blocking=True)
    for seed in (5, 9, 5):
        parameters = initialize(4, np.random.default_rng(seed))
        paths = simulate_paths(market, 129, np.random.default_rng(seed + 1))
        expected_loss, expected = loss_and_gradient(parameters, paths, market, regularization=0.002)
        with stream:
            loss, gradients = workspace.compute(
                {name: cp.asarray(value) for name, value in parameters.items()},
                cp.asarray(paths),
                market,
                0.002,
            )
        stream.synchronize()
        assert float(loss) == pytest.approx(expected_loss, rel=1e-11, abs=1e-12)
        for name in NAMES:
            np.testing.assert_allclose(
                cp.asnumpy(gradients[name]), expected[name], rtol=1e-11, atol=1e-12
            )


@pytest.mark.parametrize(
    "case", ["dtype", "shape", "strided", "nan", "regularization", "workspace"]
)
def test_fused_public_wrapper_rejects_invalid_inputs(gpu, case):
    from benchmarks.hedging import Market, initialize

    cp, fused = gpu
    market = Market(steps=4)
    parameters = {
        name: cp.asarray(value) for name, value in initialize(3, np.random.default_rng(1)).items()
    }
    paths, reg, workspace = cp.ones((7, 5)), 0.002, None
    if case == "dtype":
        paths = paths.astype(cp.float32)
    elif case == "shape":
        paths = cp.ones((7, 4))
    elif case == "strided":
        paths = cp.ones((7, 10))[:, ::2]
    elif case == "nan":
        parameters["b1"][0] = cp.nan
    elif case == "regularization":
        reg = -1.0
    else:
        workspace = fused.TrainingWorkspace(cp, 8, 4, 3)
    with pytest.raises((ValueError, TypeError)):
        fused.loss_and_gradient(parameters, paths, market, regularization=reg, workspace=workspace)


@pytest.mark.parametrize("target", ["loss", "gradient"])
def test_optimizer_accumulates_nonfinite_failure(gpu, target):
    from benchmarks.hedging import initialize

    cp, fused = gpu
    parameters = {
        name: cp.asarray(value) for name, value in initialize(3, np.random.default_rng(1)).items()
    }
    gradients = {name: cp.ones_like(value) for name, value in parameters.items()}
    first = {name: cp.zeros_like(value) for name, value in parameters.items()}
    second = {name: cp.zeros_like(value) for name, value in parameters.items()}
    loss = cp.array(cp.inf if target == "loss" else 1.0)
    if target == "gradient":
        gradients["w2"][0] = cp.nan
    finite = cp.ones((), dtype=cp.bool_)
    fused.update(parameters, gradients, first, second, loss, finite, learning_rate=0.03, epoch=1)
    assert not bool(finite)
    gradients = {name: cp.ones_like(value) for name, value in parameters.items()}
    fused.update(
        parameters, gradients, first, second, cp.array(1.0), finite, learning_rate=0.03, epoch=2
    )
    assert not bool(finite)


@pytest.mark.parametrize(
    "count,width,epochs", [(128, 8, 64), (513, 8, 8), (32, 65, 8), (129, 64, 3), (512, 16, 8)]
)
def test_optimized_and_eager_training_agree_across_dispatch_boundaries(gpu, count, width, epochs):
    from aster_hpo import Replication
    from benchmarks import hedging_cuda
    from benchmarks.hedging import Market

    cp, _ = gpu
    market = Market(epochs=epochs, batch_paths=count, evaluation_paths=129)
    config = {"width": width, "learning_rate": 0.03, "regularization": 0.0001}
    rep = Replication.create(31000, config, 0, "search", True)
    eager = hedging_cuda.train_policy_eager(config, rep, market)
    optimized = hedging_cuda.train_policy(config, rep, market)
    for name in NAMES:
        np.testing.assert_allclose(
            cp.asnumpy(optimized[name]), cp.asnumpy(eager[name]), rtol=1e-7, atol=1e-9
        )
    assert hedging_cuda.evaluate(config, rep, market) == pytest.approx(
        hedging_cuda.evaluate(config, rep, market, eager=True), rel=1e-6, abs=1e-8
    )

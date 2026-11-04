import json
from dataclasses import asdict

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")


def pilot(path):
    from benchmarks.common import write_json
    from benchmarks.hedging import Market

    path.mkdir()
    write_json(
        path / "manifest.json",
        {
            "kind": "financial pilot",
            "seed": 10000,
            "market": asdict(Market(epochs=1, batch_paths=4, evaluation_paths=16)),
        },
    )
    write_json(path / "summary.json", {"full_budget": {"audit_mean": 0.01}})


def test_frozen_study_runs_resumes_and_regenerates_from_evidence(tmp_path, monkeypatch):
    import importlib.metadata

    from benchmarks.study import freeze_protocol, regenerate, run_study

    pilot(tmp_path / "pilot")
    protocol = tmp_path / "protocol.json"
    settings = freeze_protocol(
        tmp_path / "pilot",
        protocol,
        seeds=(0, 1),
        count=2,
        workers=1,
        checkpoints=(2, 4),
        audit_replications=2,
        scaling_workers=(),
    )
    assert settings["quality_margin"] == pytest.approx(0.0001)
    output = tmp_path / "study"
    report = run_study(protocol, output)
    assert all(row["runs"] == 2 for row in report["methods"].values())
    assert report["comparisons"]["race"]["n"] == 2
    assert regenerate(output) == report
    assert run_study(protocol, output, resume=True) == report
    original_version = importlib.metadata.version
    with monkeypatch.context() as patch:
        patch.setattr(
            importlib.metadata,
            "version",
            lambda name: "changed" if name == "threadpoolctl" else original_version(name),
        )
        with pytest.raises(ValueError, match="threadpoolctl"):
            run_study(protocol, output, resume=True)
    with pytest.raises(FileExistsError):
        run_study(protocol, output)
    manifest = json.loads((output / "manifest.json").read_text())
    run_id = manifest["campaigns"][0]["id"]
    (output / "campaigns" / run_id / "audit.json").unlink()
    with pytest.raises(FileNotFoundError):
        regenerate(output)


def test_protocol_is_exclusive_and_rejects_pilot_seed_reuse(tmp_path):
    from benchmarks.study import freeze_protocol

    pilot(tmp_path / "pilot")
    with pytest.raises(ValueError, match="pilot"):
        freeze_protocol(tmp_path / "pilot", tmp_path / "bad.json", seeds=(10000, 10001))
    freeze_protocol(tmp_path / "pilot", tmp_path / "protocol.json", seeds=(0, 1))
    with pytest.raises(FileExistsError):
        freeze_protocol(tmp_path / "pilot", tmp_path / "protocol.json", seeds=(0, 1))


def test_altered_protocol_cannot_resume_an_existing_study(tmp_path):
    from benchmarks.study import freeze_protocol, run_study

    pilot(tmp_path / "pilot")
    protocol = tmp_path / "protocol.json"
    freeze_protocol(
        tmp_path / "pilot",
        protocol,
        seeds=(0, 1),
        count=2,
        workers=1,
        checkpoints=(2, 4),
        audit_replications=2,
        scaling_workers=(),
    )
    run_study(protocol, tmp_path / "study")
    data = json.loads(protocol.read_text())
    data["quality_margin"] *= 2
    protocol.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol"):
        run_study(protocol, tmp_path / "study", resume=True)


def test_frozen_protocol_records_backend_and_rejects_ambiguous_gpu_workers(tmp_path):
    from benchmarks.study import freeze_protocol

    pilot(tmp_path / "pilot")
    settings = freeze_protocol(
        tmp_path / "pilot", tmp_path / "numpy.json", seeds=(0, 1), backend="numpy"
    )
    assert settings["backend"] == "numpy"
    with pytest.raises(ValueError, match="backend"):
        freeze_protocol(tmp_path / "pilot", tmp_path / "bad.json", backend="automatic")
    with pytest.raises(ValueError, match="one worker"):
        freeze_protocol(tmp_path / "pilot", tmp_path / "cuda.json", backend="cuda", workers=4)


def test_source_fingerprint_detects_cuda_kernel_changes(tmp_path):
    from benchmarks.protocol import source_fingerprint

    folder = tmp_path / "cuda" / "include"
    folder.mkdir(parents=True)
    kernel = folder / "accounting_kernel.cuh"
    kernel.write_text("// original accounting\n")
    before = source_fingerprint(root=tmp_path)
    kernel.write_text("// changed accounting\n")
    assert source_fingerprint(root=tmp_path) != before
    build = tmp_path / "cuda" / "CMakeLists.txt"
    build.write_text("set(strict_float64 ON)\n")
    before = source_fingerprint(root=tmp_path)
    build.write_text("set(strict_float64 OFF)\n")
    assert source_fingerprint(root=tmp_path) != before


def test_protocol_rejects_pilot_backend_mismatch(tmp_path):
    from benchmarks.common import write_json
    from benchmarks.study import freeze_protocol

    pilot(tmp_path / "pilot")
    manifest = tmp_path / "pilot" / "manifest.json"
    write_json(manifest, {**json.loads(manifest.read_text()), "backend": "cuda"})
    with pytest.raises(ValueError, match="Pilot backend"):
        freeze_protocol(tmp_path / "pilot", tmp_path / "protocol.json", backend="numpy")


def test_source_fingerprint_ignores_generated_cuda_build_sources(tmp_path):
    from benchmarks.protocol import source_fingerprint

    before = source_fingerprint(root=tmp_path)
    generated = tmp_path / "cuda/build/CMakeFiles/CompilerIdCUDA"
    generated.mkdir(parents=True)
    (generated / "CMakeCUDACompilerId.cu").write_text("// generated compiler probe\n")
    assert source_fingerprint(root=tmp_path) == before

"""Exercise the built executable's CPU-only command-line boundaries."""

import pathlib
import subprocess
import sys
import tempfile

binary = pathlib.Path(sys.argv[1]).resolve()


def run(*args):
    return subprocess.run([str(binary), *args], text=True, capture_output=True)


help_result = run("--help")
assert help_result.returncode == 0, help_result.stderr
assert "--block-size" in help_result.stdout
for args in (
    ("--mode", "wrong"),
    ("--block-size", "0"),
    ("--paths", "0"),
    ("--repeats", "0"),
    ("--intervals", "257"),
    ("--warmups", "-1"),
    ("--paths", "18446744073709551615"),
    ("--paths", "1x"),
    ("--unknown",),
    ("--output",),
):
    result = run(*args)
    assert result.returncode != 0, args
    assert "error:" in result.stderr, (args, result.stderr)
with tempfile.TemporaryDirectory() as temporary:
    output = pathlib.Path(temporary) / "existing.json"
    output.write_text("existing evidence\n")
    result = run("--mode", "smoke", "--output", str(output))
    assert result.returncode != 0, result.stderr
    assert "already exists" in result.stderr, result.stderr
    assert output.read_text() == "existing evidence\n"
print("CLI checks passed: help, invalid settings, overflow, and no-overwrite.")

import subprocess
import sys
from pathlib import Path


def test_pipeline_script_exposes_reproducible_run_options():
    script = Path(__file__).parents[1] / "scripts" / "pipeline.py"

    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    assert "--outdir" in process.stdout
    assert "--top-n" in process.stdout
    assert "--fraction" in process.stdout
    assert "--conditions" in process.stdout
    assert "--holdout-fraction" in process.stdout
    assert "--geometry-neighbors" in process.stdout
    assert "--local-profile-mix" in process.stdout
    assert "--group-prior-mix" in process.stdout
    assert "--count-noise-model" in process.stdout
    assert "--novelty-mode" in process.stdout
    assert "--novelty-strength" in process.stdout
    assert "--seed" in process.stdout
    assert "--formats" in process.stdout

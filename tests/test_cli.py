import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run(args, **kw):
    return subprocess.run(
        [sys.executable, "-m", "bardecode"] + args,
        capture_output=True, text=True, **kw,
    )


def test_cli_help():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "bardecode" in r.stdout.lower()


def test_cli_version():
    r = _run(["--version"])
    assert r.returncode == 0


def test_cli_missing_image_exits_2(tmp_path):
    r = _run([str(tmp_path / "nope.png")])
    assert r.returncode == 2
    data = json.loads(r.stdout)
    assert data["results"][0]["error"] == "file not found"


def test_cli_quiet_outputs_only_values(blank_image):
    r = _run(["--quiet", blank_image])
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_cli_json_structure_on_blank(blank_image):
    r = _run([blank_image])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["tool"] == "bardecode"
    assert data["results"][0]["barcodes"] == []
    assert data["results"][0]["error"] is None


@pytest.mark.slow
def test_cli_real_image():
    """使用用户提供的真实 webp 图。"""
    fixture = Path(__file__).parent / "fixtures" / "top-barcodes-in-retail-header.jpg.webp"
    if not fixture.exists():
        pytest.skip("fixture webp not available")
    r = _run([str(fixture)])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    total = sum(len(ir["barcodes"]) for ir in data["results"])
    assert total >= 2, f"expected at least 2 barcodes (multi-barcode goal), got {data}"


def test_cli_mixed_batch_exit_code_and_results(blank_image, tmp_path):
    r = _run([blank_image, str(tmp_path / "nope.png")])
    assert r.returncode == 2
    data = json.loads(r.stdout)
    assert len(data["results"]) == 2
    errors = [ir["error"] for ir in data["results"]]
    assert None in errors               # blank succeeded
    assert "file not found" in errors   # missing failed

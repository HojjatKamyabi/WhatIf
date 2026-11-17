"""The repository itself: docs and config are tracked, private notes are not, and tests stay offline."""

import shutil
import socket
import subprocess
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def ignored(path: str) -> bool:
    """Whether .gitignore hides `path`, whether or not the file is tracked or exists yet."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, capture_output=True
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "scenarios.toml",
        "README.md",
        "docs/headline/rolling-2020-2026.png",
        "docs/headline/summary.json",
        ".github/workflows/tests.yml",
    ],
)
def test_project_docs_and_config_are_tracked(path):
    assert not ignored(path), f"{path} is hidden by .gitignore"


@pytest.mark.parametrize(
    "path",
    [
        ".scratch/handoff-cv-plan-2026-11-04.md",  # personal planning notes
        ".scratch/party-support-twin-v1/spec.md",  # working notes: spec and issues
        ".scratch/party-support-twin-v1/issues/17-continuous-integration-and-tracked-docs.md",
        "CLAUDE.md",  # AI coding-agent setup, local only
        "CONTEXT.md",
        "docs/agents/issue-tracker.md",
        "docs/adapter-contract.md",  # design notes, local only
        "docs/adr/0001-answer-with-outcome-distributions-not-predictions.md",
        "data/dawum/20260101T000000Z.json",  # downloads stay local; their licences differ
        "results/backtest-2024-09-30-seed0-x.json",
        ".venv/pyvenv.cfg",
    ],
)
def test_private_notes_downloads_and_results_stay_untracked(path):
    assert ignored(path), f"{path} would be committed"


def test_the_ci_workflow_installs_the_project_and_runs_pytest_on_every_push():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "push" in workflow
    assert 'pip install -e ".[dev]"' in workflow
    assert "pytest" in workflow


@pytest.mark.parametrize(
    "reach",
    [
        lambda: socket.create_connection(("127.0.0.1", 9), timeout=1),
        lambda: socket.getaddrinfo("example.org", 80),
        lambda: urllib.request.urlopen("http://example.org", timeout=1),
    ],
    ids=["socket", "name-lookup", "urllib"],
)
def test_tests_cannot_reach_the_network(reach):
    with pytest.raises(RuntimeError, match="network"):
        reach()


def test_the_readme_shows_the_ci_status():
    assert "actions/workflows/tests.yml/badge.svg" in (ROOT / "README.md").read_text(encoding="utf-8")


# --- The README -------------------------------------------------------------------------------------

import json  # noqa: E402
import re  # noqa: E402

from whatif.cli import main  # noqa: E402
from whatif.sources import ADAPTERS  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").is_file() else ""
HEADLINE = ROOT / "docs" / "headline"


def test_the_readme_credits_every_data_source_as_its_licence_asks():
    for adapter in ADAPTERS:
        if adapter.licence is not None and adapter.licence.attribution:
            assert adapter.licence.attribution in README, f"{adapter.name}: attribution missing from the README"
    assert "stays ODbL" in README  # any published database derived from DAWUM polls


def test_the_readme_states_its_limits():
    assert "## Out of scope" in README
    assert "influencing real voters" in README
    assert "Calibration Period" in README


def test_the_readme_opens_with_the_tracked_headline_plot():
    image = re.search(r"!\[[^\]]*\]\((?!https?://)([^)]+)\)", README)  # a file in the repo, not the CI badge
    assert image, "the README shows a plot"
    assert README.index(image.group(0)) < README.index("## "), "the plot comes before the first section"
    assert (ROOT / image.group(1)).is_file()
    assert not ignored(image.group(1))


def test_the_readme_headline_numbers_match_the_saved_summary():
    saved = json.loads((HEADLINE / "summary.json").read_text(encoding="utf-8"))
    summary = saved["summary"]
    for name in ("twin", "naive", "statistical"):
        assert f"{summary[name]['crps_mean']:.3f}" in README, f"{name}'s mean CRPS"
    twin = summary["twin"]
    assert f"{twin['lowest_crps']} of the {twin['backtests']}" in README


@pytest.mark.parametrize("command", sorted(set(re.findall(r"^\s*whatif ([a-z-]+)", README, re.MULTILINE))))
def test_every_command_the_readme_shows_exists(command):
    with pytest.raises(SystemExit) as exit_:
        main([command, "--help"])
    assert exit_.value.code == 0

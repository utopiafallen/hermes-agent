"""Behavioral tests for ``hermes update`` branch resolution (``_resolve_update_branch``).

Contract under test: an explicit ``--branch`` always wins; otherwise the target is the
checkout's CURRENT branch, so a fork install living on e.g. ``custom`` updates itself
instead of silently flipping to ``main``. When no current branch is resolvable (not a git
repo, or detached HEAD) it falls back to ``main`` — the historical default.

These run against REAL git checkouts (init/commit/branch/detach), not mocked subprocess, so
they exercise the actual ``git rev-parse --abbrev-ref HEAD`` semantics the resolver depends on.
"""

import subprocess
from types import SimpleNamespace

from hermes_cli import main as hermes_main


GIT = ["git"]


def _git(cwd, *args, check=True):
    return subprocess.run(GIT + list(args), cwd=cwd, capture_output=True, text=True, check=check)


def _init_repo_on_branch(path, branch):
    """A real repo with one commit, checked out on *branch* (``main`` when branch == 'main')."""
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "a.txt").write_text("one\n")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-qm", "c1")
    if branch != "main":
        _git(path, "checkout", "-qb", branch)
    return path


def test_explicit_branch_wins(tmp_path, monkeypatch):
    # The flag must win even when the checkout is on a different branch.
    repo = _init_repo_on_branch(tmp_path / "repo", "custom")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    assert hermes_main._resolve_update_branch(SimpleNamespace(branch="feature")) == "feature"


def test_default_uses_current_checkout_branch(tmp_path, monkeypatch):
    # The fix: an install parked on a non-default branch updates that branch, not main.
    repo = _init_repo_on_branch(tmp_path / "repo", "custom")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    assert hermes_main._resolve_update_branch(SimpleNamespace(branch=None)) == "custom"


def test_default_main_checkout_stays_main(tmp_path, monkeypatch):
    # The common managed install sits on main; the new default must still resolve to main.
    repo = _init_repo_on_branch(tmp_path / "repo", "main")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    assert hermes_main._resolve_update_branch(SimpleNamespace(branch=None)) == "main"


def test_default_falls_back_to_main_when_detached(tmp_path, monkeypatch):
    # Detached HEAD reports "HEAD" (no branch name); fall back to the historical default.
    repo = _init_repo_on_branch(tmp_path / "repo", "custom")
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    assert hermes_main._resolve_update_branch(SimpleNamespace(branch=None)) == "main"


def test_default_falls_back_to_main_when_not_a_repo(tmp_path, monkeypatch):
    # No .git: git rev-parse fails (non-zero), so fall back to the historical default.
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    (not_a_repo / "a.txt").write_text("x\n")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", not_a_repo)
    assert hermes_main._resolve_update_branch(SimpleNamespace(branch=None)) == "main"

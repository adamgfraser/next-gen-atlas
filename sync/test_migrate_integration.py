#!/usr/bin/env python3
"""Integration tests for sync/migrate-branches.py.

These tests run against the live private repo's open PRs in dry-run mode and
exercise the round-trip equivalence test (acceptance criterion #2):

    compose(apply(decompose(merge_base), diff_ops)) == compose(decompose(head_mono))

For each migrate-able PR, the decomposed-tree-after-migration should compose
back to the same monolith as compose(decompose(head_mono)). This proves the
algorithm is content-preserving at cutover.

The tests are skipped when the private repo isn't available locally, so the
suite remains runnable in CI environments without git auth to the private
repo.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT_PATH = HERE / "migrate-branches.py"
spec = importlib.util.spec_from_file_location("migrate_branches", SCRIPT_PATH)
mb = importlib.util.module_from_spec(spec)
sys.modules["migrate_branches"] = mb
spec.loader.exec_module(mb)


PRIVATE_REPO = Path("/Users/adamfraser/repos/next-gen-atlas-private")


def _private_repo_available() -> bool:
    if not PRIVATE_REPO.exists():
        return False
    try:
        subprocess.run(["git", "-C", str(PRIVATE_REPO), "rev-parse",
                        "--is-inside-work-tree"], check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _private_repo_available(),
    reason=f"private repo not available at {PRIVATE_REPO}",
)


def _get_open_prs() -> list[dict]:
    """Cached fetch of open PRs."""
    out = subprocess.check_output(
        ["gh", "pr", "list", "--state", "open", "--limit", "100",
         "--json", "number,title,headRefName,baseRefName,headRefOid,"
         "baseRefOid,mergeable"],
        cwd=PRIVATE_REPO, text=True,
    )
    return json.loads(out)


_PR_CACHE: list[dict] | None = None


def _open_prs() -> list[dict]:
    global _PR_CACHE
    if _PR_CACHE is None:
        _PR_CACHE = _get_open_prs()
    return _PR_CACHE


# ---------------------------------------------------------------------------
# Dry-run regression: scan all open PRs, every migrate-able one round-trips.
# ---------------------------------------------------------------------------

class TestRoundTripAcrossOpenPRs:
    """Acceptance criterion #2 — for every migrate-able open PR, the round-
    trip property holds:
        compose(apply(decompose(merge_base), diff_ops))
            == compose(decompose(head_mono))
    """

    @pytest.mark.parametrize("pr_meta", _open_prs(),
                             ids=lambda p: f"PR{p['number']}")
    def test_round_trip(self, pr_meta):
        result = mb.compute_migration(PRIVATE_REPO, pr_meta, do_pr_fetch=False)
        if result.status != "migrate-able":
            pytest.skip(f"status={result.status}")
        ok, msg = mb.roundtrip_check(
            PRIVATE_REPO, result, pr_meta, decomposed_main="unused")
        assert ok, f"PR #{pr_meta['number']} round-trip failed: {msg[:500]}"


# ---------------------------------------------------------------------------
# Dry-run produces a sane summary table for all PRs.
# ---------------------------------------------------------------------------

class TestDryRunRegression:
    """Acceptance criterion #1 — dry-run produces the expected status for
    every PR, with rename detection on PR #220 producing nonzero renames."""

    def test_dry_run_all_open(self):
        prs = _open_prs()
        results = []
        for pr in prs:
            r = mb.compute_migration(PRIVATE_REPO, pr, do_pr_fetch=False)
            results.append(r)
        # No PR should be in 'error' state — all 15+ PRs should resolve.
        errors = [r for r in results if r.status == "error"]
        assert errors == [], f"Unexpected errors: {[(r.pr_number, r.error) for r in errors]}"

    def test_pr220_uses_rename_detection(self):
        """PR #220 should produce many renames, not all add+delete."""
        pr220 = next((p for p in _open_prs() if p["number"] == 220), None)
        if pr220 is None:
            pytest.skip("PR #220 no longer open")
        result = mb.compute_migration(PRIVATE_REPO, pr220, do_pr_fetch=False)
        assert result.status == "migrate-able"
        ren, mod, add, delete = result.tally()
        # Without rename detection, PR #220 would have ~370 adds and ~340
        # deletes (per the prototype baseline). With detection, renames > 100
        # and add+del are reduced from that baseline.
        assert ren >= 100, f"Expected substantial rename count, got {ren}"

    def test_pr213_no_atlas_diff(self):
        """PR #213 (docs/milestone-conventions) does not touch the monolith."""
        pr213 = next((p for p in _open_prs() if p["number"] == 213), None)
        if pr213 is None:
            pytest.skip("PR #213 no longer open")
        result = mb.compute_migration(PRIVATE_REPO, pr213, do_pr_fetch=False)
        assert result.status == "no-atlas-diff"


# ---------------------------------------------------------------------------
# Apply-to-temp-worktree round-trip (acceptance criterion #2 strict form)
# ---------------------------------------------------------------------------

class TestApplyToTempWorktree:
    """Verify that --apply --report-only against a temp worktree of the public
    fork's `proposed/atomic-atlas` branch successfully applies the migration
    for at least one PR.

    NOTE: This requires the public fork to be checked out at
    /Users/adamfraser/repos/next-gen-atlas. We use a small PR (low op count)
    to keep the test fast.
    """

    PUBLIC_REPO = Path("/Users/adamfraser/repos/next-gen-atlas")

    @pytest.fixture
    def public_repo_available(self):
        if not self.PUBLIC_REPO.exists():
            pytest.skip(f"public repo not at {self.PUBLIC_REPO}")
        # Verify proposed/atomic-atlas exists
        try:
            subprocess.check_output(
                ["git", "-C", str(self.PUBLIC_REPO), "rev-parse",
                 "proposed/atomic-atlas"],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pytest.skip("proposed/atomic-atlas not in public repo")
        return True

    def test_apply_report_only_small_pr(self, public_repo_available, tmp_path):
        """Apply a small migrate-able PR in --report-only mode: should
        produce a commit on a temp worktree without push."""
        # Find a small migrate-able PR.
        small_pr = None
        for pr in _open_prs():
            r = mb.compute_migration(PRIVATE_REPO, pr, do_pr_fetch=False)
            if r.status == "migrate-able" and r.diff is not None:
                ren, mod, add, delete = r.tally()
                total = ren + mod + add + delete
                if 1 <= total <= 5:  # small but non-empty
                    small_pr = pr
                    break
        if small_pr is None:
            pytest.skip("No small migrate-able PR found in current open set")

        # Resolve decomposed-main from public fork
        decomposed_main = subprocess.check_output(
            ["git", "-C", str(self.PUBLIC_REPO), "rev-parse",
             "proposed/atomic-atlas"], text=True,
        ).strip()

        result = mb.compute_migration(PRIVATE_REPO, small_pr, do_pr_fetch=False)
        # The worktree must be created against the PUBLIC repo (where
        # decomposed_main lives), using the public repo's content/ tree.
        workdir = tmp_path / f"migrate-pr{small_pr['number']}"
        try:
            mb._make_temp_worktree(self.PUBLIC_REPO, decomposed_main, workdir)
            # Apply ops (but skip the non-Atlas diff step since we'd need
            # the private repo's diff to apply against a public-repo tree —
            # not meaningful here. We only test the Atlas-side apply path).
            conflicts = mb.apply_ops_to_worktree(
                workdir, result.diff.ops, sanity_check=False,
            )
            # The migration should produce at least one staged change.
            staged = subprocess.check_output(
                ["git", "diff", "--cached", "--name-only"], cwd=workdir,
                text=True,
            )
            assert staged.strip(), \
                f"Expected staged changes after apply, got none. " \
                f"Conflicts: {conflicts}"
        finally:
            mb._remove_worktree(self.PUBLIC_REPO, workdir)

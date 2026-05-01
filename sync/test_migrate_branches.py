#!/usr/bin/env python3
"""Unit tests for sync/migrate-branches.py.

Covers the moving parts that don't require git/network:
  - UUID-aware rename detection
  - Op application order (delete → rename → add → modify)
  - Conflict-report rendering (markdown + JSON shape)
  - Idempotency check (mocked git refs)
  - normalize_for_rename (doc_no shift equivalence)
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Load migrate-branches.py as the `migrate_branches` module.
HERE = Path(__file__).resolve().parent
SCRIPT_PATH = HERE / "migrate-branches.py"
spec = importlib.util.spec_from_file_location("migrate_branches", SCRIPT_PATH)
migrate_branches = importlib.util.module_from_spec(spec)
sys.modules["migrate_branches"] = migrate_branches
spec.loader.exec_module(migrate_branches)

mb = migrate_branches  # short alias


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(uuid: str, doc_no: str, name: str = "Test", body: str = "Body") -> str:
    """Build a minimal but valid document.md text."""
    return (
        f"---\n"
        f"id: {uuid}\n"
        f"docNo: {doc_no}\n"
        f"name: {name}\n"
        f"type: Core\n"
        f"depth: 3\n"
        f"childType: sections_and_primary_docs\n"
        f"---\n"
        f"\n"
        f"#### {doc_no} - {name} [Core]\n"
        f"\n"
        f"{body}\n"
    )


# ---------------------------------------------------------------------------
# normalize_for_rename
# ---------------------------------------------------------------------------

class TestNormalizeForRename:
    def test_strips_doc_no_field_and_heading(self):
        a = make_doc("u1", "A.1.2.3", body="Same body")
        b = make_doc("u1", "A.1.2.4", body="Same body")
        assert a != b
        assert mb.normalize_for_rename(a) == mb.normalize_for_rename(b)

    def test_different_body_does_not_normalize_equal(self):
        a = make_doc("u1", "A.1.2.3", body="Body one")
        b = make_doc("u1", "A.1.2.4", body="Body two")
        assert mb.normalize_for_rename(a) != mb.normalize_for_rename(b)


# ---------------------------------------------------------------------------
# extract_uuid / extract_doc_no
# ---------------------------------------------------------------------------

class TestExtract:
    def test_extract_uuid(self):
        doc = make_doc("the-uuid", "A.1")
        assert mb.extract_uuid(doc) == "the-uuid"

    def test_extract_doc_no(self):
        doc = make_doc("u", "A.5.2.1")
        assert mb.extract_doc_no(doc) == "A.5.2.1"


# ---------------------------------------------------------------------------
# diff_trees with rename detection
# ---------------------------------------------------------------------------

class TestDiffTrees:
    def test_pure_modify(self):
        u = "u-1"
        base = {f"content/A/1/document.md": make_doc(u, "A.1", body="old")}
        head = {f"content/A/1/document.md": make_doc(u, "A.1", body="new")}
        diff = mb.diff_trees(base, head)
        assert len(diff.modifies) == 1
        assert len(diff.renames) == 0
        assert len(diff.adds) == 0
        assert len(diff.deletes) == 0

    def test_pure_add(self):
        base = {}
        head = {"content/A/1/document.md": make_doc("u-1", "A.1")}
        diff = mb.diff_trees(base, head)
        assert len(diff.adds) == 1
        assert len(diff.modifies) == 0

    def test_pure_delete(self):
        base = {"content/A/1/document.md": make_doc("u-1", "A.1")}
        head = {}
        diff = mb.diff_trees(base, head)
        assert len(diff.deletes) == 1
        assert len(diff.modifies) == 0

    def test_uuid_rename_detection(self):
        """Same UUID at different paths with content matching modulo doc_no
        produces a rename op, not add+delete."""
        u = "u-rename"
        base = {f"content/A/1/3/document.md": make_doc(u, "A.1.3")}
        head = {f"content/A/1/4/document.md": make_doc(u, "A.1.4")}
        diff = mb.diff_trees(base, head)
        assert len(diff.renames) == 1
        assert len(diff.adds) == 0
        assert len(diff.deletes) == 0
        rename = diff.renames[0]
        assert rename.path == "content/A/1/3/document.md"
        assert rename.new_path == "content/A/1/4/document.md"
        assert rename.uuid == u

    def test_rename_plus_real_edit(self):
        """Same UUID at different paths AND content edited beyond doc_no:
        emit rename + modify on new path."""
        u = "u-1"
        base = {"content/A/1/3/document.md": make_doc(u, "A.1.3", body="orig")}
        head = {"content/A/1/4/document.md": make_doc(u, "A.1.4", body="edited")}
        diff = mb.diff_trees(base, head)
        # Should be 1 rename + 1 modify (on the new path)
        assert len(diff.renames) == 1
        assert len(diff.modifies) == 1
        modify = diff.modifies[0]
        assert modify.path == "content/A/1/4/document.md"

    def test_index_md_modify(self):
        """_index.md changes are diffed by path (no UUID)."""
        base = {"content/A/_index.md": "old index"}
        head = {"content/A/_index.md": "new index"}
        diff = mb.diff_trees(base, head)
        assert len(diff.modifies) == 1
        assert diff.modifies[0].path == "content/A/_index.md"


# ---------------------------------------------------------------------------
# apply_ops_to_worktree — tested via roundtrip_check style (no git, just FS)
# ---------------------------------------------------------------------------

class TestOpApplication:
    """Tests apply_ops_to_worktree against a real git worktree.

    Uses a fresh in-memory git repo as workdir (init + initial commit) so
    that `git add`, `git mv`, `git rm` all work. Each test creates a fresh
    repo to keep isolation.
    """

    @pytest.fixture
    def repo_dir(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        return repo

    def _commit(self, repo, msg="init"):
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg, "--allow-empty"],
                       cwd=repo, check=True)

    def test_chained_rename(self, repo_dir):
        """Test rename chain: 12→11, 13→12, 14→13."""
        # Set up content/A/X/document.md for X in 12, 13, 14
        for x in [12, 13, 14]:
            p = repo_dir / "content" / "A" / str(x) / "document.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(make_doc(f"u-{x}", f"A.{x}"))
        self._commit(repo_dir)

        ops = []
        for src, dst in [(12, 11), (13, 12), (14, 13)]:
            ops.append(mb.Op(
                kind="rename",
                path=f"content/A/{src}/document.md",
                new_path=f"content/A/{dst}/document.md",
                old_content=make_doc(f"u-{src}", f"A.{src}"),
                new_content=make_doc(f"u-{src}", f"A.{dst}"),
                uuid=f"u-{src}",
            ))
        conflicts = mb.apply_ops_to_worktree(repo_dir, ops, sanity_check=True)
        assert conflicts == []
        for x in [11, 12, 13]:
            p = repo_dir / "content" / "A" / str(x) / "document.md"
            assert p.exists(), f"{p} should exist"
        # 14 should be gone (moved to 13)
        assert not (repo_dir / "content" / "A" / "14" / "document.md").exists()

    def test_delete_then_rename_into_path(self, repo_dir):
        """Test: delete A/5/doc.md (UUID X), then rename A/6 → A/5 (UUID Y).
        Result: A/5 should contain UUID Y, not be deleted afterwards."""
        for x in [5, 6]:
            p = repo_dir / "content" / "A" / str(x) / "document.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(make_doc(f"u-{x}", f"A.{x}"))
        self._commit(repo_dir)

        ops = [
            mb.Op(kind="delete", path="content/A/5/document.md",
                  old_content=make_doc("u-5", "A.5"), uuid="u-5"),
            mb.Op(kind="rename",
                  path="content/A/6/document.md",
                  new_path="content/A/5/document.md",
                  old_content=make_doc("u-6", "A.6"),
                  new_content=make_doc("u-6", "A.5"),
                  uuid="u-6"),
        ]
        conflicts = mb.apply_ops_to_worktree(repo_dir, ops, sanity_check=True)
        assert conflicts == []
        # A/5/document.md should now contain UUID u-6 (the renamed file)
        existing = (repo_dir / "content" / "A" / "5" / "document.md").read_text()
        assert "id: u-6" in existing
        assert not (repo_dir / "content" / "A" / "6" / "document.md").exists()

    def test_modify_conflict_detected(self, repo_dir):
        p = repo_dir / "content" / "A" / "1" / "document.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(make_doc("u1", "A.1", body="actual"))
        self._commit(repo_dir)

        op = mb.Op(
            kind="modify",
            path="content/A/1/document.md",
            old_content=make_doc("u1", "A.1", body="expected-different"),
            new_content=make_doc("u1", "A.1", body="new"),
            uuid="u1",
        )
        conflicts = mb.apply_ops_to_worktree(repo_dir, [op], sanity_check=True)
        assert len(conflicts) == 1
        c = conflicts[0]
        assert c["op"] == "modify"
        assert c["path"] == "content/A/1/document.md"
        assert "expected-different" in c["expected"]
        assert "actual" in c["actual"]
        assert c["diff"]


# ---------------------------------------------------------------------------
# Idempotency check (G5)
# ---------------------------------------------------------------------------

class TestIdempotency:
    @pytest.fixture
    def repo_dir(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        return repo

    def _commit_files(self, repo, files: dict, msg: str):
        for path, content in files.items():
            p = repo / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg, "--allow-empty"],
                       cwd=repo, check=True)

    def test_no_pre_cutover_tag_returns_false(self, repo_dir):
        """A branch with no <branch>-pre-cutover tag is not considered
        already-migrated."""
        self._commit_files(repo_dir, {"Sky Atlas/Sky Atlas.md": "# A.0 ...\n"},
                           "init")
        # Make a fake "origin" reference
        subprocess.run(["git", "branch", "test-branch"], cwd=repo_dir, check=True)
        # Need an `origin/test-branch` ref. Create a remote pointing to self.
        subprocess.run(["git", "remote", "add", "origin", str(repo_dir)],
                       cwd=repo_dir, check=True)
        subprocess.run(["git", "fetch", "origin"], cwd=repo_dir, check=True,
                       capture_output=True)

        assert not mb.is_already_migrated(repo_dir, "test-branch")

    def test_already_migrated_detected(self, repo_dir):
        """When tag exists AND monolith is gone AND content/_index.md is
        present, treat as already-migrated."""
        # Initial commit with monolith
        self._commit_files(repo_dir, {"Sky Atlas/Sky Atlas.md": "# A.0 ...\n"},
                           "init")
        subprocess.run(["git", "branch", "test-branch"], cwd=repo_dir, check=True)

        # Tag the branch as <branch>-pre-cutover
        subprocess.run(["git", "tag", "test-branch-pre-cutover", "test-branch"],
                       cwd=repo_dir, check=True)

        # Switch to branch, replace monolith with content/_index.md
        subprocess.run(["git", "checkout", "-q", "test-branch"], cwd=repo_dir,
                       check=True)
        os.remove(repo_dir / "Sky Atlas" / "Sky Atlas.md")
        (repo_dir / "Sky Atlas").rmdir()
        (repo_dir / "content").mkdir()
        (repo_dir / "content" / "_index.md").write_text("# index\n")
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "migrate"], cwd=repo_dir,
                       check=True)
        # Set up an origin remote pointing at self (so origin/test-branch exists)
        subprocess.run(["git", "remote", "add", "origin", str(repo_dir)],
                       cwd=repo_dir, check=True)
        subprocess.run(["git", "fetch", "origin"], cwd=repo_dir, check=True,
                       capture_output=True)

        assert mb.is_already_migrated(repo_dir, "test-branch")

    def test_tag_present_but_monolith_still_there(self, repo_dir):
        """Tag exists but Sky Atlas.md still in branch — not already-migrated."""
        self._commit_files(repo_dir, {"Sky Atlas/Sky Atlas.md": "# A.0\n"},
                           "init")
        subprocess.run(["git", "branch", "test-branch"], cwd=repo_dir, check=True)
        subprocess.run(["git", "tag", "test-branch-pre-cutover", "test-branch"],
                       cwd=repo_dir, check=True)
        subprocess.run(["git", "remote", "add", "origin", str(repo_dir)],
                       cwd=repo_dir, check=True)
        subprocess.run(["git", "fetch", "origin"], cwd=repo_dir, check=True,
                       capture_output=True)
        assert not mb.is_already_migrated(repo_dir, "test-branch")


# ---------------------------------------------------------------------------
# Conflict-report rendering (G4)
# ---------------------------------------------------------------------------

class TestConflictReport:
    def _result_with_conflict(self) -> mb.MigrationResult:
        diff = mb.DiffSummary(
            ops=[
                mb.Op(kind="modify", path="content/A/1/document.md",
                      old_content="OLD", new_content="NEW"),
            ],
            base_count=10, head_count=10, unchanged_count=8,
        )
        return mb.MigrationResult(
            pr_number=999,
            branch="edit/test-conflict",
            title="Test PR",
            status="conflict",
            merge_base="abc123def456",
            head_oid="ffffeeeedddd",
            diff=diff,
            conflicts=[{
                "op": "modify",
                "path": "content/A/1/document.md",
                "expected": "expected old content\n",
                "actual": "actual current content\n",
                "diff": "--- a/content/A/1/document.md\n+++ b/...\n",
            }],
        )

    def test_markdown_report_has_required_fields(self):
        r = self._result_with_conflict()
        md = mb.render_conflict_report_md(r)
        assert "PR #999" in md
        assert "edit/test-conflict" in md
        assert "Test PR" in md
        assert "abc123def456"[:12] in md
        assert "ffffeeeedddd"[:12] in md
        assert "expected old content" in md
        assert "actual current content" in md
        assert "Conflicts (1)" in md
        assert "## Diff summary" in md

    def test_json_report_shape(self):
        r = self._result_with_conflict()
        j = mb.render_conflict_report_json(r)
        assert j["pr_number"] == 999
        assert j["branch"] == "edit/test-conflict"
        assert j["status"] == "conflict"
        assert j["merge_base"] == "abc123def456"
        assert j["head_oid"] == "ffffeeeedddd"
        assert isinstance(j["conflicts"], list) and len(j["conflicts"]) == 1
        c = j["conflicts"][0]
        assert c["path"] == "content/A/1/document.md"
        assert "expected" in c
        assert "actual" in c
        assert "diff" in c
        assert "diff_summary" in j
        assert j["diff_summary"]["modify"] == 1

    def test_write_reports_creates_both_files(self, tmp_path):
        r = self._result_with_conflict()
        mb.write_reports(r, tmp_path / "reports")
        assert (tmp_path / "reports" / "PR999.json").exists()
        assert (tmp_path / "reports" / "PR999.md").exists()
        # JSON should be valid
        loaded = json.loads((tmp_path / "reports" / "PR999.json").read_text())
        assert loaded["pr_number"] == 999

    def test_truncate_for_report(self):
        long_text = "\n".join(f"line {i}" for i in range(50))
        truncated = mb.truncate_for_report(long_text, max_lines=5)
        assert "line 0" in truncated
        assert "line 4" in truncated
        assert "line 49" not in truncated
        assert "more lines" in truncated


# ---------------------------------------------------------------------------
# fetch_pr_head — tests the `pull/<n>/head` fetch path (G2)
# ---------------------------------------------------------------------------

class TestFetchPRHead:
    """Test that fetch_pr_head invokes the pull/<n>/head path."""

    def test_fetch_pr_head_uses_pull_refspec(self, monkeypatch):
        captured: list = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)

            class CP:
                returncode = 0
                stdout = ""
                stderr = ""
            return CP()

        monkeypatch.setattr(mb.subprocess, "run", fake_run)
        local_ref = mb.fetch_pr_head(Path("/tmp/repo"), 123)
        assert local_ref == "pr-123-head"
        assert any("pull/123/head:pr-123-head" in part
                   for cmd in captured for part in cmd)


# ---------------------------------------------------------------------------
# Apply ops via pure-FS (mirrors the roundtrip_check inner logic) — sanity
# check that op order produces the right tree.
# ---------------------------------------------------------------------------

class TestApplyOpsPureFS:
    """Tests the synthetic-FS application used inside roundtrip_check."""

    def test_delete_then_rename(self, tmp_path):
        # base tree
        base_tree = {
            "content/A/5/document.md": make_doc("u5", "A.5"),
            "content/A/6/document.md": make_doc("u6", "A.6"),
        }
        for path, c in base_tree.items():
            full = tmp_path / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(c)

        ops = [
            mb.Op(kind="delete", path="content/A/5/document.md",
                  old_content=base_tree["content/A/5/document.md"], uuid="u5"),
            mb.Op(kind="rename", path="content/A/6/document.md",
                  new_path="content/A/5/document.md",
                  old_content=base_tree["content/A/5/document.md"],
                  new_content=make_doc("u6", "A.5"), uuid="u6"),
        ]

        # Inline-replicate the order: delete, rename, add, modify
        delete_ops = [o for o in ops if o.kind == "delete"]
        rename_ops = [o for o in ops if o.kind == "rename"]

        for op in delete_ops:
            (tmp_path / op.path).unlink(missing_ok=True)

        staging = tmp_path / "stage"
        staging.mkdir()
        staged = []
        for i, op in enumerate(rename_ops):
            src = tmp_path / op.path
            if not src.exists():
                staged.append((op, None))
                continue
            s = staging / f"r{i}"
            src.rename(s)
            staged.append((op, s))
        for op, s in staged:
            dst = tmp_path / op.new_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if s is None:
                if op.new_content is not None: dst.write_text(op.new_content)
                continue
            if dst.exists(): dst.unlink()
            s.rename(dst)
            if op.new_content is not None: dst.write_text(op.new_content)

        # Result: content/A/5 has UUID u6 (the renamed one)
        final = (tmp_path / "content" / "A" / "5" / "document.md").read_text()
        assert "id: u6" in final
        assert "id: u5" not in final
        # 6 should be gone
        assert not (tmp_path / "content" / "A" / "6" / "document.md").exists()


# ---------------------------------------------------------------------------
# Integration smoke: decompose-twice + diff_trees on synthetic monoliths
# ---------------------------------------------------------------------------

SAMPLE_BASE = """\
# A.0 - Atlas Preamble [Scope]  <!-- UUID: 8650a584-01f8-45d6-882b-c14eab9879c4 -->

This Preamble.

## A.0.1 - Atlas Preamble [Article]  <!-- UUID: 56b15d7d-cdd4-4594-bd95-4f094564ac04 -->

This Article contains definitions.

### A.0.1.1 - Definitions [Section]  <!-- UUID: c7d62f28-1d64-4632-8cd8-4f2b44c51bba -->

This Section contains essential definitions.

#### A.0.1.1.1 - Organizational Alignment [Core]  <!-- UUID: 4f6fda1e-7450-4065-8095-e93cb10b3a2a -->

Organizational alignment is a traditional business concept.
"""

SAMPLE_HEAD_MODIFY = """\
# A.0 - Atlas Preamble [Scope]  <!-- UUID: 8650a584-01f8-45d6-882b-c14eab9879c4 -->

This Preamble.

## A.0.1 - Atlas Preamble [Article]  <!-- UUID: 56b15d7d-cdd4-4594-bd95-4f094564ac04 -->

This Article contains definitions.

### A.0.1.1 - Definitions [Section]  <!-- UUID: c7d62f28-1d64-4632-8cd8-4f2b44c51bba -->

This Section contains essential definitions.

#### A.0.1.1.1 - Organizational Alignment [Core]  <!-- UUID: 4f6fda1e-7450-4065-8095-e93cb10b3a2a -->

EDITED: Organizational alignment is a traditional business concept.
"""


class TestDecomposeTwice:
    def test_simple_modify_diff(self):
        base_tree = mb.decompose_text(SAMPLE_BASE)
        head_tree = mb.decompose_text(SAMPLE_HEAD_MODIFY)
        diff = mb.diff_trees(base_tree, head_tree)
        # Exactly one document modified
        assert len(diff.modifies) == 1
        assert "Organizational Alignment" in diff.modifies[0].new_content or \
               "EDITED" in diff.modifies[0].new_content


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

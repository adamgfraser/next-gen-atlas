#!/usr/bin/env python3
"""Atlas Phase 2 — branch-migration script.

Translates open Atlas PRs from monolith format (`Sky Atlas/Sky Atlas.md`) to
decomposed format (`content/` tree) at cutover. Algorithm: decompose-twice-and-
rewrite.

For each open PR:
  1. Find merge-base between branch and old-main (the still-monolith main).
  2. Decompose the merge-base monolith → `base_tree`.
  3. Decompose the branch HEAD's monolith → `head_tree`.
  4. Diff the trees with UUID-aware rename detection.
  5. In --dry-run, just report. In --apply, rewrite the branch on top of
     decomposed-main, preserving non-Atlas changes via `git diff | git apply`.

Spec: ~/projects/atomic-atlas-phase-2-spec.md
Build spec: ~/projects/atomic-atlas-phase-2-build-spec.md

Usage:
    python sync/migrate-branches.py [--dry-run | --apply [--report-only]]
                                    [--pr N | --all-open]
                                    [--decomposed-main <commit>]
                                    [--yes]
                                    [--report-dir PATH]
                                    [--repo-dir PATH]
                                    [--remote REMOTE]
                                    [--push-remote REMOTE]

Default with no flags: --dry-run --all-open against
~/repos/next-gen-atlas-private. The script lives in the public fork (where
sync/decompose.py and sync/compose.py live) but operates against the private
repo's open PRs by default.

Remote configuration:
  --remote (default `origin`) names the remote that has `pull/<n>/head` virtual
  refs and the canonical `main` branch — i.e., the repo where the PRs were
  opened. Used for fetch + merge-base lookups.

  --push-remote (default = --remote) names the remote where branches live and
  pushes go. Differs from --remote when working with forked PRs: e.g. clone
  sky-ecosystem/next-gen-atlas as `origin` (--remote origin), add your fork as
  `fork` (--push-remote fork). The script fetches `pull/<n>/head` from --remote,
  resolves `main` as `<remote>/main`, fetches branch refs from --push-remote,
  resolves branch trees as `<push-remote>/<branch>`, and pushes the migrated
  commit + pre-cutover tag to --push-remote.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Import decompose + compose from the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from decompose import parse_atlas, plan_output  # noqa: E402
from compose import compose as compose_tree  # noqa: E402

DEFAULT_PRIVATE_REPO = Path("/Users/adamfraser/repos/next-gen-atlas-private")
ATLAS_PATH = "Sky Atlas/Sky Atlas.md"
DEFAULT_REPORT_DIR = Path(".scratch/cutover-reports")

# Atlas heading regex — same shape as decompose.HEADING_RE but applied to a
# document.md body to extract the doc_no. We strip the doc_no out for content
# comparison so a UUID at path X (doc_no A.1.4.3) and the same UUID at path Y
# (doc_no A.1.4.4) are recognized as the same content modulo doc_no.
DOC_NO_LINE_RE = re.compile(r"^docNo:\s*\S+\s*$", re.MULTILINE)
HEADING_LINE_RE = re.compile(
    r"^(#{1,6})\s+(\S+)(\s+-\s+.+?\s+\[[^\]]+\]\s*)$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def sh(cmd: list[str], cwd: Optional[Path] = None, check: bool = True,
       input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a command and return CompletedProcess. Raises on non-zero by default."""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        input=input_text,
        capture_output=True,
        check=check,
    )


def sh_out(cmd: list[str], cwd: Optional[Path] = None) -> str:
    """Run a command and return stdout."""
    return sh(cmd, cwd=cwd).stdout


# ---------------------------------------------------------------------------
# Diff operation model
# ---------------------------------------------------------------------------

@dataclass
class Op:
    """A diff operation against the decomposed tree."""
    kind: str  # 'add' | 'modify' | 'delete' | 'rename'
    path: str
    new_content: Optional[str] = None
    old_content: Optional[str] = None
    new_path: Optional[str] = None  # for rename
    uuid: Optional[str] = None       # rename / debug only


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------

def strip_prefix(path: str, prefix: str) -> str:
    """Strip a prefix from a path (no error if absent)."""
    prefix = prefix.rstrip("/") + "/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def short_path(path: str) -> str:
    """Strip leading 'content/' for friendlier display."""
    if path.startswith("content/"):
        return path[len("content/"):]
    return path


# ---------------------------------------------------------------------------
# Decompose a monolith string → {path: content}
# ---------------------------------------------------------------------------

def decompose_text(text: str) -> dict[str, str]:
    """Decompose a monolith string → dict of relative-path → content.

    Paths are relative to the eventual content-root (no `content/` prefix
    stripped — they are returned as `content/...` so they sit naturally on top
    of `decomposed-main`).
    """
    documents = parse_atlas(text)
    if not documents:
        raise ValueError("No documents parsed from input text")
    files = plan_output(documents, "content")
    # plan_output returns absolute-style paths joined with the output_root; we
    # used "content" as root, so everything is already content-relative.
    return dict(files)


# ---------------------------------------------------------------------------
# UUID extraction from a decomposed document.md
# ---------------------------------------------------------------------------

def extract_uuid(content: str) -> Optional[str]:
    """Return the UUID from a decomposed document.md, or None."""
    for line in content.splitlines():
        if line.startswith("id:"):
            return line.split(":", 1)[1].strip()
        if line == "---":
            # past the frontmatter on the second --- — no id line found
            continue
    return None


def extract_doc_no(content: str) -> Optional[str]:
    """Return the docNo from a decomposed document.md, or None."""
    for line in content.splitlines():
        if line.startswith("docNo:"):
            return line.split(":", 1)[1].strip()
    return None


def normalize_for_rename(content: str) -> str:
    """Normalize document.md content for "same modulo doc_no" comparison.

    Strips the docNo: frontmatter line and rewrites the heading-line doc_no to
    a placeholder, so that two document.md files representing the same UUID
    but at different positions in the doc-no tree compare equal.
    """
    # Drop the docNo frontmatter line
    out = DOC_NO_LINE_RE.sub("docNo: __NORMALIZED__", content)
    # Replace the doc_no token in the heading line, e.g. "## A.1.2.3 - Foo [Type]"
    out = HEADING_LINE_RE.sub(r"\1 __NORMALIZED__\3", out)
    return out


# ---------------------------------------------------------------------------
# Diff with UUID-aware rename detection (G1)
# ---------------------------------------------------------------------------

@dataclass
class DiffSummary:
    ops: list[Op] = field(default_factory=list)
    base_count: int = 0
    head_count: int = 0
    unchanged_count: int = 0

    @property
    def renames(self) -> list[Op]:
        return [o for o in self.ops if o.kind == "rename"]

    @property
    def modifies(self) -> list[Op]:
        return [o for o in self.ops if o.kind == "modify"]

    @property
    def adds(self) -> list[Op]:
        return [o for o in self.ops if o.kind == "add"]

    @property
    def deletes(self) -> list[Op]:
        return [o for o in self.ops if o.kind == "delete"]


def _index_by_uuid(tree: dict[str, str]) -> dict[str, list[str]]:
    """Map UUID → list of document.md paths in the tree."""
    out: dict[str, list[str]] = {}
    for path, content in tree.items():
        if not path.endswith("/document.md") and not path.endswith("\\document.md"):
            continue
        uuid = extract_uuid(content)
        if uuid is None:
            continue
        out.setdefault(uuid, []).append(path)
    return out


def diff_trees(base: dict[str, str], head: dict[str, str]) -> DiffSummary:
    """Compute decomposed diff with UUID-aware rename detection.

    Strategy:
      1. Index both trees by UUID (only `document.md` files have UUIDs).
      2. For each UUID present in both trees:
         - If at the same path: emit `modify` if content differs.
         - If at different paths AND content matches modulo doc_no: emit
           `rename(old → new)`. If new content additionally differs from the
           rename-normalized old, also emit a `modify` on the new path.
         - If at different paths AND content differs structurally: treat as
           `delete(old) + add(new)` (the safe fallback — this is rare; means
           someone moved AND edited at the same time in a way that's not just
           a doc_no shift).
      3. UUIDs in base but not in head → `delete`.
      4. UUIDs in head but not in base → `add`.
      5. Non-document.md files (e.g. `_index.md`) — diff by path only:
         - same path & different content: `modify`
         - in head only: `add`
         - in base only: `delete`

    Note: `_index.md` files are byproducts of `decompose.py`. After cutover the
    decomposed-main will have its own `_index.md` files. We still emit ops for
    them so the resulting tree's _index files are accurate, but we expect most
    `_index.md` diffs to be no-ops once applied to decomposed-main (because
    decomposed-main's _index.md will already match the head_tree's _index.md
    for unchanged subtrees).
    """
    base_uuids = _index_by_uuid(base)
    head_uuids = _index_by_uuid(head)

    ops: list[Op] = []
    consumed_base_paths: set[str] = set()
    consumed_head_paths: set[str] = set()
    unchanged_count = 0

    # Pass 1: docs by UUID
    all_uuids = set(base_uuids) | set(head_uuids)
    for uuid in sorted(all_uuids):
        bp = base_uuids.get(uuid, [])
        hp = head_uuids.get(uuid, [])
        if not bp:
            for path in hp:
                consumed_head_paths.add(path)
                ops.append(Op(kind="add", path=path, new_content=head[path], uuid=uuid))
            continue
        if not hp:
            for path in bp:
                consumed_base_paths.add(path)
                ops.append(Op(kind="delete", path=path, old_content=base[path], uuid=uuid))
            continue
        # Both present. Today every UUID is unique → both are 1-element lists.
        # Defensive: pair them off positionally.
        for old_path, new_path in zip(sorted(bp), sorted(hp)):
            consumed_base_paths.add(old_path)
            consumed_head_paths.add(new_path)
            old_content = base[old_path]
            new_content = head[new_path]
            if old_path == new_path:
                if old_content == new_content:
                    unchanged_count += 1
                else:
                    ops.append(Op(
                        kind="modify",
                        path=old_path,
                        old_content=old_content,
                        new_content=new_content,
                        uuid=uuid,
                    ))
            else:
                # Path changed. Is it just a doc_no shift?
                if normalize_for_rename(old_content) == normalize_for_rename(new_content):
                    ops.append(Op(
                        kind="rename",
                        path=old_path,
                        new_path=new_path,
                        old_content=old_content,
                        new_content=new_content,
                        uuid=uuid,
                    ))
                else:
                    # Real edit + path change. Emit rename + modify on new
                    # path so we still get nice rename rendering plus the
                    # content edit applied.
                    ops.append(Op(
                        kind="rename",
                        path=old_path,
                        new_path=new_path,
                        old_content=old_content,
                        new_content=old_content,  # rename-only target
                        uuid=uuid,
                    ))
                    ops.append(Op(
                        kind="modify",
                        path=new_path,
                        old_content=old_content,
                        new_content=new_content,
                        uuid=uuid,
                    ))
        # Stragglers if list lengths differ (shouldn't happen with unique UUIDs)
        for extra in sorted(bp)[len(hp):]:
            consumed_base_paths.add(extra)
            ops.append(Op(kind="delete", path=extra, old_content=base[extra], uuid=uuid))
        for extra in sorted(hp)[len(bp):]:
            consumed_head_paths.add(extra)
            ops.append(Op(kind="add", path=extra, new_content=head[extra], uuid=uuid))

    # Pass 2: non-document.md files (mostly _index.md). Diff by path.
    base_keys = set(base.keys()) - consumed_base_paths
    head_keys = set(head.keys()) - consumed_head_paths
    in_both = base_keys & head_keys
    only_base = base_keys - head_keys
    only_head = head_keys - base_keys

    for path in sorted(only_head):
        ops.append(Op(kind="add", path=path, new_content=head[path]))
    for path in sorted(only_base):
        ops.append(Op(kind="delete", path=path, old_content=base[path]))
    for path in sorted(in_both):
        if base[path] != head[path]:
            ops.append(Op(
                kind="modify",
                path=path,
                old_content=base[path],
                new_content=head[path],
            ))
        else:
            unchanged_count += 1

    return DiffSummary(
        ops=ops,
        base_count=len(base),
        head_count=len(head),
        unchanged_count=unchanged_count,
    )


# ---------------------------------------------------------------------------
# Git operations (G2 fetch path, G7 stale-cache safety)
# ---------------------------------------------------------------------------

def fetch_remote_prune(repo: Path, remote: str) -> None:
    """`git fetch <remote> --prune` — call at script start (G7).

    Call once per distinct remote in use (i.e., once if --remote == --push-remote,
    twice if they differ).
    """
    sh(["git", "fetch", remote, "--prune"], cwd=repo)


def fetch_pr_head(repo: Path, pr_number: int, remote: str = "origin") -> str:
    """Fetch PR head via `pull/<n>/head:pr-<n>-head` (G2). Returns local ref name.

    `remote` must point at the base repo where the PR was opened — that's where
    `pull/<n>/head` virtual refs live, even for PRs opened from a fork.
    """
    local_ref = f"pr-{pr_number}-head"
    sh(["git", "fetch", remote, f"pull/{pr_number}/head:{local_ref}", "--force"],
       cwd=repo)
    return local_ref


def get_merge_base(repo: Path, ref_a: str, ref_b: str) -> str:
    return sh_out(["git", "merge-base", ref_a, ref_b], cwd=repo).strip()


def get_file_at_ref(repo: Path, ref: str, path: str) -> Optional[str]:
    """Get content of a file at a git ref. Returns None if missing."""
    proc = sh(["git", "show", f"{ref}:{path}"], cwd=repo, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout


def list_open_prs(repo: Path) -> list[dict]:
    """List open PRs via gh CLI."""
    out = sh_out(
        ["gh", "pr", "list", "--state", "open", "--limit", "100",
         "--json", "number,title,headRefName,baseRefName,headRefOid,baseRefOid,mergeable"],
        cwd=repo,
    )
    return json.loads(out)


def get_pr_metadata(repo: Path, pr_number: int) -> dict:
    out = sh_out(
        ["gh", "pr", "view", str(pr_number),
         "--json", "number,title,headRefName,baseRefName,headRefOid,baseRefOid,mergeable"],
        cwd=repo,
    )
    return json.loads(out)


def tag_exists(repo: Path, tag: str) -> bool:
    proc = sh(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
              cwd=repo, check=False)
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Per-PR migration result
# ---------------------------------------------------------------------------

@dataclass
class MigrationResult:
    pr_number: int
    branch: str
    title: str = ""
    status: str = ""        # 'migrate-able' | 'no-atlas-diff' | 'already-migrated' | 'error' | 'conflict'
    merge_base: str = ""
    head_oid: str = ""
    diff: Optional[DiffSummary] = None
    conflicts: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    applied_commit: Optional[str] = None  # set when --apply succeeds

    def tally(self) -> tuple[int, int, int, int]:
        if self.diff is None:
            return (0, 0, 0, 0)
        return (
            len(self.diff.renames),
            len(self.diff.modifies),
            len(self.diff.adds),
            len(self.diff.deletes),
        )


# ---------------------------------------------------------------------------
# Idempotency (G5)
# ---------------------------------------------------------------------------

def is_already_migrated(repo: Path, branch: str, push_remote: str = "origin") -> bool:
    """Check whether the branch looks already-migrated.

    Criteria (all must hold):
      1. Tag `<branch>-pre-cutover` exists.
      2. The branch HEAD does NOT contain `Sky Atlas/Sky Atlas.md`.
      3. The branch HEAD contains the `content/` directory.

    Branch is read at `<push_remote>/<branch>` — that's where the migrated
    branch lives (and where we'd push the migration to).
    """
    pre_cutover_tag = f"{branch}-pre-cutover"
    if not tag_exists(repo, pre_cutover_tag):
        return False

    # Look at the branch's tree at HEAD via <push_remote>/<branch>
    try:
        # Check Sky Atlas.md is gone
        proc = sh(["git", "cat-file", "-e", f"{push_remote}/{branch}:{ATLAS_PATH}"],
                  cwd=repo, check=False)
        if proc.returncode == 0:
            return False  # monolith still present
        # Check content/_index.md exists
        proc = sh(["git", "cat-file", "-e", f"{push_remote}/{branch}:content/_index.md"],
                  cwd=repo, check=False)
        if proc.returncode != 0:
            return False
    except subprocess.CalledProcessError:
        return False
    return True


# ---------------------------------------------------------------------------
# Conflict reporting (G4)
# ---------------------------------------------------------------------------

def truncate_for_report(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = lines[:max_lines]
    return "\n".join(head) + f"\n... ({len(lines) - max_lines} more lines)"


def render_conflict_report_md(result: MigrationResult) -> str:
    """Render a markdown conflict report for one PR."""
    lines: list[str] = []
    lines.append(f"# PR #{result.pr_number} — Cutover Conflict Report")
    lines.append("")
    lines.append(f"**Branch:** `{result.branch}`")
    lines.append(f"**Title:** {result.title}")
    lines.append(f"**Status:** {result.status}")
    lines.append(f"**Merge-base:** `{result.merge_base[:12]}`")
    lines.append(f"**Head:** `{result.head_oid[:12]}`")
    lines.append("")
    if result.diff:
        r, m, a, d = result.tally()
        lines.append("## Diff summary")
        lines.append("")
        lines.append(f"- Rename: {r}")
        lines.append(f"- Modify: {m}")
        lines.append(f"- Add: {a}")
        lines.append(f"- Delete: {d}")
        lines.append("")
    if result.conflicts:
        lines.append(f"## Conflicts ({len(result.conflicts)})")
        lines.append("")
        for c in result.conflicts:
            lines.append(f"### `{c['path']}`")
            lines.append("")
            lines.append("**Expected old content (truncated):**")
            lines.append("")
            lines.append("```")
            lines.append(truncate_for_report(c.get("expected", "")))
            lines.append("```")
            lines.append("")
            lines.append("**Actual content (truncated):**")
            lines.append("")
            lines.append("```")
            lines.append(truncate_for_report(c.get("actual", "")))
            lines.append("```")
            lines.append("")
            if c.get("diff"):
                lines.append("**Unified diff (truncated):**")
                lines.append("")
                lines.append("```diff")
                lines.append(truncate_for_report(c["diff"], max_lines=40))
                lines.append("```")
                lines.append("")
    elif result.error:
        lines.append("## Error")
        lines.append("")
        lines.append(f"```\n{result.error}\n```")
    return "\n".join(lines) + "\n"


def render_conflict_report_json(result: MigrationResult) -> dict:
    """Build a machine-readable conflict report."""
    out: dict = {
        "pr_number": result.pr_number,
        "branch": result.branch,
        "title": result.title,
        "status": result.status,
        "merge_base": result.merge_base,
        "head_oid": result.head_oid,
        "error": result.error,
    }
    if result.diff is not None:
        r, m, a, d = result.tally()
        out["diff_summary"] = {
            "rename": r,
            "modify": m,
            "add": a,
            "delete": d,
            "base_count": result.diff.base_count,
            "head_count": result.diff.head_count,
            "unchanged_count": result.diff.unchanged_count,
        }
    out["conflicts"] = result.conflicts
    if result.applied_commit:
        out["applied_commit"] = result.applied_commit
    return out


def write_reports(result: MigrationResult, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"PR{result.pr_number}.json"
    md_path = report_dir / f"PR{result.pr_number}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(render_conflict_report_json(result), f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_conflict_report_md(result))


# ---------------------------------------------------------------------------
# Apply path (G3, G6)
# ---------------------------------------------------------------------------

def _ensure_clean_workdir(workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.parent.mkdir(parents=True, exist_ok=True)


def _make_temp_worktree(repo: Path, base_ref: str, workdir: Path) -> None:
    """Create a temp worktree at workdir checked out at base_ref."""
    _ensure_clean_workdir(workdir)
    # Use a detached HEAD so we don't conflict with any branch name.
    sh(["git", "worktree", "add", "--detach", str(workdir), base_ref], cwd=repo)


def _remove_worktree(repo: Path, workdir: Path) -> None:
    if workdir.exists():
        sh(["git", "worktree", "remove", "--force", str(workdir)], cwd=repo, check=False)
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)


def apply_ops_to_worktree(
    workdir: Path,
    ops: list[Op],
    *,
    sanity_check: bool = True,
    repo_for_git: Optional[Path] = None,
) -> list[dict]:
    """Apply diff ops to the working tree at workdir. Returns list of conflicts.

    A conflict occurs when sanity_check=True and the existing content of a file
    being modified does not match `op.old_content`.

    Renames are applied in two phases (move-to-staging then move-to-final) so
    that chained renames like `A→B, C→A` don't clobber each other.
    """
    git_cwd = workdir
    conflicts: list[dict] = []

    # Apply ops in the order: delete → rename → add → modify.
    # Reason: deletes free up paths, renames may target a freed path, adds go
    # to fresh paths, modifies happen last on settled paths.
    # Renames are applied two-phase to handle chains/cycles correctly.
    delete_ops = [o for o in ops if o.kind == "delete"]
    rename_ops = [o for o in ops if o.kind == "rename"]
    add_ops = [o for o in ops if o.kind == "add"]
    modify_ops = [o for o in ops if o.kind == "modify"]

    # 1. Apply deletes first.
    for op in delete_ops:
        target = workdir / op.path
        if not target.exists():
            continue
        sh(["git", "rm", "-f", op.path], cwd=git_cwd)

    staging_root = workdir / ".__migrate_rename_staging__"
    staging_root.mkdir(exist_ok=True)
    staged_renames: list[tuple[Op, Optional[Path]]] = []
    # First sanity-check & stage. Sources that fail sanity check are
    # recorded as conflicts and skipped.
    for i, op in enumerate(rename_ops):
        src = workdir / op.path
        if not src.exists():
            # No source — handle as ADD on new path in phase 2.
            staged_renames.append((op, None))
            continue
        if sanity_check and op.old_content is not None:
            existing = src.read_text(encoding="utf-8")
            if existing != op.old_content and \
               normalize_for_rename(existing) != normalize_for_rename(op.old_content):
                conflicts.append({
                    "op": "rename",
                    "path": op.path,
                    "new_path": op.new_path,
                    "expected": op.old_content,
                    "actual": existing,
                    "diff": _unified_diff(op.old_content, existing, op.path),
                })
                staged_renames.append((op, None))
                continue
        # Stage via git mv so the rename is tracked.
        staged = staging_root / f"r{i}"
        # Use git mv to a temp filename outside the source's tree.
        sh(["git", "mv", op.path, str(staged.relative_to(workdir))], cwd=git_cwd)
        staged_renames.append((op, staged))

    # Phase 2: move staged files to their final destinations and overwrite
    # content with op.new_content if provided.
    for op, staged in staged_renames:
        dst = workdir / op.new_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if staged is None:
            # Source was missing or skipped; if we have new_content, just
            # write the file.
            if op.new_content is not None:
                dst.write_text(op.new_content, encoding="utf-8")
                sh(["git", "add", op.new_path], cwd=git_cwd)
            continue
        # If destination already has a file (from a different op), fall back
        # to remove-then-place. With proper UUID-based diff this should be
        # rare; if it happens, prefer the most-recent op's content.
        if dst.exists():
            sh(["git", "rm", "-f", op.new_path], cwd=git_cwd, check=False)
            if dst.exists():
                dst.unlink()
        sh(["git", "mv", str(staged.relative_to(workdir)), op.new_path],
           cwd=git_cwd)
        if op.new_content is not None:
            dst.write_text(op.new_content, encoding="utf-8")
            sh(["git", "add", op.new_path], cwd=git_cwd)

    # Clean up staging dir
    if staging_root.exists():
        try:
            staging_root.rmdir()
        except OSError:
            shutil.rmtree(staging_root, ignore_errors=True)

    # 4. Apply adds.
    for op in add_ops:
        target = workdir / op.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(op.new_content or "", encoding="utf-8")
        sh(["git", "add", op.path], cwd=git_cwd)

    # 5. Apply modifies.
    for op in modify_ops:
        target = workdir / op.path
        if not target.exists():
            conflicts.append({
                "op": "modify",
                "path": op.path,
                "expected": op.old_content or "",
                "actual": "",
                "diff": _unified_diff(op.old_content or "", "", op.path),
            })
            continue
        if sanity_check and op.old_content is not None:
            existing = target.read_text(encoding="utf-8")
            if existing != op.old_content:
                conflicts.append({
                    "op": "modify",
                    "path": op.path,
                    "expected": op.old_content,
                    "actual": existing,
                    "diff": _unified_diff(op.old_content, existing, op.path),
                })
                continue
        target.write_text(op.new_content or "", encoding="utf-8")
        sh(["git", "add", op.path], cwd=git_cwd)

    return conflicts


def _unified_diff(a: str, b: str, path: str, max_lines: int = 60) -> str:
    import difflib
    diff = difflib.unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=2,
    )
    text = "".join(diff)
    return truncate_for_report(text, max_lines=max_lines)


def apply_non_atlas_diff(
    repo: Path,
    workdir: Path,
    merge_base: str,
    head_ref: str,
) -> bool:
    """Apply non-Atlas branch changes via `git diff | git apply` (G6).

    Returns True if a non-empty patch was applied, False if the patch was
    empty (no non-Atlas changes).
    """
    diff_proc = sh(
        ["git", "diff", "--binary", f"{merge_base}..{head_ref}", "--",
         f":!{ATLAS_PATH}"],
        cwd=repo, check=True,
    )
    patch = diff_proc.stdout
    if not patch.strip():
        return False
    # Apply in the workdir
    apply_proc = sh(["git", "apply", "--index", "--whitespace=nowarn"],
                    cwd=workdir, input_text=patch, check=False)
    if apply_proc.returncode != 0:
        # Try without --index (some patches don't apply cleanly to index)
        apply_proc2 = sh(["git", "apply", "--whitespace=nowarn"],
                         cwd=workdir, input_text=patch, check=False)
        if apply_proc2.returncode != 0:
            raise RuntimeError(
                f"git apply failed for non-Atlas diff:\n"
                f"  --index attempt: {apply_proc.stderr}\n"
                f"  fallback attempt: {apply_proc2.stderr}"
            )
        # Stage everything if --index path failed.
        sh(["git", "add", "-A"], cwd=workdir)
    return True


# ---------------------------------------------------------------------------
# Per-PR migration logic
# ---------------------------------------------------------------------------

def compute_migration(
    repo: Path,
    pr_meta: dict,
    *,
    do_pr_fetch: bool = True,
    remote: str = "origin",
    push_remote: str = "origin",
) -> MigrationResult:
    """Compute the diff for one PR. Pure read — no writes.

    `remote` is where `pull/<n>/head` and `main` live (the base repo of the PR).
    `push_remote` is where the branch ref lives (typically the head/fork repo).
    For single-repo workflows they're the same; for forked PRs they differ.
    """
    branch = pr_meta["headRefName"]
    result = MigrationResult(
        pr_number=pr_meta["number"],
        branch=branch,
        title=pr_meta.get("title", ""),
        head_oid=pr_meta.get("headRefOid", ""),
    )

    try:
        # G2: fetch via pull/<n>/head to guarantee the OID is locally available.
        if do_pr_fetch:
            try:
                fetch_pr_head(repo, pr_meta["number"], remote)
            except subprocess.CalledProcessError as e:
                # Fall back to fetching the branch ref directly from push_remote
                # (which is where the head branch lives for forked PRs).
                try:
                    sh(["git", "fetch", push_remote, branch], cwd=repo, check=True)
                except subprocess.CalledProcessError as e2:
                    result.status = "error"
                    result.error = f"fetch failed: {e.stderr.strip()} / {e2.stderr.strip()}"
                    return result

        head_ref = pr_meta["headRefOid"]
        # Find merge-base with <remote>/main
        try:
            merge_base = get_merge_base(repo, f"{remote}/main", head_ref)
        except subprocess.CalledProcessError as e:
            result.status = "error"
            result.error = f"merge-base failure: {e.stderr.strip()}"
            return result
        result.merge_base = merge_base

        base_mono = get_file_at_ref(repo, merge_base, ATLAS_PATH)
        if base_mono is None:
            result.status = "error"
            result.error = "base monolith not found at merge-base"
            return result

        head_mono = get_file_at_ref(repo, head_ref, ATLAS_PATH)
        if head_mono is None:
            # No Sky Atlas.md at head — branch may have deleted it, or doesn't
            # touch the monolith path. Treat as no-atlas-diff.
            result.status = "no-atlas-diff"
            return result

        if base_mono == head_mono:
            result.status = "no-atlas-diff"
            return result

        base_tree = decompose_text(base_mono)
        head_tree = decompose_text(head_mono)
        result.diff = diff_trees(base_tree, head_tree)
        result.status = "migrate-able"
    except Exception as e:
        result.status = "error"
        result.error = f"{type(e).__name__}: {e}"
    return result


def apply_migration(
    repo: Path,
    result: MigrationResult,
    pr_meta: dict,
    *,
    decomposed_main: str,
    report_only: bool = False,
    workdir: Optional[Path] = None,
    push: bool = False,
    push_remote: str = "origin",
) -> MigrationResult:
    """Apply the migration for one PR. Updates result in place.

    If `report_only` is True, applies in a temp worktree and does NOT push.
    Otherwise (when push=True), pushes the migrated branch back to `push_remote`
    after tagging `<branch>-pre-cutover`. `push_remote` is the remote where the
    head branch lives — the fork repo for forked PRs, the canonical repo for
    single-repo workflows.

    `workdir` lets callers supply their own temp directory (used for the
    round-trip test). If omitted, a temp dir is created and removed.
    """
    if result.status != "migrate-able" or result.diff is None:
        return result

    branch = result.branch
    head_ref = pr_meta["headRefOid"]
    merge_base = result.merge_base

    cleanup = workdir is None
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix=f"atlas-migrate-pr{result.pr_number}-"))
        # We want a worktree, not a plain dir. Throw away the tempdir.
        shutil.rmtree(workdir)

    try:
        _make_temp_worktree(repo, decomposed_main, workdir)
        # Apply Atlas ops
        conflicts = apply_ops_to_worktree(workdir, result.diff.ops, sanity_check=True)
        result.conflicts = conflicts

        if conflicts:
            result.status = "conflict"
            return result

        # Apply non-Atlas diff (G6)
        try:
            apply_non_atlas_diff(repo, workdir, merge_base, head_ref)
        except RuntimeError as e:
            result.status = "error"
            result.error = str(e)
            return result

        # Commit
        commit_msg = (
            f"Phase 6 cutover: migrate {branch} to decomposed format\n"
            f"\nOriginal head: {head_ref}\n"
            f"Merge-base: {merge_base}\n"
            f"PR #{result.pr_number}"
        )
        # Configure a committer just for the worktree (may be unset in CI).
        sh(["git", "-c", "user.email=migrate@atlas-axis.io",
            "-c", "user.name=Atlas Migration",
            "commit", "-m", commit_msg, "--allow-empty"],
           cwd=workdir, check=True)
        commit_oid = sh_out(["git", "rev-parse", "HEAD"], cwd=workdir).strip()
        result.applied_commit = commit_oid

        if push and not report_only:
            # Tag <branch>-pre-cutover BEFORE force-pushing the new content.
            pre_cutover_tag = f"{branch}-pre-cutover"
            if not tag_exists(repo, pre_cutover_tag):
                sh(["git", "tag", pre_cutover_tag, head_ref], cwd=repo)
                sh(["git", "push", push_remote, f"refs/tags/{pre_cutover_tag}"], cwd=repo)
            # Force-push the new commit to <branch> on push_remote.
            sh(["git", "push", "--force-with-lease", push_remote,
                f"{commit_oid}:refs/heads/{branch}"], cwd=repo)
    finally:
        if cleanup:
            _remove_worktree(repo, workdir)

    return result


# ---------------------------------------------------------------------------
# Round-trip equivalence test (acceptance criterion #2)
# ---------------------------------------------------------------------------

def roundtrip_check(
    repo: Path,
    result: MigrationResult,
    pr_meta: dict,
    *,
    decomposed_main: str,
) -> tuple[bool, str]:
    """Verify compose(migrated_tree) == apply_pr_diff_to(old_main_monolith).

    For migrate-able PRs only. Skips silently otherwise.

    The test:
      - Apply the PR's Atlas diff onto a monolith of `merge_base`. This is
        what the branch's Sky Atlas.md effectively contains at the head OID.
      - Apply the migration into a fresh worktree of `decomposed_main`.
      - compose() the migrated tree.
      - The two resulting monoliths should both decompose to the same set of
        documents (modulo doc_no/structural details that compose recomputes).

    Note: byte-identity is the strict claim from the spec, but in practice
    decomposed-main and merge-base have different surrounding context. The
    realistic invariant is: after applying the PR diff to merge-base AND
    running decompose then compose, you get the same document set as composing
    the migrated tree restricted to docs touched by the diff.

    For a truly local round-trip (no surrounding-context coupling), we
    construct a SYNTHETIC decomposed-main that equals decompose(merge_base) —
    that is, we treat `merge_base` AS the decomposed-main for round-trip
    purposes. Under that synthetic setup, compose(migrated_tree) should equal
    head_mono byte-exactly.

    Returns (ok, message).
    """
    if result.status != "migrate-able" or result.diff is None:
        return True, "skipped (not migrate-able)"

    head_ref = pr_meta["headRefOid"]
    merge_base = result.merge_base

    base_mono = get_file_at_ref(repo, merge_base, ATLAS_PATH)
    head_mono = get_file_at_ref(repo, head_ref, ATLAS_PATH)
    if base_mono is None or head_mono is None:
        return False, "monolith retrieval failed"

    # Synthetic decomposed-main = decompose(merge_base) materialized to a dir.
    workdir = Path(tempfile.mkdtemp(prefix=f"atlas-rt-pr{result.pr_number}-"))
    try:
        # Materialize the synthetic decomposed-main
        synth_root = workdir / "synthetic-decomposed-main"
        synth_root.mkdir()
        base_tree = decompose_text(base_mono)
        for path, content in base_tree.items():
            full = synth_root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")

        # Apply ops in a separate working dir (no git, no need)
        target_root = workdir / "migrated"
        shutil.copytree(synth_root, target_root)

        # Pure-FS application (no git): order delete → rename (two-phase) →
        # add → modify, mirroring apply_ops_to_worktree.
        delete_ops = [o for o in result.diff.ops if o.kind == "delete"]
        rename_ops = [o for o in result.diff.ops if o.kind == "rename"]
        add_ops = [o for o in result.diff.ops if o.kind == "add"]
        modify_ops = [o for o in result.diff.ops if o.kind == "modify"]

        for op in delete_ops:
            p = target_root / op.path
            if p.exists():
                p.unlink()
            parent = p.parent
            while parent != target_root and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent

        staging_root = target_root / ".__migrate_rename_staging__"
        staging_root.mkdir(exist_ok=True)
        staged_renames: list = []
        for i, op in enumerate(rename_ops):
            src = target_root / op.path
            if not src.exists():
                staged_renames.append((op, None))
                continue
            staged = staging_root / f"r{i}"
            src.rename(staged)
            staged_renames.append((op, staged))
        for op, staged in staged_renames:
            dst = target_root / op.new_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if staged is None:
                if op.new_content is not None:
                    dst.write_text(op.new_content, encoding="utf-8")
                continue
            if dst.exists():
                dst.unlink()
            staged.rename(dst)
            if op.new_content is not None:
                dst.write_text(op.new_content, encoding="utf-8")
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)

        for op in add_ops:
            p = target_root / op.path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(op.new_content or "", encoding="utf-8")

        for op in modify_ops:
            p = target_root / op.path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(op.new_content or "", encoding="utf-8")

        # compose() the migrated tree.
        composed_migrated = compose_tree(str(target_root / "content"))

        # The reference is `compose(decompose(head_mono))`: this absorbs any
        # decompose/compose lossiness that exists on head_mono itself (e.g.
        # the Anchorage doc-no collision in the current Atlas — see the build
        # spec note about PR #235 being a cutover prereq). The round-trip
        # invariant we're checking is:
        #   compose(apply(decompose(merge_base), diff_ops))
        #     == compose(decompose(head_mono))
        # which is equivalent to `migrated_tree == head_tree` (verified on
        # PR-by-PR basis), modulo file-walk ordering inside compose.
        ref_root = workdir / "ref"
        ref_root.mkdir()
        head_tree = decompose_text(head_mono)
        for path, content in head_tree.items():
            full = ref_root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
        composed_ref = compose_tree(str(ref_root / "content"))

        if composed_migrated == composed_ref:
            return True, "byte-identical (against compose(decompose(head_mono)))"
        # Provide diff snippet for debugging
        return False, _unified_diff(composed_ref, composed_migrated,
                                    "Sky Atlas.md", max_lines=40)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def status_line(result: MigrationResult) -> str:
    num = f"#{result.pr_number}"
    if result.status == "error":
        return f"{num:<6} ERROR          ----  ----  ----  ----  {result.branch}  ({result.error})"
    if result.status == "no-atlas-diff":
        return f"{num:<6} no-atlas-diff  ----  ----  ----  ----  {result.branch}"
    if result.status == "already-migrated":
        return f"{num:<6} already-mig.   ----  ----  ----  ----  {result.branch}"
    if result.status == "conflict":
        r, m, a, d = result.tally()
        return (
            f"{num:<6} conflict({len(result.conflicts):<2})   "
            f"{r:>4}  {m:>4}  {a:>4}  {d:>4}  {result.branch}"
        )
    r, m, a, d = result.tally()
    return (
        f"{num:<6} migrate-able   "
        f"{r:>4}  {m:>4}  {a:>4}  {d:>4}  {result.branch}"
    )


def print_summary_table(results: list[MigrationResult]) -> None:
    print(f"{'PR':<6} {'Status':<14} {'Ren':>4}  {'Mod':>4}  {'Add':>4}  {'Del':>4}  Branch")
    print("-" * 100)
    for r in results:
        print(status_line(r))
    print()
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    parts = [f"{n} {s}" for s, n in sorted(counts.items())]
    print("Total: " + ", ".join(parts))
    errs = [r for r in results if r.status == "error"]
    if errs:
        print("\nErrors:")
        for r in errs:
            print(f"  PR #{r.pr_number}: {r.error}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Atlas Phase 2 branch-migration script.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan but don't write.")
    mode.add_argument("--apply", action="store_true",
                      help="Apply the migration. Combine with --report-only "
                           "to apply in a temp worktree without pushing.")
    ap.add_argument("--report-only", action="store_true",
                    help="With --apply: apply in a temp worktree, write report, do not push.")
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--pr", type=int, help="Single PR number.")
    target.add_argument("--all-open", action="store_true",
                        help="All open PRs (default if no target flag given).")
    ap.add_argument("--decomposed-main", default=None,
                    help="The decomposed-main commit to apply onto. "
                         "Defaults to HEAD of the PUBLIC fork's "
                         "proposed/atomic-atlas branch.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip per-PR confirmation prompts in --apply mode.")
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR,
                    help=f"Where to write per-PR reports (default {DEFAULT_REPORT_DIR}).")
    ap.add_argument("--repo-dir", type=Path, default=DEFAULT_PRIVATE_REPO,
                    help=f"Target repo path (default {DEFAULT_PRIVATE_REPO}).")
    ap.add_argument("--public-repo-dir", type=Path,
                    default=Path("/Users/adamfraser/repos/next-gen-atlas"),
                    help="Public-fork repo path (used to resolve --decomposed-main).")
    ap.add_argument("--remote", default="origin",
                    help="Remote with `pull/<n>/head` virtual refs and `main` "
                         "(the PR base repo). Default `origin`.")
    ap.add_argument("--push-remote", default=None,
                    help="Remote where the head branch lives and pushes go. "
                         "Differs from --remote for forked PRs (--remote points at "
                         "the upstream/base repo, --push-remote at the fork). "
                         "Default: same as --remote.")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Skip the initial `git fetch <remote> --prune` (testing only).")
    return ap.parse_args(argv)


def resolve_decomposed_main(public_repo: Path, ref: Optional[str]) -> str:
    """Resolve the decomposed-main commit OID."""
    if ref:
        # Already a SHA or ref. Resolve in public repo.
        return sh_out(["git", "rev-parse", ref], cwd=public_repo).strip()
    # Default: HEAD of proposed/atomic-atlas in the public fork.
    return sh_out(["git", "rev-parse", "proposed/atomic-atlas"],
                  cwd=public_repo).strip()


def select_prs(repo: Path, args: argparse.Namespace) -> list[dict]:
    if args.pr:
        return [get_pr_metadata(repo, args.pr)]
    return list_open_prs(repo)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    repo = args.repo_dir
    remote = args.remote
    push_remote = args.push_remote or args.remote
    apply_mode = args.apply
    dry_run = args.dry_run or (not args.apply)

    if not repo.exists():
        print(f"ERROR: repo path does not exist: {repo}", file=sys.stderr)
        return 2

    if not args.no_fetch:
        try:
            fetch_remote_prune(repo, remote)
            if push_remote != remote:
                fetch_remote_prune(repo, push_remote)
        except subprocess.CalledProcessError as e:
            print(f"WARN: initial git fetch failed: {e.stderr}", file=sys.stderr)

    prs = select_prs(repo, args)
    print(f"=== Atlas Phase 2 migration: scanning {len(prs)} PR(s) "
          f"({'apply' if apply_mode else 'dry-run'}) ===\n")

    decomposed_main: Optional[str] = None
    if apply_mode:
        decomposed_main = resolve_decomposed_main(args.public_repo_dir,
                                                   args.decomposed_main)
        print(f"Decomposed-main: {decomposed_main[:12]}\n")

    results: list[MigrationResult] = []
    for pr_meta in prs:
        # Idempotency check (G5).
        branch = pr_meta["headRefName"]
        if apply_mode and is_already_migrated(repo, branch, push_remote):
            r = MigrationResult(
                pr_number=pr_meta["number"],
                branch=branch,
                title=pr_meta.get("title", ""),
                head_oid=pr_meta.get("headRefOid", ""),
                status="already-migrated",
            )
            results.append(r)
            continue

        result = compute_migration(repo, pr_meta, remote=remote,
                                   push_remote=push_remote)
        results.append(result)

        if apply_mode and result.status == "migrate-able":
            if not args.yes:
                resp = input(f"Apply migration for PR #{result.pr_number} "
                             f"({result.branch})? [y/N] ").strip().lower()
                if resp != "y":
                    print(f"  Skipped PR #{result.pr_number}")
                    continue
            apply_migration(
                repo, result, pr_meta,
                decomposed_main=decomposed_main,
                report_only=args.report_only,
                push=not args.report_only,
                push_remote=push_remote,
            )

        # Always write a report when --apply.
        if apply_mode:
            write_reports(result, args.report_dir)

    print_summary_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

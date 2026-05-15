#!/usr/bin/env python3
"""Compose two refs and produce side-by-side outputs (and optional unified diff).

Use case: verify what changed between two versions of the Atlas. For a
proposal, you can compose `main` and the proposal branch — or compose the
post-vote merge commit and the pre-vote tag — and inspect the diff.

This addresses two stakeholder asks in one tool:
- LDR (JanSky) 2026-05-13: "byte-integrity hash" of the composed monolith,
  so users can verify what's served matches what they voted on.
- Brendan_Navigator (AD) 2026-05-12: a "review map" of the 2,378-file PR —
  what changed where, classified.

The hash answers "did anything change"; the diff answers "what changed."
The diff strictly subsumes the hash (if no diff, hashes match; if there's
a diff, the bytes differ). So this script gives both questions a single
canonical answer.

Each ref is composed using THAT REF'S OWN `sync/compose.py` — not the
caller's. This protects against compose.py drift biasing the comparison.
If compose.py changed between refs, the user-facing rendered output
(which is what they vote on) reflects each ref's compose semantics.

Usage:
    # Side-by-side composed files only:
    python compare-refs.py --ref-a main --ref-b proposal/2026-05-18 \
        --output-dir /tmp/atlas-compare

    # Plus a unified diff to stdout:
    python compare-refs.py --ref-a main --ref-b proposal/2026-05-18 \
        --output-dir /tmp/atlas-compare --diff

    # Plus a unified diff written to a file:
    python compare-refs.py --ref-a main --ref-b proposal/2026-05-18 \
        --output-dir /tmp/atlas-compare --diff --diff-file /tmp/atlas.diff

After running with `--output-dir`, upload both `.md` files to
diffchecker.com or any diff viewer of your choice.

Run from anywhere in the repo (cwd inside the working tree is fine);
script auto-discovers the repo root.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Find the enclosing git repo root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        sys.exit(f"error: not inside a git repo ({result.stderr.strip()})")
    return Path(result.stdout.strip())


def resolve_ref(ref: str, root: Path) -> str:
    """Resolve a ref string (branch / tag / SHA / HEAD~1 etc.) to a full SHA."""
    result = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        sys.exit(f"error: cannot resolve ref '{ref}': {result.stderr.strip()}")
    return result.stdout.strip()


def add_worktree(root: Path, ref: str, dest: Path) -> None:
    """Add a detached worktree at `dest` pointing at `ref`."""
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", str(dest), ref],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        sys.exit(f"error: worktree add failed for '{ref}': {result.stderr.strip()}")


def remove_worktree(root: Path, dest: Path) -> None:
    """Remove a worktree previously created by add_worktree. Best-effort."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(dest)],
        cwd=root, capture_output=True, text=True, check=False,
    )


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def compose_at_worktree(worktree: Path, output_path: Path) -> None:
    """Run `sync/compose.py --input content/ --output <output_path>` in the worktree."""
    compose_script = worktree / "sync" / "compose.py"
    content_dir = worktree / "content"
    if not compose_script.exists():
        sys.exit(f"error: {compose_script} not found — does this ref have decomposed Atlas?")
    if not content_dir.exists():
        sys.exit(f"error: {content_dir} not found — does this ref have decomposed Atlas?")

    result = subprocess.run(
        [sys.executable, str(compose_script),
         "--input", str(content_dir),
         "--output", str(output_path)],
        cwd=worktree, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        sys.exit(f"error: compose failed: {result.stderr.strip() or result.stdout.strip()}")


def sha256_of_file(path: Path) -> str:
    """Return hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose two refs and produce side-by-side outputs (+ optional diff).",
    )
    parser.add_argument("--ref-a", required=True, help="First git ref (branch / tag / SHA)")
    parser.add_argument("--ref-b", required=True, help="Second git ref")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write composed monoliths to (created if missing)")
    parser.add_argument("--diff", action="store_true",
                        help="Also emit a unified diff to stdout (or to --diff-file if given)")
    parser.add_argument("--diff-file",
                        help="Write the unified diff to this path instead of stdout (implies --diff)")
    args = parser.parse_args()

    if args.diff_file:
        args.diff = True

    root = repo_root()
    sha_a = resolve_ref(args.ref_a, root)
    sha_b = resolve_ref(args.ref_b, root)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolved {args.ref_a} → {sha_a[:12]}", file=sys.stderr)
    print(f"Resolved {args.ref_b} → {sha_b[:12]}", file=sys.stderr)

    # Output filenames: short-sha for stable, unambiguous names that survive
    # branch renames + show provenance at a glance.
    out_a = output_dir / f"atlas-{sha_a[:12]}.md"
    out_b = output_dir / f"atlas-{sha_b[:12]}.md"

    # Build side-by-side worktrees, compose each, then tear down.
    with tempfile.TemporaryDirectory(prefix="atlas-compare-") as tmpdir:
        wt_a = Path(tmpdir) / f"wt-{sha_a[:12]}"
        wt_b = Path(tmpdir) / f"wt-{sha_b[:12]}"
        try:
            add_worktree(root, sha_a, wt_a)
            add_worktree(root, sha_b, wt_b)

            print(f"Composing {args.ref_a} ({sha_a[:12]})…", file=sys.stderr)
            compose_at_worktree(wt_a, out_a)

            print(f"Composing {args.ref_b} ({sha_b[:12]})…", file=sys.stderr)
            compose_at_worktree(wt_b, out_b)
        finally:
            remove_worktree(root, wt_a)
            remove_worktree(root, wt_b)

    size_a = out_a.stat().st_size
    size_b = out_b.stat().st_size
    hash_a = sha256_of_file(out_a)
    hash_b = sha256_of_file(out_b)

    print()
    print(f"Wrote:")
    print(f"  {out_a}  ({size_a:,} bytes, SHA-256 {hash_a})")
    print(f"  {out_b}  ({size_b:,} bytes, SHA-256 {hash_b})")
    if hash_a == hash_b:
        print()
        print("Composed monoliths are BYTE-IDENTICAL.")
        # Still emit empty diff if requested, for consistent output shape.

    if args.diff:
        with open(out_a, "r", encoding="utf-8") as f:
            lines_a = f.readlines()
        with open(out_b, "r", encoding="utf-8") as f:
            lines_b = f.readlines()
        diff = difflib.unified_diff(
            lines_a, lines_b,
            fromfile=f"atlas@{sha_a[:12]}",
            tofile=f"atlas@{sha_b[:12]}",
            n=3,
        )
        diff_text = "".join(diff)
        if args.diff_file:
            diff_path = Path(args.diff_file).expanduser().resolve()
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diff_path, "w", encoding="utf-8") as f:
                f.write(diff_text)
            print()
            print(f"Diff written to: {diff_path} ({len(diff_text):,} bytes)")
        else:
            print()
            print("=== UNIFIED DIFF ===")
            sys.stdout.write(diff_text)
            if not diff_text:
                print("(no differences)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

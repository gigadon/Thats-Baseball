"""Rebuild missing slates of record from git history — a one-time repair.

Before the slate lock existed (mlb.etl.slate_record), the only record of a day's
predictions was ``data/predictions/{date}.json``, which every run of the day
rewrote. The last run of the night only predicts games that haven't finished, so
it replaced most of the day's real calls with placeholders — and the morning's
grading then had almost nothing left to grade. 2026-08-02 graded 1 game of 15;
2026-07-22 graded none of 17.

The pre-clobber slate is still recoverable: the repo commits that file after
every run, so the commit made right after the main card went out still holds the
full slate. This module finds that commit, writes the slate it implies, and lets
`track_accuracy` re-grade from it.

Reconstructed slates carry a ``reconstructed`` marker, so they never masquerade
as slates that were genuinely locked live: accuracy records them as
`slate:reconstructed` (below a real lock, above the cache), and a real pipeline
run replaces them.

Usage:
    python -m mlb.etl.slate_repair --start 2026-07-08 --end 2026-08-02 --dry-run
    python -m mlb.etl.slate_repair --start 2026-07-08 --end 2026-08-02 --regrade
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from mlb.etl.slate_record import load_slate, lock_slate
from mlb.models.accuracy import is_stub_prediction

logger = logging.getLogger(__name__)

# The main card's cron (see .github/workflows/daily-predictions.yml). Its data
# commit lands after this; runs are routinely 1-4 hours late, so this is a floor,
# not a window.
MAIN_RUN_UTC = (16, 30)

SELECTION_RULE = "first-commit-at-or-after-16:30Z-among-max-coverage"


@dataclass(frozen=True)
class Candidate:
    """One committed version of a day's prediction file."""

    sha: str
    committed_at: datetime
    blob: dict
    real_predictions: int

    @property
    def num_bets(self) -> int:
        slip = self.blob.get("betting_slip")
        return len(slip.get("bets", [])) if isinstance(slip, dict) else 0

    @property
    def total_predictions(self) -> int:
        return len(self.blob.get("predictions") or [])


def _git(args: list[str], repo: Path, allow_fail: bool = False) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        if allow_fail:
            return ""
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def candidate_commits(
    target_date: date, repo: Path = Path(".")
) -> list[Candidate]:
    """Every committed version of `target_date`'s prediction file, oldest first.

    Read-only: `git log` and `git show`, never a checkout.
    """
    rel = f"data/predictions/{target_date.isoformat()}.json"
    log = _git(["log", "--reverse", "--format=%H %cI", "--", rel], repo)

    candidates: list[Candidate] = []
    for line in log.splitlines():
        sha, _, iso = line.strip().partition(" ")
        if not sha:
            continue
        raw = _git(["show", f"{sha}:{rel}"], repo, allow_fail=True)
        if not raw.strip():
            continue
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Unparsable blob at %s for %s", sha[:8], target_date)
            continue
        preds = blob.get("predictions") or []
        candidates.append(Candidate(
            sha=sha,
            committed_at=datetime.fromisoformat(iso),
            blob=blob,
            real_predictions=sum(1 for p in preds if not is_stub_prediction(p)),
        ))
    return candidates


def choose_main_run_blob(
    target_date: date, candidates: list[Candidate]
) -> tuple[Candidate, str] | None:
    """Pick the version closest to what the main card actually sent.

    Returns `(candidate, confidence)` or None when the day has no real
    predictions in any commit (an off day, or one the pipeline never ran).

    Coverage first, then time. Taking the plain earliest commit would be wrong:
    the late-night wave often commits after midnight ET, so a day's *first*
    commit can belong to the previous evening's run. On 2026-07-27 that commit
    held the full slate but only 7 bets, against the main card's 13. Requiring
    the commit to land at or after the main cron skips the rollover without
    assuming runs are punctual.
    """
    if not candidates:
        return None

    best = max(c.real_predictions for c in candidates)
    if best == 0:
        return None
    top = [c for c in candidates if c.real_predictions == best]

    hour, minute = MAIN_RUN_UTC
    cutoff = datetime(
        target_date.year, target_date.month, target_date.day,
        hour, minute, tzinfo=timezone.utc,
    )
    after_cron = [c for c in top if c.committed_at >= cutoff]
    if after_cron:
        return min(after_cron, key=lambda c: c.committed_at), "high"

    # Nothing committed after the main cron: the best we have predates it.
    return min(top, key=lambda c: c.committed_at), "low"


def reconstruct_slate(
    target_date: date,
    data_dir: Path = Path("data"),
    *,
    repo: Path = Path("."),
    dry_run: bool = False,
) -> dict | None:
    """Rebuild and write `target_date`'s slate. Returns a summary, or None to skip.

    Skips days that already hold a genuinely locked slate — those are the record,
    and a reconstruction must never overwrite one.
    """
    existing = load_slate(target_date, data_dir)
    if existing is not None and not existing.get("reconstructed"):
        logger.info("%s already has a real locked slate — skipping", target_date)
        return None

    chosen = choose_main_run_blob(target_date, candidate_commits(target_date, repo))
    if chosen is None:
        logger.info("%s: no commit holds a real prediction — nothing to rebuild",
                    target_date)
        return None

    candidate, confidence = chosen
    predictions = candidate.blob.get("predictions") or []
    slip = candidate.blob.get("betting_slip")
    summary = {
        "date": target_date.isoformat(),
        "commit": candidate.sha[:8],
        "committed_at": candidate.committed_at.isoformat(),
        "predictions": candidate.total_predictions,
        "real": candidate.real_predictions,
        "bets": candidate.num_bets,
        "confidence": confidence,
    }

    if dry_run:
        return summary

    lock_slate(
        target_date, predictions, slip if isinstance(slip, dict) else None, data_dir,
        reconstructed_from={
            "commit": candidate.sha,
            "committed_at": candidate.committed_at.isoformat(),
            "rule": SELECTION_RULE,
            "confidence": confidence,
            "repaired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return summary


async def repair_range(
    start: date,
    end: date,
    data_dir: Path = Path("data"),
    *,
    repo: Path = Path("."),
    regrade: bool = True,
    settle: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Rebuild every slate in [start, end], optionally re-grading and settling."""
    from mlb.models.accuracy import track_accuracy

    repaired: list[dict] = []
    d = start
    while d <= end:
        summary = reconstruct_slate(d, data_dir, repo=repo, dry_run=dry_run)
        if summary:
            repaired.append(summary)
            if not dry_run and regrade:
                record = await track_accuracy(d, data_dir)
                summary["graded"] = (
                    record["summary"]["total_games"] if record else 0
                )
            if not dry_run and settle:
                summary["settled"] = await _settle_if_missing(d, data_dir)
        d += timedelta(days=1)
    return repaired


async def _settle_if_missing(target_date: date, data_dir: Path) -> int:
    """Settle a day only if it has no settlement yet — never restate a graded card."""
    from mlb.betting.settlement import settle_day

    if (data_dir / "betting" / f"{target_date.isoformat()}.json").exists():
        return 0
    result = await settle_day(target_date, data_dir)
    return result["summary"]["bets_placed"] if result else 0


# ── CLI ──────────────────────────────────────────────────────


def main():
    import argparse
    import asyncio

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Rebuild slates of record from git history"
    )
    parser.add_argument("--start", required=True, help="First date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Last date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--repo", type=str, default=".")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be rebuilt and change nothing",
    )
    parser.add_argument(
        "--no-regrade", action="store_true",
        help="Rebuild slates without re-running accuracy",
    )
    parser.add_argument(
        "--settle", action="store_true",
        help="Also settle repaired days that have no settlement file yet",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    repo = Path(args.repo)

    # A rebuild rewrites files under data/; refuse to mix it with uncommitted work
    # so the diff it produces is reviewable on its own.
    if not args.dry_run:
        dirty = _git(["status", "--porcelain", "data/"], repo).strip()
        if dirty:
            raise SystemExit(
                "data/ has uncommitted changes — commit or stash them first:\n" + dirty
            )

    repaired = asyncio.run(repair_range(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        data_dir,
        repo=repo,
        regrade=not args.no_regrade,
        settle=args.settle,
        dry_run=args.dry_run,
    ))

    if not repaired:
        print("Nothing to rebuild.")
        return

    header = (
        f"{'date':12} {'commit':9} {'committed (UTC)':22} "
        f"{'real/total':>11} {'bets':>5} {'conf':>5}"
    )
    if not args.dry_run:
        header += f" {'graded':>7}"
    print(header)
    for r in repaired:
        line = (
            f"{r['date']:12} {r['commit']:9} {r['committed_at'][:19]:22} "
            f"{r['real']:>5}/{r['predictions']:<5} {r['bets']:>5} {r['confidence']:>5}"
        )
        if "graded" in r:
            line += f" {r['graded']:>7}"
        print(line)
    print(f"\n{len(repaired)} day(s) {'would be ' if args.dry_run else ''}rebuilt.")


if __name__ == "__main__":
    main()

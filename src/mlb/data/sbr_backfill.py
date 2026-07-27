"""Backfill MLB moneyline + totals from SportsBookReview (free, no API key).

This is the repeatable, totals-capable complement to the one-time
``mlb_odds_dataset.json`` import (``convert_odds_json``) and the moneyline-only
``oddsportal_*_raw.csv`` scrapes (``convert_oddsportal``). The Odds API free
tier only captures a handful of days a month and drops totals entirely, so
``odds_history.csv`` (the training source for ``market_home_prob`` /
``market_total``) starves the most recent, highest time-decay-weight games.
Run this to refill any date range straight from SBR's public odds pages.

Mechanism (mirrors ArnavSaraogi/mlb-odds-scraper, no Selenium):
  1. MLB schedule + gameType from statsapi.
  2. SBR odds pages carry a ``__NEXT_DATA__`` JSON blob served to a plain client.
  3. Extract closing (currentLine) moneyline + totals for the preferred book,
     de-vig with the shared proportional transform (== ``devig_home_prob``),
     and merge non-destructively into odds_history.csv.

Merge is protective: existing rows that already carry a total_line (the
original import + live captures) are kept; only moneyline-only gap rows are
upgraded to SBR ML+totals, and genuinely new games are added.

Usage:
    PYTHONPATH=src python -m mlb.data.sbr_backfill --season 2026 --write
    PYTHONPATH=src python -m mlb.data.sbr_backfill --start 2025-08-01 --end 2026-07-27
    PYTHONPATH=src python -m mlb.data.sbr_backfill --start 2025-08-01 --end 2025-08-05  # dry run
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import functools
import json
import logging
import random
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

from mlb.data.convert_odds_json import (
    SKIP_TEAMS,
    _american_to_decimal,
    _map_team,
    _pick_line,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
ODDS_FILE = DATA_DIR / "odds_history.csv"
FIELDNAMES = [
    "game_date", "home_team", "away_team",
    "home_moneyline", "away_moneyline", "total_line", "market_home_prob",
]

SEASON_STARTS = {
    2021: "2021-04-01", 2022: "2022-04-07", 2023: "2023-03-30",
    2024: "2024-03-28", 2025: "2025-03-27", 2026: "2026-03-26",
}

_NEXT_DATA = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL
)
_UA = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
]
_ACCEPT_LANG = ["en-US,en;q=0.9", "en-GB,en;q=0.8", "en;q=0.7"]


@functools.lru_cache(maxsize=128)
def _normalize(name: str) -> str:
    n = (name.lower().replace(".", "").replace("'", "")
         .replace("-", " ").replace("&", "and").strip())
    # SBR builds fullName = city + " " + nickname; the A's are "Athletics
    # Athletics". Collapse consecutive duplicate tokens to match statsapi.
    out: list[str] = []
    for t in n.split():
        if not out or out[-1] != t:
            out.append(t)
    return " ".join(out)


def _schedule(start: str, end: str) -> dict[str, dict]:
    """{date: {(away_norm, home_norm): gameType}} from statsapi."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    smap: dict[str, dict] = {}
    cur = s
    with httpx.Client(timeout=30) as c:
        while cur <= e:
            seg = min(cur.replace(year=cur.year + 1) - timedelta(days=1), e)
            url = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1"
                   f"&startDate={cur:%Y-%m-%d}&endDate={seg:%Y-%m-%d}")
            for di in c.get(url).json().get("dates", []):
                d = di["date"]
                smap.setdefault(d, {})
                for g in di.get("games", []):
                    a = _normalize(g["teams"]["away"]["team"]["name"])
                    h = _normalize(g["teams"]["home"]["team"]["name"])
                    smap[d][(a, h)] = g["gameType"]
            cur = seg + timedelta(days=1)
    return smap


def _odds_url(d: str, otype: str) -> str:
    base = "https://www.sportsbookreview.com/betting-odds/mlb-baseball"
    return f"{base}/?date={d}" if otype == "moneyline" else f"{base}/{otype}/full-game/?date={d}"


async def _fetch(client, url, sem, retries=4, base_delay=1.5) -> str | None:
    for attempt in range(retries):
        async with sem:
            headers = {"User-Agent": random.choice(_UA),
                       "Accept-Language": random.choice(_ACCEPT_LANG)}
            try:
                r = await client.get(url, headers=headers, timeout=20)
                if r.status_code == 200:
                    return r.text
                logger.warning("status %s for %s", r.status_code, url)
            except Exception as e:  # noqa: BLE001 — network is best-effort
                logger.warning("fetch error %s: %s", url, str(e)[:80])
        if attempt < retries - 1:
            await asyncio.sleep(base_delay + random.uniform(0, 2))
    return None


async def _scrape_one(client, d, otype, gtmap, sem, base_delay):
    html = await _fetch(client, _odds_url(d, otype), sem, base_delay=base_delay)
    if not html:
        return d, otype, []
    m = _NEXT_DATA.search(html)
    if not m:
        logger.warning("no __NEXT_DATA__ for %s %s", otype, d)
        return d, otype, []
    try:
        data = json.loads(m.group(1))
        tables = data.get("props", {}).get("pageProps", {}).get("oddsTables", [])
        rows = tables[0].get("oddsTableModel", {}).get("gameRows", []) if tables else []
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("parse error %s %s: %s", otype, d, e)
        return d, otype, []
    out = []
    keys = (["homeOdds", "awayOdds"] if otype == "moneyline"
            else ["overOdds", "underOdds", "total"])
    for game in rows:
        gv = game.get("gameView", {})
        away = _normalize(gv.get("awayTeam", {}).get("fullName", "?"))
        home = _normalize(gv.get("homeTeam", {}).get("fullName", "?"))
        cg = {"gameKey": f"{away}_vs_{home}", "gameView": {}}
        for k in ["startDate", "awayTeam", "awayTeamScore", "homeTeam",
                  "homeTeamScore", "gameStatusText", "venueName"]:
            cg["gameView"][k] = gv.get(k)
        cg["gameView"]["gameType"] = gtmap.get(d, {}).get((away, home), "Unknown")
        views = []
        for o in game.get("oddsViews", []) or []:
            if not o:
                continue
            cl = o.get("currentLine", {}) or {}
            ol = o.get("openingLine", {}) or {}
            views.append({
                "sportsbook": o.get("sportsbook", "Unknown"),
                "openingLine": {k: ol.get(k) for k in keys},
                "currentLine": {k: cl.get(k) for k in keys},
            })
        cg["oddsViews"] = views
        out.append(cg)
    return d, otype, out


def _merge_scraped(results) -> dict:
    """[(date, otype, games)] -> {date: [{gameView, odds:{moneyline, totals}}]}."""
    by_date: dict = {}
    for d, otype, games in results:
        by_date.setdefault(d, {})[otype] = games
    merged: dict = {}
    for d, by_type in by_date.items():
        mg: dict = {}
        for otype, games in by_type.items():
            for g in games:
                key = g.get("gameKey")
                if not key:
                    continue
                mg.setdefault(key, {"gameView": dict(g["gameView"]), "odds": {}})
                mg[key]["odds"][otype] = g["oddsViews"]
        merged[d] = list(mg.values())
    return merged


async def _scrape(start: str, end: str, concurrency: int) -> dict:
    gtmap = _schedule(start, end)
    dates = sorted(gtmap.keys())
    logger.info("scraping %d scheduled dates %s..%s", len(dates), start, end)
    sem = asyncio.Semaphore(concurrency)
    otypes = ["moneyline", "totals"]
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [_scrape_one(client, d, t, gtmap, sem, 1.0)
                 for d in dates for t in otypes]
        results = []
        chunk = concurrency * 2
        for i in range(0, len(tasks), chunk):
            for r in await asyncio.gather(*tasks[i:i + chunk], return_exceptions=True):
                if isinstance(r, tuple) and len(r) == 3:
                    results.append(r)
                else:
                    logger.warning("task failed: %s", r)
            if i + chunk < len(tasks):
                await asyncio.sleep(1.0 + random.uniform(0, 1.0))
    return _merge_scraped(results)


def _rows_from_scrape(data: dict) -> list[dict]:
    """Extract odds_history rows (regular season, real ML) from scraped JSON."""
    rows, skipped = [], 0
    for game_date, games in sorted(data.items()):
        for game in games:
            gv = game["gameView"]
            home = _map_team(gv["homeTeam"]["shortName"])
            away = _map_team(gv["awayTeam"]["shortName"])
            if home in SKIP_TEAMS or away in SKIP_TEAMS or gv.get("gameType") != "R":
                skipped += 1
                continue
            ml = _pick_line(game.get("odds", {}).get("moneyline", []), "moneyline")
            if ml is None or not ml.get("homeOdds") or not ml.get("awayOdds"):
                skipped += 1
                continue
            h_ml, a_ml = ml["homeOdds"], ml["awayOdds"]
            tot = _pick_line(game.get("odds", {}).get("totals", []), "totals")
            total_val = tot.get("total") if tot else None
            h_imp, a_imp = 1.0 / _american_to_decimal(h_ml), 1.0 / _american_to_decimal(a_ml)
            rows.append({
                "game_date": game_date, "home_team": home, "away_team": away,
                "home_moneyline": h_ml, "away_moneyline": a_ml,
                "total_line": total_val if total_val is not None else "",
                "market_home_prob": round(h_imp / (h_imp + a_imp), 4),
            })
    seen: dict = {}
    for r in rows:  # dedup within scrape (preferred book already chosen)
        seen.setdefault((r["game_date"], r["home_team"], r["away_team"]), r)
    logger.info("scrape yielded %d unique rows (%d skipped non-R/missing-ml)", len(seen), skipped)
    return list(seen.values())


def _has_total(v: object) -> bool:
    return str(v).strip().lower() not in ("", "nan", "none")


def merge_into_history(new_rows: list[dict], write: bool) -> None:
    existing: dict = {}
    if ODDS_FILE.exists():
        with open(ODDS_FILE) as f:
            for r in csv.DictReader(f):
                existing[(r["game_date"], r["home_team"], r["away_team"])] = r

    before = sum(_has_total(r["total_line"]) for r in existing.values())
    added = upgraded = protected = 0
    for r in new_rows:
        k = (r["game_date"], r["home_team"], r["away_team"])
        if k not in existing:
            existing[k] = r
            added += 1
        elif _has_total(existing[k]["total_line"]):
            protected += 1  # keep original/live row that already has totals
        else:
            existing[k] = r
            upgraded += 1  # upgrade ML-only gap row to SBR ML+totals
    after = sum(_has_total(r["total_line"]) for r in existing.values())

    logger.info("added %d, upgraded %d, protected %d | total_line %d -> %d (+%d)",
                added, upgraded, protected, before, after, after - before)
    if not write:
        logger.info("dry run — pass --write to save %s", ODDS_FILE)
        return
    with open(ODDS_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for k in sorted(existing):
            w.writerow({fld: existing[k].get(fld, "") for fld in FIELDNAMES})
    logger.info("wrote %s (%d rows)", ODDS_FILE, len(existing))


def backfill(start: str, end: str, write: bool = False, concurrency: int = 5) -> None:
    data = asyncio.run(_scrape(start, end, concurrency))
    merge_into_history(_rows_from_scrape(data), write)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Backfill MLB ML+totals from SportsBookReview")
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD")
    p.add_argument("--season", type=int, help="Season year (sets start/end)")
    p.add_argument("--write", action="store_true", help="Write to odds_history.csv (default: dry run)")
    p.add_argument("--concurrency", type=int, default=5)
    args = p.parse_args()

    if args.season:
        start = SEASON_STARTS.get(args.season, f"{args.season}-03-28")
        end = min(f"{args.season}-11-05", date.today().isoformat())
    elif args.start and args.end:
        start, end = args.start, args.end
    else:
        p.error("provide --season or both --start and --end")

    backfill(start, end, write=args.write, concurrency=args.concurrency)


if __name__ == "__main__":
    main()

import csv
import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

import requests
from bs4 import BeautifulSoup

YAHOO_ADP_BASE = "https://hockey.fantasysports.yahoo.com/hockey/draftanalysis"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"

# Query variants we will try to find in Wayback
QUERY_TRIES = [
    "",  # base path
    "?pos=ALL&sort=DA_AP&tab=SD&count=1000",
    "?pos=ALL&sort=DA_AP&tab=SD&count=600",
    "?pos=ALL&sort=DA_AP&tab=SD&count=500",
    "?pos=ALL&sort=DA_AP&count=1000",  # tab fallback
    "?pos=ALL&count=1000"
]

# Preferred capture days around preseason
PREF_MMDD = ["0910", "0901", "0920", "1001", "0831"]

HDR_SYNONYMS = {
    # normalize to these keys
    "player": {"player", "name"},
    "team_pos": {"team/pos", "team - pos", "tm/pos"},
    "team": {"team", "tm"},
    "pos": {"pos", "position"},
    "preseason_rank": {"preseason", "pre-season", "preseason rank", "o-rank", "orank", "o rank", "xrank"},
    "adp_all_drafts": {"all drafts", "adp", "average pick", "avg pick", "overall adp"},
    "adp_last7": {"last 7 days", "adp (last 7)"},
    "percent_drafted": {"%drafted", "% drafted", "percent drafted"}
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; ADPResearch/2.0)"}


def norm_header(h: str) -> str:
    h = re.sub(r"\s+", " ", (h or "")).strip().lower()
    for key, variants in HDR_SYNONYMS.items():
        if h in variants:
            return key
    return h


def clean_num(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    x = x.replace(",", "").replace("%", "").strip()
    if x in {"", "-"}:
        return None
    try:
        return float(x)
    except:
        return None


def parse_table(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    out = []

    if not table:
        return out

    # Headers
    ths = [norm_header(th.get_text(" ", strip=True)) for th in table.find_all("th")]
    # Some tables repeat column groups; we’ll just map left-to-right
    for tr in table.find_all("tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(tds) < 2:
            continue
        row = dict(zip(ths[:len(tds)], tds))
        # Player & (Team - Pos)
        player_cell = row.get("player") or row.get("name")
        if not player_cell:
            # Sometimes the first column is effectively player even if header missing
            player_cell = tds[0]
        team, pos = None, None
        m = re.search(r"\b([A-Z]{2,3})\s*-\s*([A-Z,/ ]+)\b", player_cell)
        if m:
            team = m.group(1)
            pos = m.group(2).replace(" ", "")
            player = re.sub(r"\s+[A-Z]{2,3}\s*-\s*[A-Z,/ ]+\s*$", "", player_cell).strip()
        else:
            # Try separate columns
            player = player_cell
            if "team" in row:
                team = row["team"]
            if "pos" in row:
                pos = row["pos"]

        rec = {
            "player": player or None,
            "team": (team or (row.get("team") or None)) or None,
            "pos": (pos or (row.get("pos") or None)) or None,
            "preseason_rank": clean_num(row.get("preseason_rank") or row.get("preseason")),
            "adp_all_drafts": clean_num(row.get("adp_all_drafts") or row.get("adp") or row.get("average pick")),
            "adp_last7": clean_num(row.get("adp_last7") or row.get("last 7 days")),
            "percent_drafted": clean_num(row.get("percent_drafted") or row.get("% drafted")),
        }
        # If nothing numeric was found, keep anyway (we may drop later)
        out.append(rec)
    return [r for r in out if r.get("player")]


def parse_embedded_json(html: str) -> List[Dict[str, Any]]:
    """
    Yahoo often embeds a big JSON blob:
      root.App.main = { context: { dispatcher: { stores: { ... }}}}
    We try to extract player rows if present.
    """
    m = re.search(r"root\.App\.main\s*=\s*(\{.*?\})\s*;", html, flags=re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except Exception:
        return []

    # Heuristic walk
    out = []
    try:
        stores = data["context"]["dispatcher"]["stores"]
        # Look for something that smells like Draft Analysis grid
        # Keys vary; scan for lists of players with ADP fields
        for store_name, store in stores.items():
            if not isinstance(store, dict):
                continue
            # Common patterns: a list named 'players' or 'rows'
            candidates = []
            for k in ("players", "rows", "data", "items"):
                v = store.get(k)
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    candidates.append(v)
            for cand in candidates:
                for r in cand:
                    # Try to map common keys
                    player = r.get("name") or r.get("player") or r.get("player_full_name")
                    team = r.get("team_abbr") or r.get("team") or r.get("tm")
                    pos = r.get("position") or r.get("pos")
                    adp = r.get("adp") or r.get("avg_pick") or r.get("average_pick")
                    last7 = r.get("adp_last7") or r.get("last7") or r.get("last_7")
                    preseason = r.get("preseason_rank") or r.get("o_rank") or r.get("xrank") or r.get("rank")
                    pct = r.get("percent_drafted") or r.get("pct_drafted") or r.get("drafted_pct")

                    if player:
                        out.append({
                            "player": player,
                            "team": team,
                            "pos": pos,
                            "preseason_rank": clean_num(str(preseason) if preseason is not None else None),
                            "adp_all_drafts": clean_num(str(adp) if adp is not None else None),
                            "adp_last7": clean_num(str(last7) if last7 is not None else None),
                            "percent_drafted": clean_num(str(pct) if pct is not None else None),
                        })
        return [r for r in out if r.get("player")]
    except Exception:
        return []


@dataclass
class Snapshot:
    url: str
    ts: str
    length: int


def cdx_query(original_url: str, year: int) -> List[Snapshot]:
    params = {
        "url": original_url,
        "from": str(year),
        "to": str(year),
        "output": "json",
        "filter": "statuscode:200",
        "matchType": "exact"
    }
    r = requests.get(WAYBACK_CDX, params=params, headers=UA, timeout=30)
    if r.status_code != 200:
        return []
    js = r.json()
    if not js or len(js) <= 1:
        return []
    out = []
    for row in js[1:]:
        # ["urlkey","timestamp","original","mimetype","statuscode","digest","length"]
        ts, orig, length = row[1], row[2], int(row[6]) if row[6].isdigit() else 0
        out.append(Snapshot(url=f"https://web.archive.org/web/{ts}/{orig}", ts=ts, length=length))
    return out


def best_snapshot_for_year(year: int) -> Optional[str]:
    # Try multiple query variants and pick the "best" (by preferred date then by largest length)
    candidates: List[Snapshot] = []
    for q in QUERY_TRIES:
        url = YAHOO_ADP_BASE + q
        snaps = cdx_query(url, year)
        candidates.extend(snaps)

    if not candidates:
        return None

    # Prefer Sept/Aug/Oct captures, then by length
    def rank_key(s: Snapshot):
        mmdd = s.ts[4:8]
        pref_rank = PREF_MMDD.index(mmdd) if mmdd in PREF_MMDD else 999
        # Larger length tends to mean more content kept
        return (pref_rank, -s.length, s.ts)

    candidates.sort(key=rank_key)
    return candidates[0].url


def fetch(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200 and r.text:
            return r.text
    except requests.RequestException:
        pass
    return None


def scrape_one_html(html: str) -> List[Dict[str, Any]]:
    rows = parse_table(html)
    if len(rows) < 10:
        # Maybe all content is in embedded JSON
        rows = parse_embedded_json(html)
    # If we still have weak rows missing numbers, keep them (we log later)
    return rows


def normalize_season(y: int) -> str:
    return f"{y}-{(y+1)%100:02d}"


def main():
    out_path = Path("yahoo_adp_2009_2025.csv")
    out = out_path.open("w", newline="", encoding="utf-8")
    w = csv.writer(out)
    w.writerow(["season","player","team","pos","preseason_rank","adp_all_drafts","adp_last7","percent_drafted","source","snapshot_url"])

    # Historical seasons 2009–2024 (covering 2009-10 … 2024-25 preseasons)
    for year in range(2009, 2025):
        season = normalize_season(year)
        snap = best_snapshot_for_year(year)
        if not snap:
            print(f"[warn] {season}: no snapshot")
            continue
        html = fetch(snap)
        if not html:
            print(f"[warn] {season}: snapshot fetch failed")
            continue
        rows = scrape_one_html(html)
        # Heuristic drop if only the first 50 and all numerics missing
        n_numeric = sum(1 for r in rows if any([r.get("adp_all_drafts"), r.get("adp_last7"), r.get("preseason_rank")]))
        if len(rows) <= 55 and n_numeric == 0:
            print(f"[warn] {season}: only ~{len(rows)} rows and no numeric cols parsed — likely a light capture")
        else:
            print(f"[ok]   {season}: {len(rows)} rows, numeric rows={n_numeric}")

        for r in rows:
            w.writerow([
                season, r.get("player"), r.get("team"), r.get("pos"),
                r.get("preseason_rank"), r.get("adp_all_drafts"),
                r.get("adp_last7"), r.get("percent_drafted"),
                "yahoo_wayback", snap
            ])
        time.sleep(1.0)

    # Live season (2025-26): we’ll just grab the public page (single page, big count)
    # Note: Yahoo often ignores pagination params; we still try with a high 'count'.
    live_params = "?pos=ALL&sort=DA_AP&tab=SD&count=1000"
    live_url = YAHOO_ADP_BASE + live_params
    html = fetch(live_url) or fetch(YAHOO_ADP_BASE)
    if html:
        rows = scrape_one_html(html)
        season = normalize_season(2025)
        print(f"[ok]   {season} live: {len(rows)} rows")
        for r in rows:
            w.writerow([
                season, r.get("player"), r.get("team"), r.get("pos"),
                r.get("preseason_rank"), r.get("adp_all_drafts"),
                r.get("adp_last7"), r.get("percent_drafted"),
                "yahoo_live", live_url
            ])
    else:
        print("[warn] live fetch failed")

    out.close()
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    main()

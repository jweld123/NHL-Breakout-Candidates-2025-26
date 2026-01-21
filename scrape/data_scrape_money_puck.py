# scrape_moneypuck_all_players.py
import io, os, time, requests, pandas as pd
from datetime import datetime

OUT_DIR = "../moneypuck_players"      # where the combined per-season files go
START_END_YEAR = 2008              # 2008-09 => 2008
END_END_YEAR   = datetime.now().year + (1 if datetime.now().month >= 9 else 0)
SEASON_TYPE = "regular"            # or "playoffs"

TEAMS = [
    "ANA","ARI","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM","FLA","LAK",
    "MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SJS","STL","TBL","TOR","VAN",
    "VGK","WSH","WPG","SEA"
]

def get_csv_text(url, tries=3, sleep=0.7):
    last = None
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise last

os.makedirs(OUT_DIR, exist_ok=True)

for year in range(START_END_YEAR, END_END_YEAR + 1):
    season_url_base = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/{SEASON_TYPE}/teams/skaters/"
    frames = []
    for team in TEAMS:
        url = season_url_base + f"{team}.csv"
        try:
            csv_text = get_csv_text(url)
            df = pd.read_csv(io.StringIO(csv_text))
            df["moneypuckTeam"] = team
            frames.append(df)
        except Exception:
            # Team may not exist this season (e.g., SEA/VGK in early years) — skip
            pass

    if not frames:
        continue

    # Combine, drop exact dup rows if any, and save ONE file per season
    season_df = pd.concat(frames, ignore_index=True).drop_duplicates()
    out_path = os.path.join(OUT_DIR, f"moneypuck_players_{year}.csv")
    season_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

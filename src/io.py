import pandas as pd
import numpy as np
from typing import List, Optional
import re


def to_int(x) -> int | float:
    """Safe int cast; supports 'YYYY-YY' like '2008-09' by returning the first year."""
    try:
        # handle '2008-09' / '2019-20'
        s = str(x).strip()
        m = re.match(r"^(\d{4})\s*[-/]\s*\d{2,4}$", s)
        if m:
            return int(m.group(1))
        return int(float(x))
    except Exception:
        return np.nan

def parse_toi_to_minutes(x) -> float:
    """
    Convert TOI strings to minutes.
    Supports 'mm:ss', 'hh:mm:ss', numeric strings, and returns np.nan on bad input.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 3:  # hh:mm:ss
            h, m, sec = map(float, parts)
            return h * 60.0 + m + sec / 60.0
        if len(parts) == 2:  # mm:ss (common for ATOI / total mm:ss)
            m, sec = map(float, parts)
            return m + sec / 60.0
        # fallback: plain number in minutes
        return float(s)
    except Exception:
        return np.nan

def is_percent_col(col: str) -> bool:
    """
    Heuristic: treat %/rates as rate-like columns (weighted-avg in aggregation).
    """
    c = col.lower()
    tokens = ("pct", "percentage", "per60", "/60", " rate", "rate_", "share", "%")
    return any(tok in c for tok in tokens)

def pick_weight_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return first weight column that exists in df.
    If none of the provided candidates are found, try sensible defaults.
    """
    for c in candidates:
        if c in df.columns:
            return c
    for c in ["all_icetime", "TOI", "toi", "gp", "GP"]:
        if c in df.columns:
            return c
    return None

def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    """
    Robust weighted mean: aligns indices, coerces to float, ignores NaN/non-positive weights.
    """
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w.reindex(x.index), errors="coerce").fillna(0.0)
    mask = (~x.isna()) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def load_hr() -> pd.DataFrame:
    """
    Load Hockey-Reference skaters, keep ALL numeric features (incl. advanced),
    and aggregate to one row per (player, season) with:
      - sums for counting stats
      - weighted means for rates/percentages (weight by TOI if available, else GP)
      - role & per-game features
    """
    # load the data and rename the core columns
    inp = "../data/cleaned_data/hockey_reference2008-09to2024-25_names_ascii.csv"
    inp = "/Users/jacobweldon/PycharmProjects/PNHLbreakout/data/cleaned_data/hockey_reference2008-09to2024-25_names_ascii.csv"
    df = pd.read_csv(inp, low_memory=False)
    core_columns = [
        "Player", "Age", "Team", "Pos", "Season", "GP", "G", "A", "PTS",
        "PP", "HIT", "BLK", "TAKE", "GIVE", "PIM", "+/-", "SOG", "SPCT",
        "TOI", "ATOI", "PTS/GP"
    ]
    rename_columns = [
        "player", "age", "team", "pos", "season", "gp", "g", "a", "pts",
        "pp", "hit", "blk", "take", "give", "pim", "plusminus", "sog", "shpct",
        "toi", "atoi", "pts_per_gp"
    ]
    core_map = {orig: new for orig, new in zip(core_columns, rename_columns) if orig in df.columns}
    df = df.rename(columns=core_map)
    # make sure season is an int
    if "season" in df.columns:
        df["season"] = df["season"].apply(to_int)
    # make sure toi and atoi are in minutes
    if "toi" in df.columns:
        df["toi"] = df["toi"].apply(parse_toi_to_minutes)
    if "atoi" in df.columns:
        df["atoi"] = df["atoi"].apply(parse_toi_to_minutes)
    # cast to numeric if the column is not an identification column
    id_cols = {"player", "team", "pos", "season"}
    for c in df.columns:
        if c not in id_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # explicitly split rate and total columns
    all_numeric = [c for c in df.columns if c not in id_cols and df[c].dtype.kind in "if"]
    explicit_rates = {"pts_per_gp", "g_per_gp", "a_per_gp", "pp_per_gp",
                      "hit_per_gp", "blk_per_gp", "take_per_gp", "give_per_gp", "sog_per_gp", "atoi"}
    rate_cols = [c for c in all_numeric if (is_percent_col(c) or c in explicit_rates)]
    sum_cols = [c for c in all_numeric if c not in rate_cols]
    # ensure core counters are in sum_cols (if present)
    for col in ["gp", "g", "a", "pts", "pp", "hit", "blk", "take", "give", "pim", "sog", "plusminus", "toi"]:
        if col in df.columns and col not in sum_cols:
            sum_cols.append(col)
    # ensure one row per player per season
    keys = ["player", "season"]
    w_col = pick_weight_col(df, ["TOI", "toi", "gp", "GP"])
    # sums
    agg_dict = {c: "sum" for c in sum_cols}
    if "team" in df.columns: agg_dict["team"] = "first"
    if "pos" in df.columns: agg_dict["pos"] = "first"
    df_sum = df.groupby(keys, as_index=False).agg(agg_dict)
    # weighted means for rates
    if rate_cols:
        d = df.copy()

        def col_weight_name(col: str) -> str | None:
            # per-game fields → weight by GP if available; else fall back to w_col
            if col in explicit_rates and "gp" in d.columns:
                return "gp"
            return w_col if (w_col and w_col in d.columns) else None

        num_den_cols = []  # keep track to select for groupby sum
        for c in rate_cols:
            x = pd.to_numeric(d[c], errors="coerce")
            wname = col_weight_name(c)
            if wname is not None and wname in d.columns:
                w = pd.to_numeric(d[wname], errors="coerce").clip(lower=0).fillna(0.0)
            else:
                # unweighted mean: weight = 1 for non-NaN entries
                w = (~x.isna()).astype(float)

            num_col = f"__num__{c}"
            den_col = f"__den__{c}"
            d[num_col] = x * w
            d[den_col] = w
            num_den_cols.extend([num_col, den_col])

        # sum numerators/denominators per (player, season)
        grouped = d.groupby(keys)[num_den_cols].sum()

        # compute weighted means safely
        rate_frames = []
        for c in rate_cols:
            num = grouped[f"__num__{c}"]
            den = grouped[f"__den__{c}"].replace(0, np.nan)
            rate_frames.append((num / den).rename(c))
        df_rate = pd.concat(rate_frames, axis=1).reset_index()

        df_agg = df_sum.merge(df_rate, on=keys, how="left")
    else:
        df_agg = df_sum
    # set roles and per game
    if "pos" in df_agg.columns:
        def infer_role(p):
            if pd.isna(p):
                return "forward"
            toks = re.findall(r"[A-Z]+", str(p).upper())  # e.g. "C/LW" -> ["C","LW"], "LD" -> ["LD"]
            if "G" in toks:
                return "goalie"
            if any(t in {"D", "LD", "RD", "DEF"} for t in toks):
                return "defense"
            return "forward"

        df_agg["role"] = df_agg["pos"].apply(infer_role)
    else:
        df_agg["role"] = "forward"  # safe fallback

    # Per-game: skaters only
    if "gp" in df_agg.columns:
        gp_safe = df_agg["gp"].replace({0: np.nan})
        sk_mask = df_agg["role"] != "goalie"
        for stat in ["pts", "g", "a", "pp", "hit", "blk", "take", "give", "sog"]:
            if stat in df_agg.columns:
                df_agg.loc[sk_mask, f"{stat}_per_gp"] = df_agg.loc[sk_mask, stat] / gp_safe[sk_mask]
    # final tidy
    return df_agg.dropna(subset=["season"])

def load_mp_skaters() -> pd.DataFrame:
    """
    Load Moneypuck skaters, keep ALL numeric features, and aggregate to (player, season):
      - sums for counting stats
      - weighted means for rate/percentage columns
        * default weight: all_icetime if present, else gp/all_games_played
      - role inference (goalie/defense/forward)
    """
    inp = "../data/cleaned_data/moneypuck_players_2008-09to2024-25.csv"
    df = pd.read_csv(inp, low_memory=False)
    # rename identity columns
    id_map = {}
    if "name" in df.columns:     id_map["name"] = "player"
    if "position" in df.columns: id_map["position"] = "pos"
    if "season" in df.columns:   id_map["season"] = "season"
    if "team" in df.columns:     id_map["team"] = "team"
    df = df.rename(columns=id_map)

    # make sure season is an int
    if "season" in df.columns:
        df["season"] = df["season"].apply(to_int)
    # gp alias
    if "gp" not in df.columns and "all_games_played" in df.columns:
        df["gp"] = pd.to_numeric(df["all_games_played"], errors="coerce")

    # cast numerics for non-ID columns
    id_cols = {"player", "team", "pos", "season"}
    for c in df.columns:
        if c not in id_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # split numeric into rate vs sum
    all_numeric = [c for c in df.columns if c not in id_cols and df[c].dtype.kind in "if"]
    rate_cols = [c for c in all_numeric if is_percent_col(c)]
    sum_cols = [c for c in all_numeric if c not in rate_cols]

    # make sure typical counters land in sums if present
    for col in ["gp", "all_icetime", "all_goals", "all_assists", "all_points",
                "all_shotsOnGoal", "all_blocked_shot_attempts",
                "all_unblocked_shot_attempts"]:
        if col in df.columns and col not in sum_cols:
            sum_cols.append(col)

    # group keys + default weight
    keys = ["player", "season"]
    default_w = pick_weight_col(df, ["all_icetime", "gp", "all_games_played"])
    # sums to player, season
    agg_dict = {c: "sum" for c in sum_cols}
    if "team" in df.columns: agg_dict["team"] = "first"
    if "pos" in df.columns: agg_dict["pos"] = "first"
    df_sum = df.groupby(keys, as_index=False).agg(agg_dict)
    # vectorized weighted means for rate columns
    if rate_cols:
        d = df.copy()

        def col_weight(col: str) -> str | None:
            # most MP skater rates are per60/% → weight by icetime if available
            return default_w if (default_w and default_w in d.columns) else None

        num_den_cols = []
        for c in rate_cols:
            x = pd.to_numeric(d[c], errors="coerce")
            wname = col_weight(c)
            if wname:
                w = pd.to_numeric(d[wname], errors="coerce").clip(lower=0).fillna(0.0)
            else:
                w = (~x.isna()).astype(float)  # unweighted mean fallback
            num_col = f"__num__{c}"
            den_col = f"__den__{c}"
            d[num_col], d[den_col] = x * w, w
            num_den_cols.extend([num_col, den_col])

        grouped = d.groupby(keys)[num_den_cols].sum()
        rate_frames = []
        for c in rate_cols:
            num = grouped[f"__num__{c}"]
            den = grouped[f"__den__{c}"].replace(0, np.nan)
            rate_frames.append((num / den).rename(c))
        df_rate = pd.concat(rate_frames, axis=1).reset_index()

        df_agg = df_sum.merge(df_rate, on=keys, how="left")
    else:
        df_agg = df_sum
    # find roles
    if "pos" in df_agg.columns:
        def infer_role(p):
            if pd.isna(p): return "forward"
            toks = re.findall(r"[A-Z]+", str(p).upper())
            if "G" in toks: return "goalie"
            if any(t in {"D", "LD", "RD", "DEF"} for t in toks): return "defense"
            return "forward"

        df_agg["role"] = df_agg["pos"].apply(infer_role)
    else:
        df_agg["role"] = "forward"

    return df_agg.dropna(subset=["season"])

def load_mp_goalies() -> pd.DataFrame:
    """
    Load Moneypuck goalies, keep ALL numeric features, and aggregate to (player, season):
      - sums for counting stats (gp, ga, sa, icetime, etc.)
      - weighted means for rate/percentage columns (svpct weighted by shots-against)
      - derives svpct = 1 - ga/sa when possible
      - infers role
    """
    inp = "data/cleaned_data/moneypuck_goalies_2008-09to2024-25.csv"
    df = pd.read_csv(inp, low_memory=False)
    # rename ID columns; keep everything else intact
    id_map = {}
    if "name" in df.columns:     id_map["name"] = "player"
    if "position" in df.columns: id_map["position"] = "pos"
    if "season" in df.columns:   id_map["season"] = "season"
    if "team" in df.columns:     id_map["team"] = "team"
    df = df.rename(columns=id_map)
    # season to int
    if "season" in df.columns:
        df["season"] = df["season"].apply(to_int)
    # gp alias
    if "gp" not in df.columns and "all_games_played" in df.columns:
        df["gp"] = pd.to_numeric(df["all_games_played"], errors="coerce")
    # standardize GA/SA and compute svpct
    # choose the best SA column available
    sa_col = None
    for cand in ["all_shotsOnGoal", "all_ongoal", "all_shotsAgainst"]:
        if cand in df.columns:
            sa_col = cand
            break
    # standardize 'ga' & 'sa'
    if "all_goals" in df.columns:
        df["ga"] = pd.to_numeric(df["all_goals"], errors="coerce")
    if sa_col:
        df["sa"] = pd.to_numeric(df[sa_col], errors="coerce")
    # derive svpct if both ga & sa exist
    if "ga" in df.columns and "sa" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["svpct"] = 1.0 - (df["ga"] / df["sa"].replace(0, np.nan))
    # cast numerics for non-ID columns
    id_cols = {"player", "team", "pos", "season"}
    for c in df.columns:
        if c not in id_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # split numeric into rate vs sum
    all_numeric = [c for c in df.columns if c not in id_cols and df[c].dtype.kind in "if"]
    rate_cols = [c for c in all_numeric if is_percent_col(c)]
    sum_cols  = [c for c in all_numeric if c not in rate_cols]
    # ensure core counters in sums (keep raw columns too if they exist)
    for col in ["gp", "ga", "sa", "all_icetime", "all_goals", "all_shotsOnGoal", "all_ongoal", "all_shotsAgainst"]:
        if col in df.columns and col not in sum_cols:
            sum_cols.append(col)
    # ensure svpct is treated as a rate
    if "svpct" in df.columns and "svpct" not in rate_cols:
        rate_cols.append("svpct")
    # keys & default weight
    keys = ["player", "season"]
    default_w = pick_weight_col(df, ["all_icetime", "sa", "all_shotsOnGoal", "all_ongoal", "gp", "all_games_played"])
    # sums to player, season
    agg_dict = {c: "sum" for c in sum_cols}
    if "team" in df.columns: agg_dict["team"] = "first"
    if "pos"  in df.columns: agg_dict["pos"]  = "first"
    df_sum = df.groupby(keys, as_index=False).agg(agg_dict)
    # vectorized weighted means for rate columns
    if rate_cols:
        d = df.copy()
        def col_weight(col: str) -> Optional[str]:
            # best weight for svpct is shots-against; otherwise default
            if col == "svpct":
                for cand in ["sa", "all_shotsOnGoal", "all_ongoal"]:
                    if cand in d.columns:
                        return cand
            return default_w if (default_w and default_w in d.columns) else None
        num_den_cols = []
        for c in rate_cols:
            x = pd.to_numeric(d[c], errors="coerce")
            wname = col_weight(c)
            if wname:
                w = pd.to_numeric(d[wname], errors="coerce").clip(lower=0).fillna(0.0)
            else:
                w = (~x.isna()).astype(float)  # unweighted fallback
            num_col = f"__num__{c}"
            den_col = f"__den__{c}"
            d[num_col], d[den_col] = x * w, w
            num_den_cols.extend([num_col, den_col])
        grouped = d.groupby(keys)[num_den_cols].sum()
        rate_frames = []
        for c in rate_cols:
            num = grouped[f"__num__{c}"]
            den = grouped[f"__den__{c}"].replace(0, np.nan)
            rate_frames.append((num / den).rename(c))
        df_rate = pd.concat(rate_frames, axis=1).reset_index()
        df_agg = df_sum.merge(df_rate, on=keys, how="left")
    else:
        df_agg = df_sum
    # find role
    if "pos" in df_agg.columns:
        df_agg["role"] = np.where(df_agg["pos"].astype(str).str.upper().str.contains("G"), "goalie", "skater")
    else:
        df_agg["role"] = "goalie"  # goalie file
    return df_agg.dropna(subset=["season"])


"""
if __name__ == "__main__":
    sk = load_hr()
    print(sk.shape)
    print(sk[['player', 'season', 'pos', 'role', 'gp', 'pts', 'pts_per_gp']].head(3))
    print(sorted([c for c in sk.columns if 'pct' in c.lower() or '%'
                  in c or 'per60' in c.lower()])[:12])

    mp_sk = load_mp_skaters()
    print(mp_sk.shape)
    print(mp_sk[['player', 'season', 'pos', 'role']].head(3))
    print(sorted([c for c in mp_sk.columns if 'per60' in c.lower() or 'percentage' in c.lower()])[:12])

    mp_g = load_mp_goalies()
    print(mp_g.shape)
    print(mp_g[['player', 'season', 'pos', 'role']].head(3))

    cols = [c for c in ['gp', 'ga', 'sa', 'svpct', 'all_goals', 'all_shotsOnGoal', 'all_ongoal', 'all_shotsAgainst'] if
            c in mp_g.columns]
    print(mp_g[cols].head(5))
"""


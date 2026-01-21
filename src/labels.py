import pandas as pd
import numpy as np
from typing import List, Dict

SKATER_DELTA_COLS = ["pts", "g", "a", "pp", "sog", "hit", "blk", "pts_per_gp", "g_per_gp", "a_per_gp"]
GOALIE_DELTA_COLS = ["svpct", "gp", "ga", "sa"]

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize keys"""
    d = df.copy()
    assert {"player", "season"}.issubset(d.columns)
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    return d.dropna(subset=["player", "season"])

def _safe_get(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """Return first existing column from candidates, else zeros."""
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)

# Skaters (G=4, A=3, +/-=0.25, PPP=1, SHP=1, GWG=0.5, SOG=0.25, HIT=0.2, BLK=0.5)
def fantasy_points_skaters(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fantasy points for skaters"""
    d = df.copy()
    gp  = pd.to_numeric(d.get("gp", np.nan), errors="coerce")

    g   = _safe_get(d, ["g"])
    a   = _safe_get(d, ["a"])
    pm  = _safe_get(d, ["plusminus", "+/-"])
    ppp = _safe_get(d, ["pp", "ppp"])
    shp = _safe_get(d, ["shp", "short_handed_points", "shortHandedPoints"])
    gwg = _safe_get(d, ["gwg", "game_winning_goals", "gameWinningGoals"])
    sog = _safe_get(d, ["sog", "shotsOnGoal", "all_shotsOnGoal"])
    hit = _safe_get(d, ["hit", "hits"])
    blk = _safe_get(d, ["blk", "blocks"])

    fp = 4*g + 3*a + 0.25*pm + 1*ppp + 1*shp + 0.5*gwg + 0.25*sog + 0.2*hit + 0.5*blk
    d["fp"] = fp
    d["fp_per_gp"] = fp / gp.replace({0: np.nan})
    return d

# Goalies (W=5, GA=-1, SV=0.2, SHO=3). Derive saves if missing.
def fantasy_points_goalies(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fantasy points for goalies"""
    d = df.copy()
    gp = pd.to_numeric(d.get("gp", np.nan), errors="coerce")

    w  = _safe_get(d, ["w", "wins", "all_wins"])
    ga = _safe_get(d, ["ga", "all_goals"])
    sa = _safe_get(d, ["sa", "all_ongoal", "all_shotsOnGoal", "all_shotsAgainst"])

    sv = _safe_get(d, ["sv", "saves"])
    if (sv == 0).all():  # derive if needed
        sv = (sa - ga).clip(lower=0)

    sho = _safe_get(d, ["sho", "shutouts", "all_shutouts"])

    fp = 5*w + 0.2*sv - 1*ga + 3*sho
    d["fp"] = fp
    d["fp_per_gp"] = fp / gp.replace({0: np.nan})
    return d

def pair_t_to_t1_skaters(sk: pd.DataFrame, min_gp_t: int = 30) -> pd.DataFrame:
    base = _prep(sk)
    base = base.loc[base.get("role", "skater") != "goalie"].copy()
    T  = fantasy_points_skaters(base)
    T1 = fantasy_points_skaters(base.copy())
    T1["season"] = T1["season"] - 1  # align t+1 back to t

    add_t  = [c for c in T.columns  if c not in ("player","season")]
    add_t1 = [c for c in T1.columns if c not in ("player","season")]
    p = T.merge(T1[["player","season"] + add_t1], on=["player","season"],
                how="inner", suffixes=("_t","_t1"))
    if "role_t" in p.columns and "role" not in p.columns:
        p["role"] = p["role_t"]
    if "gp_t" in p.columns:
        p = p[p["gp_t"].fillna(0) >= min_gp_t]

    # statistical deltas
    for c in ["pts","pts_per_gp","g","a","pp","sog","hit","blk"]:
        if f"{c}_t" in p.columns and f"{c}_t1" in p.columns:
            p[f"d_{c}"] = p[f"{c}_t1"] - p[f"{c}_t"]

    # fantasy deltas
    for c in ["fp","fp_per_gp"]:
        if f"{c}_t" in p.columns and f"{c}_t1" in p.columns:
            p[f"d_{c}"] = p[f"{c}_t1"] - p[f"{c}_t"]

    return p.reset_index(drop=True)

def pair_t_to_t1_goalies(g: pd.DataFrame, min_gp_t: int = 10) -> pd.DataFrame:
    base = _prep(g)
    base = base.loc[base.get("role", "goalie") == "goalie"].copy()
    T  = fantasy_points_goalies(base)
    T1 = fantasy_points_goalies(base.copy())
    T1["season"] = T1["season"] - 1

    add_t  = [c for c in T.columns  if c not in ("player","season")]
    add_t1 = [c for c in T1.columns if c not in ("player","season")]
    p = T.merge(T1[["player","season"] + add_t1], on=["player","season"],
                how="inner", suffixes=("_t","_t1"))
    if "role_t" in p.columns and "role" not in p.columns:
        p["role"] = p["role_t"]
    if "gp_t" in p.columns:
        p = p[p["gp_t"].fillna(0) >= min_gp_t]

    for c in ["svpct","gp","ga","sa","fp","fp_per_gp"]:
        if f"{c}_t" in p.columns and f"{c}_t1" in p.columns:
            p[f"d_{c}"] = p[f"{c}_t1"] - p[f"{c}_t"]

    return p.reset_index(drop=True)

def label_fixed_skaters(p: pd.DataFrame,
                        stat_ppg=0.18, stat_pts=15, stat_gp=45,
                        fan_ppg=0.50, fan_pts=40, fan_gp=45) -> pd.DataFrame:
    d = p.copy()
    # create two label columns initialized to 0
    d["y_stat_fixed"] = 0
    d["y_fant_fixed"] = 0
    # Statistical breakout: needs BOTH efficiency (PTS/GP) and volume (PTS) jumps,
    # and a minimum GP at t to avoid tiny-sample illusions
    scond = ((d.get("d_pts_per_gp", 0).fillna(0) >= stat_ppg) &
             (d.get("d_pts", 0).fillna(0) >= stat_pts) &
             (d.get("gp_t", 0).fillna(0) >= stat_gp))
    d.loc[scond, "y_stat_fixed"] = 1
    # Fantasy breakout: same idea but using fantasy scoring deltas
    fcond = ((d.get("d_fp_per_gp", 0).fillna(0) >= fan_ppg) &
             (d.get("d_fp", 0).fillna(0) >= fan_pts) &
             (d.get("gp_t", 0).fillna(0) >= fan_gp))
    d.loc[fcond, "y_fant_fixed"] = 1
    return d

def label_fixed_goalies(p: pd.DataFrame,
                        stat_svpct=0.010, stat_dgp=10, stat_gp=18,
                        fan_ppg=0.30, fan_pts=100, fan_gp=18) -> pd.DataFrame:
    d = p.copy()
    d["y_stat_fixed"] = 0
    d["y_fant_fixed"] = 0
    # Statistical breakout for goalies: a real SV% improvement + more workload
    scond = ((d.get("d_svpct", 0).fillna(0) >= stat_svpct) &
             (d.get("d_gp", 0).fillna(0) >= stat_dgp) &
             (d.get("gp_t", 0).fillna(0) >= stat_gp))
    d.loc[scond, "y_stat_fixed"] = 1
    # Fantasy breakout: fantasy FP deltas (per-game and season) + GP floor
    fcond = ((d.get("d_fp_per_gp", 0).fillna(0) >= fan_ppg) &
             (d.get("d_fp", 0).fillna(0) >= fan_pts) &
             (d.get("gp_t", 0).fillna(0) >= fan_gp))
    d.loc[fcond, "y_fant_fixed"] = 1
    return d

def fit_quantile_thresholds(paired: pd.DataFrame,
                            metrics: Dict[str, float],
                            group_cols: List[str]) -> pd.DataFrame:
    """
    Compute per-group quantile thresholds.
    Example:
      metrics = {"d_pts_per_gp": 0.80, "d_pts": 0.80}
      group_cols = ["season","role"]
    Returns a table keyed by group_cols with columns like: thr_d_pts_per_gp, thr_d_pts
    """
    d = paired.copy()
    rows = []
    for keys, g in d.groupby(group_cols, dropna=False):
        # keys may be a scalar or tuple; normalize to dict
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for m, q in metrics.items():
            # use nan if metric is missing in this group (rare)
            row[f"thr_{m}"] = g[m].quantile(q) if m in g.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def apply_quantile_labels(paired: pd.DataFrame,
                          thresh_df: pd.DataFrame,
                          metrics: List[str],
                          group_cols: List[str],
                          out_col: str) -> pd.DataFrame:
    """
    Label = 1 when, for a row's group, every metric >= its prefit group threshold.
    """
    d = paired.copy()
    # merge thresholds to each row (by the same grouping)
    merged = d.merge(thresh_df, on=group_cols, how="left")
    # start with all-True, then AND constraints one-by-one
    cond = pd.Series(True, index=merged.index)
    for m in metrics:
        thr_col = f"thr_{m}"
        # if a threshold is missing for some group, make it impossible to pass (Inf)
        thr = merged[thr_col].fillna(np.inf)
        val = merged[m].fillna(-np.inf)
        cond &= (val >= thr)
    d[out_col] = 0
    d.loc[cond, out_col] = 1
    return d

def finalize_labels_skaters(df: pd.DataFrame) -> pd.DataFrame:
    # keep keys, roles, some sanity cols, deltas, and all labels
    must = ["player", "season", "role"]
    diag = [c for c in ["gp_t", "gp_t1", "pts_t", "pts_t1", "pts_per_gp_t", "pts_per_gp_t1"] if c in df.columns]
    deltas = [c for c in df.columns if c.startswith("d_")]
    labels = [c for c in df.columns if c.startswith("y_")]
    keep = [c for c in (must + diag + deltas + labels) if c in df.columns]
    df["is_prospect"] = (df["age_t"].fillna(99) <= 23) & (df["gp_t"].fillna(0) <= 50)  # Flag low-GP youth
    return df[keep].reset_index(drop=True)

def finalize_labels_goalies(df: pd.DataFrame) -> pd.DataFrame:
    must = ["player", "season"]
    diag = [c for c in ["gp_t", "gp_t1", "svpct_t", "svpct_t1"] if c in df.columns]
    deltas = [c for c in df.columns if c.startswith("d_")]
    labels = [c for c in df.columns if c.startswith("y_")]
    keep = [c for c in (must + diag + deltas + labels) if c in df.columns]
    return df[keep].reset_index(drop=True)

def extend_thresholds_to_all_seasons(thr: pd.DataFrame, all_seasons: list[int], group_cols: list[str]) -> pd.DataFrame:
    """
    Given a threshold table keyed by group_cols (e.g., ['season','role']),
    create rows for EVERY season in `all_seasons` and forward-fill thresholds
    within each group (e.g., per role). This reuses TRAIN-era cutoffs for VAL/TEST.
    """
    thr = thr.copy()
    has_role = "role" in group_cols and "role" in thr.columns
    has_season = "season" in group_cols

    if not has_season:
        return thr

    # Build a grid of all (group) combinations we need thresholds for
    if has_role:
        roles = sorted(thr["role"].dropna().unique().tolist())
        grid = pd.MultiIndex.from_product([roles, sorted(set(all_seasons))],
                                          names=["role","season"]).to_frame(index=False)
        merged = grid.merge(thr, on=["role","season"], how="left")
        thr_cols = [c for c in merged.columns if c.startswith("thr_")]
        merged = merged.sort_values(["role","season"])
        merged[thr_cols] = merged.groupby("role", dropna=False)[thr_cols].ffill().bfill()
        return merged
    else:
        grid = pd.DataFrame({"season": sorted(set(all_seasons))})
        merged = grid.merge(thr, on=["season"], how="left")
        thr_cols = [c for c in merged.columns if c.startswith("thr_")]
        merged = merged.sort_values(["season"])
        merged[thr_cols] = merged[thr_cols].ffill().bfill()
        return merged

def tighten_skater_stat_quant(
    p: pd.DataFrame,
    min_d_pts: int = 15,
    min_d_ppg: float = 0.18,
    min_gp_t: int = 30,
    min_gp_t1: int = 45,
    in_col: str = "y_stat_quant",
    out_col: str = "y_stat_quant_strict",
) -> pd.DataFrame:
    """
    Stricter skater breakout: require quantile label AND absolute delta floors AND real next-season volume.
    """
    d = p.copy()
    for c in ["d_pts", "d_pts_per_gp", "gp_t", "gp_t1", in_col]:
        if c not in d.columns:
            d[c] = np.nan if c.startswith("d_") else 0
    cond = (
        (d[in_col] == 1)
        & (d["d_pts"].fillna(0) >= min_d_pts)
        & (d["d_pts_per_gp"].fillna(0) >= min_d_ppg)
        & (d["gp_t"].fillna(0) >= min_gp_t)
        & (d["gp_t1"].fillna(0) >= min_gp_t1)
    )
    d[out_col] = 0
    d.loc[cond, out_col] = 1
    return d

def mask_skaters_candidates(p: pd.DataFrame) -> pd.Series:
    d = p.copy()
    # already-elite in t: top quartile by pts_per_gp_t within (season, role)
    if ("pts_per_gp_t" in d.columns) and {"season","role"}.issubset(d.columns):
        q = d.groupby(["season","role"])["pts_per_gp_t"].transform(lambda s: s.quantile(0.75))
        elite_t = (d["pts_per_gp_t"].fillna(0) >= q.fillna(np.inf))
    else:
        elite_t = pd.Series(False, index=d.index)

    # minimum plausible usage (if columns exist)
    has_usage = (
        d.get("toi_t", pd.Series(0, index=d.index)).fillna(0) > 500
    ) | (
        d.get("pp_t", pd.Series(0, index=d.index)).fillna(0) > 5
    )

    return (~elite_t) & has_usage

def _agebin(x):
    try:
        a = float(x)
    except Exception:
        return "NA"
    if a <= 22: return "u23"
    if a <= 24: return "23_24"
    if a <= 27: return "25_27"
    if a <= 30: return "28_30"
    return "31p"

def fit_quantile_thresholds_safe(df, targets, group_cols, min_pos=3, try_quantiles=(0.90,0.85,0.80,0.75,0.70)):
    d = df.copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d = d.dropna(subset=["season"])
    # start with the original grouping
    group_tiers = [group_cols, [c for c in group_cols if c != "age_bin_t"], ["season"], []]
    for q in try_quantiles:
        for gc in group_tiers:
            # build thresholds per group
            rows = []
            g = d.groupby(gc) if gc else [((), d)]
            for key, part in g:
                rec = {}
                if gc:
                    if not isinstance(key, tuple): key = (key,)
                    for k, v in zip(gc, key): rec[k] = v
                for tgt, _ in targets.items():
                    rec[tgt] = pd.to_numeric(part[tgt], errors="coerce").quantile(q)
                rows.append(rec)
            thr = pd.DataFrame(rows)
            # apply to df to test positivity rate on this data window
            lab = d.copy()
            for tgt in targets.keys():
                lab[f"y_{tgt}"] = 0
            if gc:
                lab = lab.merge(thr, on=gc, how="left", suffixes=("","_thr"))
            else:
                for tgt in targets.keys():
                    lab[f"{tgt}_thr"] = thr.iloc[0][tgt]
            pos = 0
            for tgt in targets.keys():
                lab[f"y_{tgt}"] = (pd.to_numeric(lab[tgt], errors="coerce") >= pd.to_numeric(lab[f"{tgt}_thr"], errors="coerce")).astype(int)
                pos += lab[f"y_{tgt}"]
            if int(pos.sum()) >= min_pos:
                return thr.rename(columns={k: f"{k}" for k in targets.keys()}), gc, q
    # worst case: return global 70th percentile with no grouping
    fallback = {}
    for tgt in targets.keys():
        fallback[tgt] = pd.to_numeric(d[tgt], errors="coerce").quantile(0.70)
    return pd.DataFrame([fallback]), [], 0.70

def label_defense_skaters(p: pd.DataFrame,
                          stat_blk=20, stat_take=10, stat_hit=25,
                          min_gp=40) -> pd.DataFrame:
    p = p.copy()
    cond_def = (
        (p["blk_t1"] - p["blk_t"] >= stat_blk) |
        (p["take_t1"] - p["take_t"] >= stat_take) |
        (p["hit_t1"] - p["hit_t"] >= stat_hit)
    ) & (p["gp_t1"] >= min_gp)
    p["y_def_breakout"] = cond_def.astype(int)
    return p

def label_goalie_breakouts(
    p: pd.DataFrame,
    stat_svpct: float = 0.005,   # +0.5 percentage points
    stat_saves: int = 150,       # +150 saves YoY
    min_gp: int = 20             # GP floor in t+1 season
) -> pd.DataFrame:
    """
    Create a fixed goalie breakout label y_goalie_breakout based on:
      - ΔSV% >= stat_svpct  OR  Δsaves >= stat_saves
      AND GP_t1 >= min_gp

    Handles varied column names and derives missing metrics:
      SV%: uses 'sv_pct_*' or 'svpct_*', else derives as 1 - GA/SA
      Saves: uses 'saves_*' or 'sv_*', else derives as SA - GA
    """
    df = p.copy()

    def pick(*names):
        """Return the first column name that exists in df, else None."""
        for n in names:
            if n in df.columns:
                return n
        return None

    def series_or_zero(col):
        """Return a Series for col if present, else a 0-Series aligned to df.index."""
        if col and col in df.columns:
            return df[col]
        return pd.Series(0, index=df.index, dtype="float64")

    # --- SV% columns (try both styles); derive if absent ---
    sv_t  = pick("sv_pct_t",  "svpct_t")
    sv_t1 = pick("sv_pct_t1", "svpct_t1")

    if sv_t is None or sv_t1 is None:
        sa_t  = pick("sa_t",  "shots_against_t")
        ga_t  = pick("ga_t",  "goals_against_t")
        sa_t1 = pick("sa_t1", "shots_against_t1")
        ga_t1 = pick("ga_t1", "goals_against_t1")

        sv_series_t  = pd.Series(np.nan, index=df.index, dtype="float64")
        sv_series_t1 = pd.Series(np.nan, index=df.index, dtype="float64")
        if sa_t and ga_t:
            sv_series_t  = 1.0 - (series_or_zero(ga_t)  / series_or_zero(sa_t).replace(0, np.nan))
        if sa_t1 and ga_t1:
            sv_series_t1 = 1.0 - (series_or_zero(ga_t1) / series_or_zero(sa_t1).replace(0, np.nan))

        df["__svpct_t"]  = sv_series_t
        df["__svpct_t1"] = sv_series_t1
        sv_t, sv_t1 = "__svpct_t", "__svpct_t1"

    # --- Saves (use provided columns, else derive as SA - GA) ---
    saves_t  = pick("saves_t",  "sv_t")
    saves_t1 = pick("saves_t1", "sv_t1")

    if saves_t is None or saves_t1 is None:
        sa_t  = pick("sa_t",  "shots_against_t")
        ga_t  = pick("ga_t",  "goals_against_t")
        sa_t1 = pick("sa_t1", "shots_against_t1")
        ga_t1 = pick("ga_t1", "goals_against_t1")
        if saves_t is None and sa_t and ga_t:
            df["__saves_t"] = series_or_zero(sa_t)  - series_or_zero(ga_t)
            saves_t = "__saves_t"
        if saves_t1 is None and sa_t1 and ga_t1:
            df["__saves_t1"] = series_or_zero(sa_t1) - series_or_zero(ga_t1)
            saves_t1 = "__saves_t1"

    # --- GP in t+1 (for floor) ---
    gp_t1 = pick("gp_t1", "games_played_t1", "gp_next")

    # --- Deltas as Series (never scalars) ---
    sv_delta     = (series_or_zero(sv_t1)    - series_or_zero(sv_t)).astype("float64").fillna(0)
    saves_delta  = (series_or_zero(saves_t1) - series_or_zero(saves_t)).astype("float64").fillna(0)
    gp_t1_series = series_or_zero(gp_t1).astype("float64").fillna(0)

    # --- Breakout condition ---
    cond = (
            (df["d_svpct"] >= 0.005) &  # +0.5% save %
            (df["d_gp"] >= 15)  # +15 games
    )
    df["y_goalie_breakout"] = cond.astype(int)

    return df
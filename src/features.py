import numpy as np
import pandas as pd


def add_zscore(df, group_cols, value_col, out_col, ddof: int=0) -> pd.DataFrame:
    """
    Vectorized within-group z-score. If std==0 or NaN -> z=0 for that group
    """
    d = df.copy()
    x = pd.to_numeric(d[value_col], errors="coerce")
    grp = d.groupby(group_cols, dropna=False)
    # calculate mean
    mean = grp[value_col].transform("mean")
    # calculate std
    std = grp[value_col].transform(lambda s: s.std(ddof=ddof))
    # avoid /0
    safe = std.replace(0, np.nan)
    z = (x-mean)/safe
    d[out_col] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d

def make_skater_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Minimal but strong skater features. Assumes df has player, season, role, and HR totals.
        """
    d = df.copy()
    # ensure expected numeric cols exist (don’t impute yet; modeling will handle NaNs)
    needed = [
        "age", "gp", "g", "a", "pts", "pp", "hit", "blk", "take", "give", "pim", "plusminus",
        "sog", "shpct", "pts_per_gp", "g_per_gp", "a_per_gp", "pp_per_gp",
        "hit_per_gp", "blk_per_gp", "take_per_gp", "give_per_gp", "sog_per_gp"
    ]
    for c in needed:
        if c not in d.columns:
            d[c] = np.nan
        else:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # z-scores within (season, role)
    if "pts" in d.columns:
        d = add_zscore(d, ["season", "role"], "pts", "pts_z")
    # simple "defense index" blending scoring + peripherals
    d["def_index"] = (
            0.55 * d["pts"].fillna(0) +
            0.15 * d["pp"].fillna(0) +
            0.15 * d["blk"].fillna(0) +
            0.15 * d["hit"].fillna(0)
    )
    d = add_zscore(d, ["season", "role"], "def_index", "def_z")
    # optional curvature for age
    if "age" in d.columns:
        d["age2"] = (d["age"] ** 2).astype(float)
    rolling_cols = ["pts", "pts_per_gp", "sog_per_gp", "toi"] if all(
        c in d.columns for c in ["pts", "pts_per_gp", "sog_per_gp", "toi"]) else []
    if rolling_cols:
        d = d.sort_values(["player", "season"])
        d[[f"{c}_roll3" for c in rolling_cols]] = d.groupby("player")[rolling_cols].rolling(window=3,
                                                                                            min_periods=1).mean().reset_index(
            0, drop=True)

    # NEW: Team context (assuming 'team' and MoneyPuck has team_xGoals; aggregate if needed)
    if "team" in d.columns and "onIce_xGoalsPercentage_mp" in d.columns:  # From merge_skaters
        d["team_xg_pct"] = d.groupby(["season", "team"])["onIce_xGoalsPercentage_mp"].transform("mean").fillna(0.5)
    return d

def make_goalie_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal goalie features. Assumes df has gp, svpct (and optionally ga, sa).
    """
    g = df.copy()
    # ensure numerics
    for c in ["gp","svpct","ga","sa"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    # usage proxy within season: games share (safe with NaNs)
    if "gp" in g.columns:
        max_gp = g.groupby("season")["gp"].transform("max")
        g["gp_share"] = (g["gp"] / max_gp).replace([np.inf, -np.inf], np.nan)
    # sv% z-score across the league per season
    if "svpct" in g.columns:
        g = add_zscore(g, ["season"], "svpct", "svpct_z")
    return g

def merge_skaters(hr_df: pd.DataFrame, mp_df: pd.DataFrame, mp_suffix: str = "_mp") -> pd.DataFrame:
    """
    Merge HR (base identity/totals) with Moneypuck (advanced).
    - Join on (player, season).
    - Keep all MP numeric columns with suffix to avoid collisions.
    - Fill missing team/pos from MP when HR is NaN.
    """
    # avoid accidental collisions on id cols
    mp = mp_df.copy()
    for c in ["player","season"]:
        if c not in mp.columns:
            raise ValueError(f"Moneypuck skaters missing key column: {c}")
    # Prepare to keep MP advanced columns distinct
    merged = hr_df.merge(
        mp,
        on=["player","season"],
        how="left",
        suffixes=("", mp_suffix)
    )
    # If team/pos missing on HR, fill from MP and drop MP duplicates
    for col in ["team","pos"]:
        col_mp = col + mp_suffix
        if col in merged.columns and col_mp in merged.columns:
            merged[col] = merged[col].fillna(merged[col_mp])
            merged = merged.drop(columns=[col_mp])
    return merged

def add_lag_features(df: pd.DataFrame, cols: list[str], season_col: str = "season") -> pd.DataFrame:
    """
    Add t-1 columns (*_prev) and deltas (*_chg = t - (t-1)) for given cols.
    Safe if some prev values are missing.
    """
    base = df.copy()
    prev = base[["player", season_col] + cols].copy()
    prev[season_col] = prev[season_col] + 1  # align (t-1) => t
    prev = prev.rename(columns={c: f"{c}_prev" for c in cols})

    out = base.merge(prev, on=["player", season_col], how="left")
    for c in cols:
        out[f"{c}_chg"] = out[c] - out[f"{c}_prev"]
    return out

def add_context_zscores(feats: pd.DataFrame) -> pd.DataFrame:
    d = feats.copy()
    for c in ["toi", "sog"]:
        if c in d.columns and "role" in d.columns:
            d = add_zscore(d, ["season", "role"], c, f"{c}_z", ddof=0)
    return d

def add_shooting_luck(feats: pd.DataFrame) -> pd.DataFrame:
    d = feats.copy()
    # individual: compare shooting% to career/rolling baseline proxy
    # If you don’t have career baselines, just use current SOG and SH% interaction
    if "shpct" in d.columns and "sog_per_gp" in d.columns:
        d["shot_eff"] = (d["shpct"].fillna(0) * d["sog_per_gp"].fillna(0))
    return d

def add_age_penalty(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "age" in d.columns:
        d["age_over_27"] = np.clip(d["age"] - 27.0, 0, None)
    return d
# src/run.py
from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.io import load_hr, load_mp_goalies
from src.features import make_skater_features, make_goalie_features, add_lag_features, add_context_zscores, add_shooting_luck, add_age_penalty
from src.labels import (
    pair_t_to_t1_skaters, pair_t_to_t1_goalies,
    label_fixed_skaters, label_fixed_goalies,
    fit_quantile_thresholds, apply_quantile_labels,
    finalize_labels_skaters, finalize_labels_goalies,
    extend_thresholds_to_all_seasons,
    tighten_skater_stat_quant, _agebin,
    label_defense_skaters, label_goalie_breakouts
)
from src.split import split_by_season, assemble_dataset

DEFENSE_KEEP_ANY = ("hit", "blk", "take", "give", "toi", "gp", "age", "z", "chg", "lag", "season", "role", "player", "team")
DEFENSE_DROP_ANY  = ("pts", "pts_per_gp", "g", "a", "pp", "sog", "sog_per_gp", "xg", "ixg", "fp", "shpct")

def _prune_for_defense(feats: pd.DataFrame) -> pd.DataFrame:
    # whitelist core defensive & contextual signals and drop obvious scoring proxies
    keep = []
    for c in feats.columns:
        if any(k in c for k in DEFENSE_KEEP_ANY) and not any(bad in c for bad in DEFENSE_DROP_ANY):
            keep.append(c)
    # always keep identifiers
    for base in ("player","season","role","team", "age"):
        if base in feats.columns and base not in keep:
            keep.append(base)
    return feats.loc[:, keep]



def tune_model(model, X_tr, y_tr):
    """
    Tune model hyperparameters using GridSearchCV.
    """
    if "xgb" in str(type(model.named_steps['clf'])):
        param_grid = {
            'clf__max_depth': [4, 6, 8],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.7, 0.8, 0.9]
        }
    elif "hgbt" in str(type(model.named_steps['clf'])):
        param_grid = {
            'clf__max_leaf_nodes': [31, 63, 127],
            'clf__learning_rate': [0.01, 0.05, 0.1]
        }
    else:
        return model  # Skip for logreg

    grid = GridSearchCV(model, param_grid, cv=3, scoring="average_precision", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    return grid.best_estimator_

def build_labels(role: str, train_end: int, label_style, quantile: float = 0.75):
    role = role.lower()
    if role == "skater":
        sk = make_skater_features(load_hr())
        p = pair_t_to_t1_skaters(sk, min_gp_t=30)
        p = label_fixed_skaters(p)
        # add age bin based on t-season age
        p["age_bin_t"] = p["age_t"].apply(_agebin)
        if label_style == "defense":
            # expects label_defense_skaters to create p["y_defense_fixed"] (0/1)
            p = label_defense_skaters(p)
        sk_train = p[p["season"] <= train_end]
        thr_stat = fit_quantile_thresholds(sk_train, {"d_pts_per_gp": quantile, "d_pts": quantile}, ["season","role"])
        thr_fant = fit_quantile_thresholds(sk_train, {"d_fp_per_gp": quantile, "d_fp": quantile}, ["season","role"])

        # ensure thresholds exist for all later seasons (no leakage; forward-fill)
        all_seasons = p["season"].dropna().astype(int).unique().tolist()
        thr_stat = extend_thresholds_to_all_seasons(thr_stat, all_seasons, ["season","role"])
        thr_fant = extend_thresholds_to_all_seasons(thr_fant, all_seasons, ["season","role"])

        p = apply_quantile_labels(p, thr_stat, ["d_pts_per_gp","d_pts"], ["season","role"], "y_stat_quant")
        p = apply_quantile_labels(p, thr_fant, ["d_fp_per_gp","d_fp"], ["season","role"], "y_fant_quant")

        p = tighten_skater_stat_quant(
            p,
            min_d_pts=15,
            min_d_ppg=0.18,
            min_gp_t=45,
            min_gp_t1=45
        )

        return finalize_labels_skaters(p)

    elif role == "goalie":
        gg = make_goalie_features(load_mp_goalies())
        #print("Goalie features:", gg.columns.tolist())

        p = pair_t_to_t1_goalies(gg, min_gp_t=6)
        p = label_fixed_goalies(p)

        p = label_goalie_breakouts(p)

        g_train = p[p["season"] <= train_end]
        thr_stat = fit_quantile_thresholds(g_train, {"d_svpct": quantile, "d_gp": quantile}, ["season"])
        thr_fant = fit_quantile_thresholds(g_train, {"d_fp_per_gp": quantile, "d_fp": quantile}, ["season"])

        all_seasons = p["season"].dropna().astype(int).unique().tolist()
        thr_stat = extend_thresholds_to_all_seasons(thr_stat, all_seasons, ["season"])
        thr_fant = extend_thresholds_to_all_seasons(thr_fant, all_seasons, ["season"])

        p = apply_quantile_labels(p, thr_stat, ["d_svpct","d_gp"], ["season"], "y_stat_quant")
        p = apply_quantile_labels(p, thr_fant, ["d_fp_per_gp","d_fp"], ["season"], "y_fant_quant")

        return finalize_labels_goalies(p)

    else:
        raise ValueError("role must be 'skater' or 'goalie'")


def build_features(role: str, label_style) -> pd.DataFrame:
    role = role.lower()
    if role == "skater":
        feats = make_skater_features(load_hr())
    elif role == "goalie":
        feats = make_goalie_features(load_mp_goalies())
        lag_cols = [c for c in ["gp", "svpct", "sa", "ga", "gp_share"] if c in feats.columns]
        feats = add_lag_features(feats, lag_cols)

    else:
        raise ValueError("role must be 'skater' or 'goalie'")

    # === HIGH-ROI FEATURE BOOST ===
    # Add lagged t-1 columns and deltas (t - (t-1)) for core stats (safe if missing)
    lag_cols = [c for c in ["pts","pts_per_gp","g","a","pp","toi","sog","hit","blk"] if c in feats.columns]
    if lag_cols:
        feats = add_lag_features(feats, lag_cols)
    feats = add_context_zscores(feats)
    feats = add_shooting_luck(feats)
    feats = add_age_penalty(feats)

    # >>> PRUNE OFFENSE FOR DEFENSE TASK <<<
    if (role == "skater") and (label_style == "defense"):
        feats = _prune_for_defense(feats)

    return feats


def _apply_min_gp(X, y, meta, min_gp: int | None):
    if (min_gp is None) or ("gp_t" not in meta.columns):
        return X, y, meta
    # build mask from meta
    m = (meta["gp_t"].fillna(0) >= int(min_gp))
    # align mask to X index (in case indices differ)
    m = m.reindex(X.index, fill_value=False)
    Xf = X.loc[m]
    metaf = meta.loc[m]
    if y is None:
        return Xf, y, metaf
    # ensure y aligns with X index
    if isinstance(y, pd.Series):
        y = y.reindex(X.index)
    else:
        y = pd.Series(y, index=X.index)
    yf = y.loc[m]
    return Xf, yf, metaf

def assemble_inference(feats: pd.DataFrame,
                       season_t: int,
                       feature_cols: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build X_inf and meta_inf for a given season t (predict t+1).
    - Select only rows where feats['season'] == season_t
    - Ensure X columns are exactly the same (ordering too) as training (feature_cols)
    - Build meta with the columns rank_candidates expects (team_t, gp_t, sog_per_gp_t, etc.)
    """
    f = feats[feats["season"] == season_t].copy()
    if f.empty:
        raise ValueError(f"No rows found for season {season_t} in features.")

    # keep X and meta perfectly aligned
    f = f.reset_index(drop=True)  # <<< add this

    X_inf = f.reindex(columns=feature_cols, fill_value=np.nan)

    keep = ["player", "season", "role", "team", "gp", "sog_per_gp", "toi_z",
            "age", "toi_chg", "sog_per_gp_chg", "pp_per_gp_chg",
            "hit_chg", "blk_chg", "take_chg", "give_chg",
            "hit", "blk", "take", "give"]
    present = [c for c in keep if c in f.columns]
    meta_inf = f[present].copy()
    if "team" in meta_inf.columns:       meta_inf = meta_inf.rename(columns={"team": "team_t"})
    if "gp" in meta_inf.columns:         meta_inf = meta_inf.rename(columns={"gp": "gp_t"})
    if "sog_per_gp" in meta_inf.columns: meta_inf = meta_inf.rename(columns={"sog_per_gp": "sog_per_gp_t"})
    if "age" in meta_inf.columns:        meta_inf = meta_inf.rename(columns={"age": "age_t"})
    return X_inf, meta_inf

def main():
    global youth_27, youth_24
    ap = argparse.ArgumentParser(description="Train ML model to identify NHL breakout candidates.")
    ap.add_argument("--role", choices=["skater","goalie"], default="skater")
    ap.add_argument("--label_family", choices=["stat","fant"], default="stat")
    ap.add_argument("--label_style", choices=["fixed","quant","quant_strict", "defense"], default="quant")
    ap.add_argument("--model", choices=["hgbt","logreg"], default="hgbt")
    ap.add_argument("--train_end", type=int, default=2021)
    ap.add_argument("--val_end", type=int, default=2022)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--min_gp", type=int, default=45, help="Filter gp_t < this for VAL/TEST & leaderboard.")  # DEFAULT=45
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()
    # === dynamic model import ===
    if args.role == "goalie":
        from src import model_goalies as model_module
    else:
        from src import model as model_module
    build_model = model_module.build_model
    train = model_module.train
    predict_proba = model_module.predict_proba
    calibrate_model = model_module.calibrate_model
    # 1) Build labels
    labels_df = build_labels(role=args.role, train_end=args.train_end, label_style=args.label_style, quantile=0.80)

    # 2) Choose label column
    if args.label_style == "defense":
        lbl = "y_def_breakout"  # from label_defense_skaters(...)
    elif args.role == "goalie" and args.label_style == "fixed":
        lbl = "y_goalie_breakout"  # from label_goalie_breakouts(...)
    else:
        lbl = f"y_{args.label_family}_{args.label_style}"

    if lbl not in labels_df.columns:
        raise ValueError(
            f"Chosen label '{lbl}' not found. Available: "
            f"{[c for c in labels_df.columns if c.startswith('y_')]}"
        )

    # 3) Build features & split
    feats = build_features(role=args.role, label_style=args.label_style)
    split = split_by_season(labels_df, train_end=args.train_end, val_end=args.val_end)

    X_tr, y_tr, meta_tr = assemble_dataset(feats, labels_df, split["train"], label_col=lbl)
    X_va, y_va, meta_va = assemble_dataset(feats, labels_df, split["val"],   label_col=lbl)
    X_te, y_te, meta_te = assemble_dataset(feats, labels_df, split["test"],  label_col=lbl)

    # Apply min GP=45 to VAL/TEST (does not touch training set)
    X_va, y_va, meta_va = _apply_min_gp(X_va, y_va, meta_va, args.min_gp)
    X_te, y_te, meta_te = _apply_min_gp(X_te, y_te, meta_te, args.min_gp)

    # 4) Build & train model (HGBT has built-in early stopping)
    model = build_model(args.model)
    model = train(model, X_tr, y_tr, balance=True)

    # Optional: calibrate on validation set (often improves ranking coherence)
    cal = calibrate_model(model, X_va, y_va)  # returns calibrated wrapper or None
    M = cal if cal is not None else model

    # 5) Evaluate
    s_va = predict_proba(M, X_va)
    s_te = predict_proba(M, X_te)
    #print(f"[VAL] {args.role} {lbl} ::", evaluate(y_va.values, s_va, ks=[10,20,50]))
    #print(f"[TEST] {args.role} {lbl} ::", evaluate(y_te.values, s_te, ks=[10,20,50]))

    # ==== INFERENCE: predict upcoming season (t+1) from latest t ====
    # We'll use the latest season in your features as t (e.g., 2024 -> predict 2025)
    latest_t = int(feats["season"].max())

    # critical: use the exact training feature columns
    feature_cols = X_tr.columns

    # build inference X/meta for season t
    X_inf, meta_inf = assemble_inference(feats, latest_t, feature_cols)

    # optional: apply same GP filter you use for eval/leaderboard
    X_inf, _, meta_inf = _apply_min_gp(X_inf, pd.Series(np.zeros(len(X_inf))), meta_inf, args.min_gp)

    # score & rank
    from src.predict import rank_candidates
    trend = {"toi_chg": 20, "sog_per_gp_chg": 0.20, "pp_per_gp_chg": 0.06}
    min_sog_pg = 2.0

    if args.label_style == "defense":
        trend = {"toi_chg": 10, "hit_chg": 5, "blk_chg": 5, "take_chg": 2}
        min_sog_pg = 0.0
    board_inf = rank_candidates(
        M, X_inf, meta_inf,
        top_k=args.top_k,
        min_gp=args.min_gp,
        min_sog_pg=min_sog_pg,
        require_pos_toi=False,
        require_any_trend=True,
        trend_mins=trend,
        use_age_discount=True, age_discount_alpha=0.01,
    )

    # --- DIAGNOSTICS (put BEFORE printing tables) ---
    #print("\n[Diag] Inference rows before filters:", len(meta_inf))  # after min_gp gating but before rank filters
    #print("[Diag] Inference rows after filters:", len(board_inf))
    #if "age_t" in board_inf.columns:
    #    print("[Diag] Age mix (top_k):")
    #    print(
    #        board_inf["age_t"]
    #        .dropna()
    #        .astype(int)
    #        .value_counts(bins=[0, 22, 24, 27, 30, 100])
    #        .sort_index()
    #    )

    if args.label_style == "defense":
        pref_cols = ["player", "season", "role", "team_t", "gp_t", "hit", "blk", "take", "give", "hit_chg", "blk_chg",
                     "take_chg", "score"]
    else:
        pref_cols = ["player", "season", "role", "team_t", "gp_t", "sog_per_gp_t", "score"]

    cols = [c for c in pref_cols if c in board_inf.columns]

    #print(f"\nPredicted candidates for {latest_t + 1} (using t={latest_t} features):")
    #print(board_inf[cols].head(args.top_k).to_string(index=False))

    # --- YOUTH view ---
    if "age_t" in board_inf.columns:
        youth_27 = board_inf.loc[board_inf["age_t"].fillna(99) <= 27]
        if not youth_27.empty:
            print("\nPredicted candidates (youth view, age<27):")
            print(youth_27[cols].head(args.top_k).to_string(index=False))

    # --- super YOUTH view ---
    if "age_t" in board_inf.columns:
        youth_24 = board_inf.loc[board_inf["age_t"].fillna(99) <= 24]
        if not youth_24.empty:
            print("\nPredicted candidates (youth view, age<24):")
            print(youth_24[cols].head(args.top_k).to_string(index=False))

    # optional: save a CSV just for the inference year if --out_csv was provided
    if args.out_csv:
        root, ext = os.path.splitext(args.out_csv)
        out_inf = f"{root}_pred_for_{latest_t + 1}{ext}"
        os.makedirs(os.path.dirname(out_inf) or ".", exist_ok=True)
        board_inf.to_csv(out_inf, index=False)
        print(f"Saved inference leaderboard -> {out_inf}")

    # --- SAVE youth CSVs when --out_csv is provided ---
    if args.out_csv:
        root, ext = os.path.splitext(args.out_csv)
        out_inf = f"{root}_pred_for_{latest_t + 1}{ext}"
        os.makedirs(os.path.dirname(out_inf) or ".", exist_ok=True)
        board_inf.to_csv(out_inf, index=False)
        print(f"Saved inference leaderboard -> {out_inf}")

        # Save youth subsets (top_k view)
        if "age_t" in board_inf.columns:
            if not youth_27.empty:
                out27 = f"{root}_youth_u27_for_{latest_t + 1}{ext}"
                youth_27.head(args.top_k).to_csv(out27, index=False)
                print(f"Saved youth (<27) -> {out27}")
            if not youth_24.empty:
                out24 = f"{root}_youth_u24_for_{latest_t + 1}{ext}"
                youth_24.head(args.top_k).to_csv(out24, index=False)
                print(f"Saved youth (<24) -> {out24}")



if __name__ == "__main__":
    main()
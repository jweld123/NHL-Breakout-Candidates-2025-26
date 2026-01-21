# src/predict.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

from src.model import predict_proba


def rank_candidates(model,
                    X: pd.DataFrame,
                    meta: pd.DataFrame,
                    top_k: Optional[int] = 50,
                    min_gp: Optional[int] = None,
                    min_sog_pg: Optional[float] = None,
                    require_pos_toi: bool = False,
                    max_age: Optional[int] = None,
                    age_col: str = "age_t",
                    score_col: str = "score",
                    use_age_discount: bool = False,
                    age_discount_alpha: float = 0.03,
                    require_any_trend: bool = False,
                    trend_mins: Optional[dict] = None
                    ) -> pd.DataFrame:
    """
    Score rows, optionally filter by gp_t, and return sorted ranking.
    meta should contain at least: player, season and (optionally) gp_t, role, team_t
    """
    scores = predict_proba(model, X)
    out = meta.copy()
    out[score_col] = scores
    if (min_gp is not None) and ("gp_t" in out.columns):
        out = out[out["gp_t"].fillna(0) >= int(min_gp)]
    if (min_sog_pg is not None):
        sog_col = "sog_per_gp_t" if "sog_per_gp_t" in out.columns else None
        if sog_col is not None:
            out = out[out[sog_col].fillna(0) >= float(min_sog_pg)]
    if require_pos_toi and ("toi_z" in out.columns):
        out = out[out["toi_z"].fillna(0) > 0]

    # require an upward trend in at least one metric
    if require_any_trend:
        thr = trend_mins or {"toi_chg": 15.0, "sog_per_gp_chg": 0.15, "pp_per_gp_chg": 0.05}
        conds = []
        for k, t in thr.items():
            if k in out.columns:
                conds.append(out[k].fillna(0) >= float(t))
        if conds:
            keep = conds[0]
            for c in conds[1:]:
                keep = keep | c
            out = out[keep]

    if (max_age is not None) and (age_col in out.columns):
        out = out[out[age_col].fillna(99) <= int(max_age)]

    if use_age_discount and (age_col in out.columns):
        def _age_discount(age, pivot=27.0, alpha=age_discount_alpha):
            over = np.clip(pd.to_numeric(age, errors="coerce") - pivot, 0, None)
            disc = 1.0 - alpha * over
            return np.clip(disc, 0.6, 1.0)

        out["score_adj"] = out[score_col].values * _age_discount(out[age_col].values)
        sort_col = "score_adj"
    else:
        sort_col = score_col

    out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    if top_k is not None:
        out = out.head(int(top_k))
    return out
# split.py
import pandas as pd
from typing import List, Dict, Tuple

def seasons_sorted(df: pd.DataFrame) -> List[int]:
    """
    Return sorted unique season 'start years'
    """
    s = pd.to_numeric(df["season"], errors="coerce").dropna().astype(int)
    return sorted(s.unique().tolist())

def subset_by_seasons(df: pd.DataFrame, keep: List[int]) -> pd.DataFrame:
    """
    Convenience filter by a list of seasons (works for labels or features).
    """
    return df[df["season"].isin(keep)].copy()

def split_by_season(labels: pd.DataFrame, train_end: int, val_end: int) -> Dict[str, List[int]]:
    """
    Build a season-based split dict. Example:
      train: <= train_end
      val:   (train_end, ..., val_end]
      test:  (> val_end, up to the max season in `labels`)
    Returns lists of season *start years* for each split.
    """
    all_seasons = seasons_sorted(labels)
    train_seasons = [y for y in all_seasons if y <= train_end]
    val_seasons   = [y for y in all_seasons if (train_end < y <= val_end)]
    test_seasons  = [y for y in all_seasons if y > val_end]
    return {"train": train_seasons, "val": val_seasons, "test": test_seasons}

def rolling_splits(labels: pd.DataFrame, min_train: int = 5, val_window: int = 1, test_window: int = 1, step: int = 1) -> List[Dict[str, List[int]]]:
    """
    Generate rolling expanding-window splits.
    For each split i:
      train: from first season up to `train_end`
      val:   next `val_window` seasons
      test:  next `test_window` seasons
    Then advance `train_end` by `step` seasons and repeat until we run out.
    """
    seasons = seasons_sorted(labels)
    splits = []
    # index boundaries in the seasons list
    for train_end_idx in range(min_train - 1, len(seasons) - (val_window + test_window)):
        train = seasons[: train_end_idx + 1]
        val   = seasons[train_end_idx + 1 : train_end_idx + 1 + val_window]
        test  = seasons[train_end_idx + 1 + val_window : train_end_idx + 1 + val_window + test_window]
        splits.append({"train": train, "val": val, "test": test})
        train_end_idx += (step - 1)  # loop will add +1
    return splits

def masks_for(df: pd.DataFrame, split: Dict[str, List[int]]) -> Dict[str, pd.Series]:
    """
    Return boolean masks for df rows that belong to train/val/test seasons.
    """
    return {
        "train": df["season"].isin(split["train"]),
        "val":   df["season"].isin(split["val"]),
        "test":  df["season"].isin(split["test"]),
    }

def assemble_dataset(
    features_t: pd.DataFrame,
    labels_t_to_t1: pd.DataFrame,
    seasons: List[int],
    label_col: str,
    drop_future_cols: bool = True,
    id_cols: Tuple[str, str] = ("player", "season"),
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Join features@t with labels (t→t+1) on (player, season), filter by seasons list,
    and return (X, y, meta).
    - X: numeric features (drops the id cols and label_col)
    - y: the chosen label column (Series)
    - meta: useful columns for later (player, season, maybe team/pos)
    """
    # 1) seasons filter on labels (they define which (player,season) rows we train on)
    labels = subset_by_seasons(labels_t_to_t1, seasons)

    # 2) merge features@t
    df = labels.merge(features_t, on=list(id_cols), how="left", suffixes=("", "_feat"))

    # 3) (optional) drop any *_t1 columns that might have slipped in
    if drop_future_cols:
        future_cols = [c for c in df.columns if c.endswith("_t1")]
        df = df.drop(columns=future_cols)

    # 4) build y and meta
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found after merge.")
    y = df[label_col].astype(int)

    meta_cols = [c for c in ["player", "season", "role", "team_t", "gp_t"] if c in df.columns]
    meta = df[meta_cols].copy()

    # 5) build X (drop ids + labels + obvious non-features)
    drop_cols = set(meta_cols + [label_col])
    # also drop delta columns and label columns if they were carried
    drop_cols |= {c for c in df.columns if c.startswith("d_") or c.startswith("y_")}
    # keep only numeric feature columns
    X = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=["number"]).copy()

    return X, y, meta
# NHL Breakout Candidates 2025-26

---

## Data Availability Note
The raw NHL datasets used in this project were too large to upload directly to GitHub.  
However, all data can be reproduced by running the scripts provided in the `data/scrape/` directory.

---

## Overview

This project implements a **supervised machine-learning framework** to identify potential NHL breakout players for the **2025–26 season**, using **17 prior seasons of NHL data** from the websites hockey-reference.com and moneypuck.com.

Rather than predicting raw point totals, the model is designed to **classify whether a player is likely to meaningfully improve in the following season**, based on historical progression patterns observed in comparable players.

A particular emphasis is placed on **U24 skaters**, where development curves are steepest and breakout outcomes are most actionable. Older players with prior peak seasons were often flagged due to historical production alone, which introduced noise. Restricting attention to younger players significantly improves interpretability.

---

## Problem Formulation

### Breakout Definition

A **breakout** is defined **relatively, not absolutely**.

Using paired seasons *(t → t+1)*:

- Players are labeled as breakouts based on **future-season deltas** in key performance statistics
- Breakout thresholds are defined using **quantile-based cutoffs** learned from historical data
- Labels are **role-specific** (forwards, defensemen, goalies)

This approach avoids arbitrary thresholds and ensures labels are **data-driven and era-adjusted**.

---

## Label Construction (labels.py)

### Skaters

Breakout labels for skaters are assigned based on improvements in:

- Points, goals, assists
- Shots and per-game production rates
- Physical and defensive contributions (especially for defensemen)

**Separate label logic** was used to have offensive and defensive breakouts, allowing non-offensive breakout signals (e.g., hits, blocks) to qualify.

Quantile thresholds can be selectively tightened to control label strictness.

---

### Goalies

Goalie breakouts are labeled independently using changes in:

- Save percentage
- Goals against
- Shots faced
- Games played

Due to the higher variance in goalie performance, goalie predictions are interpreted more conservatively. Many identified candidates are strong backups or previously effective starters, where the model extrapolates improved performance under increased opportunity (which is not particularly likely).

---

## Feature Engineering (features.py)

Features are intentionally **minimalist and signal-driven**, including:

- Season-normalized z-scores
- Lagged performance metrics
- Contextual normalization across seasons
- Optional age-based restrictions

This design prioritizes **generalization and interpretability** over brute-force feature expansion.

---

## Modeling Approach (model.py, model_goalies.py)

- Supervised classification using:
  - Logistic Regression
  - Gradient Boosted Trees
  - XGBoost (where applicable)
- Class imbalance handled via **sample weighting**
- Probability calibration using **`CalibratedClassifierCV`**
- Evaluation metrics:
  - ROC-AUC
  - Average Precision
- SHAP support for model interpretability

Offensive statistics for skaters, defensive statistics for skaters, and goalies are trained and evaluated independently.

---

## Prediction & Ranking (predict.py)

- Models output calibrated **breakout probabilities**
- Players are ranked by predicted probability
- **Age filtering is applied post-prediction**, not during training

This allows:
- clean separation between learning and interpretation
- flexible ranking views (e.g., U24-only results)

---

## Results (2025–26 Projections)

### Top 15 U24 Offensive Breakout Candidates

| Rank | Player |
|-----:|----------------|
| 1 | Connor Bedard |
| 2 | Jake Sanderson |
| 3 | Andrei Svechnikov |
| 4 | JJ Peterka |
| 5 | Anton Lundell |
| 6 | Matt Boldy |
| 7 | Kirill Marchenko |
| 8 | Dylan Cozens |
| 9 | Mason McTavish |
| 10 | Dylan Guenther |
| 11 | Lucas Raymond |
| 12 | Pavel Dorofeyev |
| 13 | Jiri Kulich |
| 14 | Cutter Gauthier |
| 15 | Thomas Harley |

---

### Top 15 U24 Defensive Breakout Candidates

| Rank | Player |
|-----:|-----------------------|
| 1 | J.J. Moser |
| 2 | Rasmus Dahlin |
| 3 | Cutter Gauthier |
| 4 | Louis Crevier |
| 5 | Mavrik Bourque |
| 6 | Isaiah George |
| 7 | Jiri Kulich |
| 8 | Jacob Bernard-Docker |
| 9 | Denton Mateychuk |
| 10 | Albert Johansson |
| 11 | Nils Lundkvist |
| 12 | Ty Dellandrea |
| 13 | Jackson Blake |
| 14 | Jayden Struble |
| 15 | Kaiden Guhle |

---

### Top 10 Goalie Breakout Candidates

| Rank | Player |
|-----:|------------------|
| 1 | Frederik Andersen |
| 2 | Thatcher Demko |
| 3 | David Rittich |
| 4 | Joel Hofer |
| 5 | Eric Comrie |
| 6 | Philipp Grubauer |
| 7 | Charlie Lindgren |
| 8 | Ilya Samsonov |
| 9 | Samuel Ersson |
| 10 | Jonathan Quick |

---

## Repository Structure

```
NHL-Breakout-Candidates-2025-26/
├── src/            # Feature engineering, modeling, evaluation
├── data/
│   └── scrape/     # Data acquisition scripts
├── out/            # Model outputs and rankings
├── README.md
└── .gitignore
```

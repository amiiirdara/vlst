"""Quote the hosted-TabPFN token cost of ``baseline_plus_tabpfn.ipynb``.

Only that notebook calls the hosted API: the SHAP and FS/PDP interpretability
notebooks run local TabPFN (``SHAP_BACKEND="local"``, ``PDP_USE_CLIENT=False``)
and spend no tokens. The hosted arms are "TabPFN thinking v3" and
"TabPFN thinking v3.5"; each runs 5 outer x (4 inner + 1 outer) = 25 thinking
fits, each followed by one predict on the held-out fold.

Prices live on the Prior Labs server, so this asks ``estimate_cost`` for each
call shape. It sends only row/column counts and consumes no quota. A thinking
fit quote is the effort's maximum (reserved up front); the settled charge is
the measured GPU time, so the real spend is at or below the fit total.

Needs ``TABPFN_TOKEN`` in the environment (or an interactive login) and
``tabpfn-client==0.6.0``.

    python code/modeling/tools/estimate_tabpfn_tokens.py
"""

from __future__ import annotations

import numpy as np
from tabpfn_client import estimate_cost, get_api_usage

N_ROWS = 5185
# 85 raw columns minus NO., Name, TSSI, WBC and the target; R1 also drops the
# four follow-up medication columns.
N_COLUMNS = {"r1_postpci": 76, "primary": 80}
OUTER_SPLITS = 5
INNER_SPLITS = 4
THINKING_VERSIONS = ("v3", "v3.5")
THINKING_EFFORT = "high"
DAILY_LIMIT = 5_000_000


def _fold_rows(n_rows, n_splits):
    test = n_rows // n_splits
    return n_rows - test, test


def _quote(train_rows, test_rows, columns, version, operation):
    X_train = np.empty((train_rows, columns))
    X_test = None if operation == "thinking_fit" else np.empty((test_rows, columns))
    return estimate_cost(
        X_train,
        X_test,
        model_version=version,
        operation=operation,
        thinking_effort=THINKING_EFFORT if operation == "thinking_fit" else None,
    ).estimated_cost


def main():
    outer_train, outer_val = _fold_rows(N_ROWS, OUTER_SPLITS)
    inner_train, inner_cal = _fold_rows(outer_train, INNER_SPLITS)
    calls_per_arm = [
        (inner_train, inner_cal, OUTER_SPLITS * INNER_SPLITS),
        (outer_train, outer_val, OUTER_SPLITS),
    ]
    for tag, columns in N_COLUMNS.items():
        run_total = 0
        print(f"== {tag} ({columns} columns)")
        for version in THINKING_VERSIONS:
            arm_total = 0
            for train_rows, test_rows, n_calls in calls_per_arm:
                fit = _quote(train_rows, test_rows, columns, version, "thinking_fit")
                predict = _quote(train_rows, test_rows, columns, version, "thinking_predict")
                arm_total += n_calls * (fit + predict)
                print(
                    f"  thinking {version}: {n_calls} x (fit {fit:,} + predict {predict:,})"
                    f" on {train_rows}->{test_rows} rows"
                )
            print(f"  thinking {version} arm: {arm_total:,} tokens")
            run_total += arm_total
        print(
            f"  run total: {run_total:,} tokens"
            f" = {run_total / DAILY_LIMIT:.1f}x the {DAILY_LIMIT:,} daily limit\n"
        )
    print(get_api_usage())


if __name__ == "__main__":
    main()

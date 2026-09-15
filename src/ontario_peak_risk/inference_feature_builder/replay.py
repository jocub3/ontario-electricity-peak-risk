"""Historical replay validation against original model feature builders."""

from __future__ import annotations

import pandas as pd

from src.ontario_peak_risk.models.forecasting.random_forest import (
    features as rf_features_module,
)
from src.ontario_peak_risk.models.peak_risk.xgboost import (
    features as xgb_features_module,
)

from .builder import (
    build_feature_matrix,
    expected_features,
    load_model_contracts,
    model_feature_frames,
)


def historical_replay_comparison(
    *,
    feature_dataset: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    fsas: list[str],
    artifacts_dir,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare operational IFB features with the original model feature builders."""

    forecast_origin = pd.Timestamp(forecast_origin)
    timestamp_col = config["columns"]["timestamp"]

    # Historical replay must remain inside the original historical feature dataset.
    dataset_max_timestamp = pd.to_datetime(feature_dataset[timestamp_col]).max()
    if forecast_origin > dataset_max_timestamp:
        raise ValueError(
            "Historical replay must use a forecast_origin contained in the "
            "original historical feature dataset. "
            f"Requested origin: {forecast_origin}; "
            f"dataset maximum: {dataset_max_timestamp}. "
            "Use IFB_09 for post-holdout operational simulation."
        )

    contracts = load_model_contracts(artifacts_dir)

    # Build generic IFB features from the original historical dataset.
    generated = build_feature_matrix(
        feature_dataset,
        pd.DataFrame(),
        forecast_origin,
        fsas,
        config=config,
    )

    # Convert the generic matrix into task-specific model frames.
    # This preserves the exact frozen RF/XGB rolling semantics.
    task_frames = model_feature_frames(generated, contracts)

    origins = pd.DataFrame(
        {
            "fsa": fsas,
            "forecast_origin": [forecast_origin] * len(fsas),
        }
    )

    rows: list[dict] = []
    detail: list[dict] = []

    for task, module in [
        ("rf", rf_features_module),
        ("xgb", xgb_features_module),
    ]:
        features = expected_features(contracts[task])

        # `fsa` is itself a frozen categorical feature.  Do not select it twice.
        comparison_columns = list(
            dict.fromkeys(["fsa", "forecast_origin", *features])
        )

        for horizon in range(1, 25):
            reference = module.build_horizon_features(
                origins,
                feature_dataset,
                horizon=horizon,
            )

            missing_reference = [
                col for col in comparison_columns if col not in reference.columns
            ]
            if missing_reference:
                raise ValueError(
                    f"{task} reference builder is missing columns: "
                    f"{missing_reference}"
                )

            reference = (
                reference.loc[:, comparison_columns]
                .sort_values("fsa")
                .reset_index(drop=True)
            )

            candidate_source = task_frames[task][horizon]
            missing_candidate = [
                col for col in comparison_columns
                if col not in candidate_source.columns
            ]
            if missing_candidate:
                raise ValueError(
                    f"IFB candidate frame for {task} h+{horizon} is missing "
                    f"columns: {missing_candidate}"
                )

            candidate = (
                candidate_source.loc[:, comparison_columns]
                .sort_values("fsa")
                .reset_index(drop=True)
            )

            if len(reference) != len(candidate):
                raise ValueError(
                    f"Row-count mismatch for {task} h+{horizon}: "
                    f"reference={len(reference)}, candidate={len(candidate)}"
                )

            for feature in features:
                left = reference[feature]
                right = candidate[feature]

                if pd.api.types.is_numeric_dtype(left):
                    left_n = pd.to_numeric(left, errors="coerce")
                    right_n = pd.to_numeric(right, errors="coerce")

                    diff = (left_n - right_n).abs()
                    match = (
                        diff.fillna(0).le(1e-9)
                        & left_n.isna().eq(right_n.isna())
                    )
                    max_diff = diff.max()
                else:
                    match = (
                        left.astype("string").fillna("<NA>")
                        .eq(right.astype("string").fillna("<NA>"))
                    )
                    max_diff = pd.NA

                rows.append(
                    {
                        "model": task,
                        "horizon": horizon,
                        "feature": feature,
                        "rows_compared": len(match),
                        "matching_rows": int(match.sum()),
                        "match_rate_pct": float(match.mean() * 100),
                        "max_abs_difference": max_diff,
                        "status": "PASS" if bool(match.all()) else "FAIL",
                    }
                )

                for idx in match.index[~match][:10]:
                    detail.append(
                        {
                            "model": task,
                            "horizon": horizon,
                            "fsa": candidate.loc[idx, "fsa"],
                            "feature": feature,
                            "reference_value": left.loc[idx],
                            "ifb_value": right.loc[idx],
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(detail)

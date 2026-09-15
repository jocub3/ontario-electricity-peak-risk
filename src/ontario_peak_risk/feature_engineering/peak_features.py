"""Peak-Risk feature policy.

No final Peak-Risk target or threshold-dependent predictor is created in the static dataset. 
Such variables must be computed using training data only inside each temporal validation window.
"""

from __future__ import annotations

import pandas as pd


def peak_feature_design() -> pd.DataFrame:
    """Document Peak-related candidates that are deferred to model-time."""
    return pd.DataFrame(
        [
            {
                "candidate_feature": "previous_day_peak",
                "status": "deferred_to_training_window",
                "reason": "Requires a Peak threshold fitted only on training data.",
            },
            {
                "candidate_feature": "previous_week_peak",
                "status": "deferred_to_training_window",
                "reason": "Requires a Peak threshold fitted only on training data.",
            },
            {
                "candidate_feature": "hours_since_previous_peak",
                "status": "deferred_to_training_window",
                "reason": "Requires historical Peak labels derived from a training-only threshold.",
            },
        ]
    )

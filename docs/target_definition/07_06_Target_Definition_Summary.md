# Target Definition Summary

## Forecasting target

The Forecasting task uses a 24-hour target matrix (`target_h01` through `target_h24`) aligned by FSA and hourly forecast origin.

### Horizon coverage

| target     |   available_count |   missing_count |   coverage_pct |
|:-----------|------------------:|----------------:|---------------:|
| target_h01 |            262938 |               6 |        99.9977 |
| target_h02 |            262932 |              12 |        99.9954 |
| target_h03 |            262926 |              18 |        99.9932 |
| target_h04 |            262920 |              24 |        99.9909 |
| target_h05 |            262914 |              30 |        99.9886 |
| target_h06 |            262908 |              36 |        99.9863 |
| target_h07 |            262902 |              42 |        99.984  |
| target_h08 |            262896 |              48 |        99.9817 |
| target_h09 |            262890 |              54 |        99.9795 |
| target_h10 |            262884 |              60 |        99.9772 |
| target_h11 |            262878 |              66 |        99.9749 |
| target_h12 |            262872 |              72 |        99.9726 |
| target_h13 |            262866 |              78 |        99.9703 |
| target_h14 |            262860 |              84 |        99.9681 |
| target_h15 |            262854 |              90 |        99.9658 |
| target_h16 |            262848 |              96 |        99.9635 |
| target_h17 |            262842 |             102 |        99.9612 |
| target_h18 |            262836 |             108 |        99.9589 |
| target_h19 |            262830 |             114 |        99.9566 |
| target_h20 |            262824 |             120 |        99.9544 |
| target_h21 |            262818 |             126 |        99.9521 |
| target_h22 |            262812 |             132 |        99.9498 |
| target_h23 |            262806 |             138 |        99.9475 |
| target_h24 |            262800 |             144 |        99.9452 |

## Peak-Risk target

Peak-Risk thresholds are training-derived and are not calculated globally from the full dataset.

Walk-forward diagnostic labels are used only to confirm target behavior without future-year leakage.

### Diagnostic Peak distribution

|   observations |   peak_hours |   peak_rate_pct |
|---------------:|-------------:|----------------:|
|         210384 |        13784 |         6.55183 |

## Forecasting validation

| check                   |   violations | status   | description                                                                           |
|:------------------------|-------------:|:---------|:--------------------------------------------------------------------------------------|
| target_origin_row_count |            0 | PASS     | Target matrix must preserve one forecast origin per feature row.                      |
| unique_forecast_origin  |            0 | PASS     | FSA + forecast origin must remain unique.                                             |
| alignment_h01           |            0 | PASS     | h+1 target must equal observed consumption exactly 1 hour(s) after forecast origin.   |
| alignment_h02           |            0 | PASS     | h+2 target must equal observed consumption exactly 2 hour(s) after forecast origin.   |
| alignment_h03           |            0 | PASS     | h+3 target must equal observed consumption exactly 3 hour(s) after forecast origin.   |
| alignment_h04           |            0 | PASS     | h+4 target must equal observed consumption exactly 4 hour(s) after forecast origin.   |
| alignment_h05           |            0 | PASS     | h+5 target must equal observed consumption exactly 5 hour(s) after forecast origin.   |
| alignment_h06           |            0 | PASS     | h+6 target must equal observed consumption exactly 6 hour(s) after forecast origin.   |
| alignment_h07           |            0 | PASS     | h+7 target must equal observed consumption exactly 7 hour(s) after forecast origin.   |
| alignment_h08           |            0 | PASS     | h+8 target must equal observed consumption exactly 8 hour(s) after forecast origin.   |
| alignment_h09           |            0 | PASS     | h+9 target must equal observed consumption exactly 9 hour(s) after forecast origin.   |
| alignment_h10           |            0 | PASS     | h+10 target must equal observed consumption exactly 10 hour(s) after forecast origin. |
| alignment_h11           |            0 | PASS     | h+11 target must equal observed consumption exactly 11 hour(s) after forecast origin. |
| alignment_h12           |            0 | PASS     | h+12 target must equal observed consumption exactly 12 hour(s) after forecast origin. |
| alignment_h13           |            0 | PASS     | h+13 target must equal observed consumption exactly 13 hour(s) after forecast origin. |
| alignment_h14           |            0 | PASS     | h+14 target must equal observed consumption exactly 14 hour(s) after forecast origin. |
| alignment_h15           |            0 | PASS     | h+15 target must equal observed consumption exactly 15 hour(s) after forecast origin. |
| alignment_h16           |            0 | PASS     | h+16 target must equal observed consumption exactly 16 hour(s) after forecast origin. |
| alignment_h17           |            0 | PASS     | h+17 target must equal observed consumption exactly 17 hour(s) after forecast origin. |
| alignment_h18           |            0 | PASS     | h+18 target must equal observed consumption exactly 18 hour(s) after forecast origin. |
| alignment_h19           |            0 | PASS     | h+19 target must equal observed consumption exactly 19 hour(s) after forecast origin. |
| alignment_h20           |            0 | PASS     | h+20 target must equal observed consumption exactly 20 hour(s) after forecast origin. |
| alignment_h21           |            0 | PASS     | h+21 target must equal observed consumption exactly 21 hour(s) after forecast origin. |
| alignment_h22           |            0 | PASS     | h+22 target must equal observed consumption exactly 22 hour(s) after forecast origin. |
| alignment_h23           |            0 | PASS     | h+23 target must equal observed consumption exactly 23 hour(s) after forecast origin. |
| alignment_h24           |            0 | PASS     | h+24 target must equal observed consumption exactly 24 hour(s) after forecast origin. |

## Peak-Risk validation

| check                                   |   violations | status   | description                                                                               |
|:----------------------------------------|-------------:|:---------|:------------------------------------------------------------------------------------------|
| unique_peak_label_key                   |            0 | PASS     | Diagnostic Peak labels must be unique by FSA + timestamp.                                 |
| peak_label_binary                       |            0 | PASS     | Peak-Risk labels must be binary when a threshold is available.                            |
| peak_threshold_availability_consistency |            0 | PASS     | Available thresholds must produce a diagnostic binary label.                              |
| walk_forward_threshold_no_future_year   |            0 | PASS     | Each diagnostic threshold must be fitted using years strictly before its evaluation year. |

## Modeling handoff

The next modeling stages must preserve the following rules:

1. all comparisons must use the same 24-hour target definition;
2. chronological train/validation/test windows must be used;
3. Peak thresholds must be fitted independently inside each training window;
4. validation/test Peak labels must use thresholds learned from training only;
5. forecast-horizon feature availability must be respected;
6. diagnostic Peak labels generated in the Target Definition Phase must not be used as globally precomputed final model labels.

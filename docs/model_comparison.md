# Model comparison

Data: Open-Meteo (CAMS air quality, ERA5 weather), 2024-01-01 to 2026-11-13.

All errors are MAE in ug/m3 unless noted. Skill is 1 minus the model MAE divided by the persistence MAE.

## Walk-forward validation: MAE by horizon (all locations)

Monthly expanding-window folds, forecast origins every 6 hours.

| horizon | persistence | seasonal_naive | xgboost |
|---|---|---|---|
| 1.0 | 17.677 | 34.179 | 6.597 |
| 3.0 | 26.208 | 33.754 | 13.856 |
| 6.0 | 35.392 | 34.989 | 21.55 |
| 12.0 | 43.139 | 35.017 | 24.101 |
| 24.0 | 35.06 | 35.06 | 25.331 |
| 36.0 | 47.682 | 40.892 | 27.133 |
| 48.0 | 40.962 | 40.962 | 28.058 |

## Skill versus persistence, walk-forward



| model | horizon | mae | rmse | skill_vs_persistence |
|---|---|---|---|---|
| seasonal_naive | 1 | 34.179 | 54.157 | -0.934 |
| seasonal_naive | 3 | 33.754 | 51.605 | -0.288 |
| seasonal_naive | 6 | 34.989 | 55.904 | 0.011 |
| seasonal_naive | 12 | 35.017 | 55.93 | 0.188 |
| seasonal_naive | 24 | 35.06 | 55.966 | 0.0 |
| seasonal_naive | 36 | 40.892 | 62.135 | 0.142 |
| seasonal_naive | 48 | 40.962 | 62.186 | 0.0 |
| xgboost | 1 | 6.597 | 14.804 | 0.627 |
| xgboost | 3 | 13.856 | 23.119 | 0.471 |
| xgboost | 6 | 21.55 | 35.253 | 0.391 |
| xgboost | 12 | 24.101 | 39.28 | 0.441 |
| xgboost | 24 | 25.331 | 41.075 | 0.277 |
| xgboost | 36 | 27.133 | 42.621 | 0.431 |
| xgboost | 48 | 28.058 | 43.88 | 0.315 |

## Walk-forward by season



| model | horizon | season | n | mae | rmse | skill_vs_persistence |
|---|---|---|---|---|---|---|
| persistence | 1 | other | 21400 | 22.007 | 44.73 | 0.0 |
| persistence | 1 | winter | 15100 | 11.54 | 15.811 | 0.0 |
| persistence | 3 | other | 21400 | 28.598 | 49.017 | 0.0 |
| persistence | 3 | winter | 15100 | 22.819 | 31.093 | 0.0 |
| persistence | 6 | other | 21400 | 39.138 | 61.003 | 0.0 |
| persistence | 6 | winter | 15100 | 30.082 | 41.945 | 0.0 |
| persistence | 12 | other | 21400 | 44.221 | 67.161 | 0.0 |
| persistence | 12 | winter | 15100 | 41.604 | 55.329 | 0.0 |
| persistence | 24 | other | 21400 | 39.805 | 63.856 | 0.0 |
| persistence | 24 | winter | 15100 | 28.334 | 42.337 | 0.0 |
| persistence | 36 | other | 21400 | 48.073 | 71.509 | 0.0 |
| persistence | 36 | winter | 15100 | 47.126 | 63.184 | 0.0 |
| persistence | 48 | other | 21400 | 44.06 | 68.795 | 0.0 |
| persistence | 48 | winter | 15100 | 36.572 | 51.383 | 0.0 |
| seasonal_naive | 1 | other | 21400 | 38.297 | 61.23 | -0.74 |
| seasonal_naive | 1 | winter | 15100 | 28.343 | 42.145 | -1.456 |
| seasonal_naive | 3 | other | 21400 | 36.55 | 56.286 | -0.278 |
| seasonal_naive | 3 | winter | 15100 | 29.792 | 44.129 | -0.306 |
| seasonal_naive | 6 | other | 21400 | 39.751 | 63.801 | -0.016 |
| seasonal_naive | 6 | winter | 15100 | 28.239 | 42.257 | 0.061 |
| seasonal_naive | 12 | other | 21400 | 39.773 | 63.83 | 0.101 |
| seasonal_naive | 12 | winter | 15100 | 28.277 | 42.277 | 0.32 |
| seasonal_naive | 24 | other | 21400 | 39.805 | 63.856 | 0.0 |
| seasonal_naive | 24 | winter | 15100 | 28.334 | 42.337 | 0.0 |
| seasonal_naive | 36 | other | 21400 | 44.023 | 68.766 | 0.084 |
| seasonal_naive | 36 | winter | 15100 | 36.454 | 51.291 | 0.226 |
| seasonal_naive | 48 | other | 21400 | 44.06 | 68.795 | 0.0 |
| seasonal_naive | 48 | winter | 15100 | 36.572 | 51.383 | 0.0 |
| xgboost | 1 | other | 21400 | 7.791 | 18.438 | 0.646 |
| xgboost | 1 | winter | 15100 | 4.904 | 6.925 | 0.575 |
| xgboost | 3 | other | 21400 | 15.693 | 26.863 | 0.451 |
| xgboost | 3 | winter | 15100 | 11.253 | 16.408 | 0.507 |
| xgboost | 6 | other | 21400 | 25.057 | 41.767 | 0.36 |
| xgboost | 6 | winter | 15100 | 16.58 | 23.061 | 0.449 |
| xgboost | 12 | other | 21400 | 27.527 | 46.026 | 0.378 |
| xgboost | 12 | winter | 15100 | 19.245 | 26.97 | 0.537 |
| xgboost | 24 | other | 21400 | 28.645 | 47.604 | 0.28 |
| xgboost | 24 | winter | 15100 | 20.635 | 29.44 | 0.272 |
| xgboost | 36 | other | 21400 | 29.948 | 48.423 | 0.377 |
| xgboost | 36 | winter | 15100 | 23.144 | 32.68 | 0.509 |
| xgboost | 48 | other | 21400 | 30.783 | 49.357 | 0.301 |
| xgboost | 48 | winter | 15100 | 24.196 | 34.667 | 0.338 |

## AQI category accuracy, walk-forward

Category of the 24-hour mean PM2.5 at the target time.

| model | horizon | category_accuracy | within_one | poor_or_worse_recall | n |
|---|---|---|---|---|---|
| persistence | 1 | 0.984 | 1.0 | 0.996 | 7300 |
| persistence | 3 | 0.945 | 1.0 | 0.982 | 7300 |
| persistence | 6 | 0.872 | 1.0 | 0.955 | 7300 |
| persistence | 12 | 0.704 | 0.989 | 0.887 | 7300 |
| persistence | 24 | 0.47 | 0.909 | 0.773 | 7300 |
| persistence | 36 | 0.424 | 0.852 | 0.728 | 7300 |
| persistence | 48 | 0.402 | 0.821 | 0.691 | 7300 |
| seasonal_naive | 1 | 0.964 | 1.0 | 0.986 | 7300 |
| seasonal_naive | 3 | 0.899 | 1.0 | 0.968 | 7300 |
| seasonal_naive | 6 | 0.82 | 0.999 | 0.941 | 7300 |
| seasonal_naive | 12 | 0.693 | 0.981 | 0.883 | 7300 |
| seasonal_naive | 24 | 0.538 | 0.91 | 0.794 | 7300 |
| seasonal_naive | 36 | 0.498 | 0.877 | 0.744 | 7300 |
| seasonal_naive | 48 | 0.458 | 0.849 | 0.698 | 7300 |
| xgboost | 1 | 0.993 | 1.0 | 0.997 | 7300 |
| xgboost | 3 | 0.973 | 1.0 | 0.99 | 7300 |
| xgboost | 6 | 0.929 | 1.0 | 0.964 | 7300 |
| xgboost | 12 | 0.822 | 0.998 | 0.903 | 7300 |
| xgboost | 24 | 0.641 | 0.973 | 0.803 | 7300 |
| xgboost | 36 | 0.561 | 0.948 | 0.739 | 7300 |
| xgboost | 48 | 0.518 | 0.939 | 0.701 | 7300 |

## Prediction interval (80%) coverage, walk-forward

Target coverage is 0.80.

| horizon | variant | coverage | mean_width |
|---|---|---|---|
| 1 | raw quantile | 0.683 | 15.174 |
| 3 | raw quantile | 0.683 | 34.286 |
| 6 | raw quantile | 0.65 | 47.454 |
| 12 | raw quantile | 0.65 | 52.707 |
| 24 | raw quantile | 0.652 | 55.107 |
| 36 | raw quantile | 0.628 | 56.306 |
| 48 | raw quantile | 0.625 | 56.045 |

## Final holdout: MAE by horizon

Scored once, after model selection.

| horizon | persistence | seasonal_naive | xgboost |
|---|---|---|---|
| 1.0 | 10.969 | 26.486 | 4.741 |
| 3.0 | 17.204 | 25.709 | 9.918 |
| 6.0 | 25.356 | 26.363 | 15.756 |
| 9.0 | 29.937 | 25.712 | 16.826 |
| 12.0 | 34.065 | 26.257 | 18.724 |
| 15.0 | 33.917 | 25.65 | 19.011 |
| 18.0 | 31.172 | 26.292 | 20.048 |
| 21.0 | 29.373 | 25.608 | 19.77 |
| 24.0 | 26.128 | 26.128 | 19.7 |
| 27.0 | 28.953 | 30.55 | 20.759 |
| 30.0 | 31.397 | 30.752 | 21.214 |
| 33.0 | 34.789 | 30.455 | 21.382 |
| 36.0 | 37.037 | 30.539 | 22.255 |
| 39.0 | 36.857 | 30.405 | 22.435 |
| 42.0 | 33.948 | 30.516 | 22.898 |
| 45.0 | 33.087 | 30.341 | 22.564 |
| 48.0 | 30.321 | 30.321 | 22.648 |

## Final holdout: AQI category accuracy



| model | horizon | category_accuracy | within_one | poor_or_worse_recall | n |
|---|---|---|---|---|---|
| persistence | 1 | 0.993 | 1.0 | 1.0 | 1440 |
| persistence | 3 | 0.965 | 1.0 | 0.976 | 1440 |
| persistence | 6 | 0.918 | 1.0 | 0.913 | 1440 |
| persistence | 9 | 0.819 | 1.0 | 0.855 | 1440 |
| persistence | 12 | 0.746 | 1.0 | 0.777 | 1440 |
| persistence | 15 | 0.625 | 0.991 | 0.731 | 1440 |
| persistence | 18 | 0.562 | 0.98 | 0.666 | 1440 |
| persistence | 21 | 0.494 | 0.956 | 0.633 | 1440 |
| persistence | 24 | 0.469 | 0.916 | 0.603 | 1440 |
| persistence | 27 | 0.444 | 0.894 | 0.587 | 1440 |
| persistence | 30 | 0.423 | 0.89 | 0.578 | 1440 |
| persistence | 33 | 0.404 | 0.888 | 0.577 | 1440 |
| persistence | 36 | 0.388 | 0.885 | 0.574 | 1440 |
| persistence | 39 | 0.39 | 0.872 | 0.577 | 1440 |
| persistence | 42 | 0.367 | 0.865 | 0.558 | 1440 |
| persistence | 45 | 0.374 | 0.849 | 0.558 | 1440 |
| persistence | 48 | 0.361 | 0.836 | 0.542 | 1440 |
| seasonal_naive | 1 | 0.967 | 1.0 | 0.989 | 1440 |
| seasonal_naive | 3 | 0.899 | 1.0 | 0.945 | 1440 |
| seasonal_naive | 6 | 0.839 | 1.0 | 0.89 | 1440 |
| seasonal_naive | 9 | 0.76 | 1.0 | 0.864 | 1440 |
| seasonal_naive | 12 | 0.706 | 0.996 | 0.818 | 1440 |
| seasonal_naive | 15 | 0.646 | 0.989 | 0.779 | 1440 |
| seasonal_naive | 18 | 0.615 | 0.978 | 0.724 | 1440 |
| seasonal_naive | 21 | 0.576 | 0.962 | 0.698 | 1440 |
| seasonal_naive | 24 | 0.54 | 0.951 | 0.655 | 1440 |
| seasonal_naive | 27 | 0.526 | 0.941 | 0.662 | 1440 |
| seasonal_naive | 30 | 0.501 | 0.931 | 0.635 | 1440 |
| seasonal_naive | 33 | 0.486 | 0.917 | 0.623 | 1440 |
| seasonal_naive | 36 | 0.471 | 0.904 | 0.603 | 1440 |
| seasonal_naive | 39 | 0.452 | 0.894 | 0.592 | 1440 |
| seasonal_naive | 42 | 0.439 | 0.888 | 0.572 | 1440 |
| seasonal_naive | 45 | 0.417 | 0.891 | 0.569 | 1440 |
| seasonal_naive | 48 | 0.401 | 0.885 | 0.546 | 1440 |
| xgboost | 1 | 0.99 | 1.0 | 1.0 | 1440 |
| xgboost | 3 | 0.981 | 1.0 | 0.987 | 1440 |
| xgboost | 6 | 0.95 | 1.0 | 0.95 | 1440 |
| xgboost | 9 | 0.897 | 1.0 | 0.93 | 1440 |
| xgboost | 12 | 0.852 | 1.0 | 0.872 | 1440 |
| xgboost | 15 | 0.796 | 0.996 | 0.849 | 1440 |
| xgboost | 18 | 0.755 | 0.997 | 0.806 | 1440 |
| xgboost | 21 | 0.697 | 0.99 | 0.772 | 1440 |
| xgboost | 24 | 0.656 | 0.985 | 0.723 | 1440 |
| xgboost | 27 | 0.61 | 0.979 | 0.688 | 1440 |
| xgboost | 30 | 0.572 | 0.978 | 0.652 | 1440 |
| xgboost | 33 | 0.547 | 0.976 | 0.623 | 1440 |
| xgboost | 36 | 0.533 | 0.976 | 0.605 | 1440 |
| xgboost | 39 | 0.521 | 0.967 | 0.587 | 1440 |
| xgboost | 42 | 0.51 | 0.966 | 0.564 | 1440 |
| xgboost | 45 | 0.49 | 0.967 | 0.565 | 1440 |
| xgboost | 48 | 0.49 | 0.965 | 0.558 | 1440 |

## Final holdout: interval coverage



| horizon | variant | coverage | mean_width |
|---|---|---|---|
| 1 | raw quantile | 0.675 | 10.311 |
| 1 | recalibrated | 0.805 | 14.518 |
| 3 | raw quantile | 0.692 | 26.201 |
| 3 | recalibrated | 0.828 | 36.698 |
| 6 | raw quantile | 0.624 | 37.47 |
| 6 | recalibrated | 0.786 | 54.345 |
| 9 | raw quantile | 0.662 | 41.326 |
| 9 | recalibrated | 0.801 | 59.923 |
| 12 | raw quantile | 0.628 | 43.144 |
| 12 | recalibrated | 0.778 | 62.563 |
| 15 | raw quantile | 0.641 | 43.771 |
| 15 | recalibrated | 0.782 | 63.471 |
| 18 | raw quantile | 0.601 | 44.058 |
| 18 | recalibrated | 0.766 | 63.887 |
| 21 | raw quantile | 0.653 | 45.059 |
| 21 | recalibrated | 0.812 | 65.335 |
| 24 | raw quantile | 0.606 | 45.777 |
| 24 | recalibrated | 0.759 | 66.427 |
| 27 | raw quantile | 0.632 | 47.668 |
| 27 | recalibrated | 0.792 | 70.603 |
| 30 | raw quantile | 0.637 | 50.562 |
| 30 | recalibrated | 0.781 | 75.903 |
| 33 | raw quantile | 0.635 | 49.958 |
| 33 | recalibrated | 0.811 | 75.957 |
| 36 | raw quantile | 0.604 | 50.094 |
| 36 | recalibrated | 0.807 | 77.673 |
| 39 | raw quantile | 0.599 | 49.136 |
| 39 | recalibrated | 0.803 | 76.233 |
| 42 | raw quantile | 0.574 | 47.66 |
| 42 | recalibrated | 0.791 | 73.958 |
| 45 | raw quantile | 0.619 | 49.035 |
| 45 | recalibrated | 0.791 | 76.667 |
| 48 | raw quantile | 0.575 | 48.263 |
| 48 | recalibrated | 0.775 | 74.958 |

## SARIMAX and Prophet (one location, daily origins)

Evaluated on one location because both models are slow to refit.

| model | horizon | n | mae | rmse | skill_vs_persistence |
|---|---|---|---|---|---|
| persistence | 1 | 184 | 4.411 | 6.695 | 0.0 |
| persistence | 3 | 184 | 13.641 | 19.856 | 0.0 |
| persistence | 6 | 184 | 28.734 | 40.683 | 0.0 |
| persistence | 12 | 184 | 30.805 | 41.65 | 0.0 |
| persistence | 24 | 184 | 28.218 | 36.404 | 0.0 |
| persistence | 36 | 184 | 33.112 | 45.672 | 0.0 |
| persistence | 48 | 184 | 32.345 | 41.331 | 0.0 |
| sarimax | 1 | 184 | 8.066 | 11.934 | -0.828 |
| sarimax | 3 | 184 | 16.159 | 23.151 | -0.185 |
| sarimax | 6 | 184 | 25.0 | 36.786 | 0.13 |
| sarimax | 12 | 184 | 27.682 | 39.694 | 0.101 |
| sarimax | 24 | 184 | 31.344 | 40.485 | -0.111 |
| sarimax | 36 | 184 | 29.45 | 41.072 | 0.111 |
| sarimax | 48 | 184 | 32.562 | 41.924 | -0.007 |
| seasonal_naive | 1 | 184 | 28.604 | 37.374 | -5.484 |
| seasonal_naive | 3 | 184 | 27.333 | 38.929 | -1.004 |
| seasonal_naive | 6 | 184 | 26.044 | 40.813 | 0.094 |
| seasonal_naive | 12 | 184 | 24.086 | 37.442 | 0.218 |
| seasonal_naive | 24 | 184 | 28.218 | 36.404 | 0.0 |
| seasonal_naive | 36 | 184 | 29.548 | 44.56 | 0.108 |
| seasonal_naive | 48 | 184 | 32.345 | 41.331 | 0.0 |

## XGBoost ablations at 24 hours



| variant | mae_h24 | note |
|---|---|---|
| log target, with forecast weather | 24.765 | every second walk-forward fold (6 of 12) |
| raw target, with forecast weather | 24.771 | every second walk-forward fold (6 of 12) |
| log target, without forecast weather | 27.09 | every second walk-forward fold (6 of 12) |

## Top features by mean absolute SHAP, 24 h



| Unnamed: 0 | mean_abs_shap |
|---|---|
| pm25_now | 0.192 |
| fw_wind_mean | 0.062 |
| fw_rain_sum | 0.055 |
| pm25_mean3 | 0.038 |
| pm25_lag24 | 0.032 |
| now_hour_cos | 0.028 |
| pm25_lag168 | 0.027 |
| fw_wind_speed_10m | 0.025 |
| rain_sum24 | 0.023 |
| fw_relative_humidity_2m | 0.021 |

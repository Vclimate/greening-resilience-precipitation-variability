# Random Forest Support

`run_rf.py` supplies shared Random Forest utilities. The current repeated-refit and spatial-validation scripts use its:

- `iqr`: training-response IQR filtering;
- `sample`: reproducible stratified subsampling;
- `metrics`: classification metrics and the majority-class reference.

The module also retains its original RF/TPE/OOB routines for traceability. Its historical standalone defaults are not the primary QA01 analysis configuration. Use `code/train_RF_QA01_unified22_20261005.py` for the primary 30 repeated RF fits and `code/run_RF_QA01_spatial5_20261006.py` for spatial validation.

No XGBoost model or SHAP attribution is used by these final RF entry points. The directory name describes the actual supporting method; it is not an analysis version or a separate data source.

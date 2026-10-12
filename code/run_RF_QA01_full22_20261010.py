"""Full-22-endpoint RF sensitivity, preserving the approved first-19 analysis.

Only the response trend period and its significance-selected cohort change.
Predictor values, original masks, QA0/1 missingness period, tuning, seeds,
training-only VIF/IQR and exact per-tree OOB importance remain unchanged.
"""

import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
import time

import recompute_QA01_22periods_20261005 as source
from scipy.stats import linregress

rf, prep, np, pd, rasterio = source.rf, source.prep, source.np, source.pd, source.rasterio
ORIGINAL = source.OUT
OUT = ORIGINAL / "RF_full22_endpoints_2005_2026_20261010"
DATA = OUT / "data"
MODELS = OUT / "models"
YEARS = np.arange(2005, 2027)
FORBIDDEN = "05_\u968f\u673a\u68ee\u6797\u5f52\u56e0\u90e8\u5206"


def preflight():
    rng = np.random.default_rng(20261010)
    a = rng.normal(0, .15, (22, 8)) + .008 * np.arange(22)[:, None]
    a[::4, 1::2] = np.nan
    slope, p, count = prep.fit_pixels(a, YEARS)
    for j in range(a.shape[1]):
        good = np.isfinite(a[:, j])
        expected = linregress(YEARS[good], a[good, j])
        np.testing.assert_allclose([slope[j], p[j]], [expected.slope, expected.pvalue], rtol=1e-10, atol=1e-12)
        assert count[j] == good.sum()
    shifted = prep.fit_pixels(a, YEARS - 1)
    np.testing.assert_array_equal(slope, shifted[0])
    np.testing.assert_array_equal(p, shifted[1])
    return dict(OLS_matches_independent_linregress=True, endpoint_vs_reference_labels_same_slope=True)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    frozen = rf.read(ORIGINAL / "data/source_manifest.json")
    original_table = Path(frozen["table"])
    assert rf.digest(original_table) == frozen["table_sha256"]
    marker = DATA / "source_manifest.json"
    if marker.exists():
        saved = rf.read(marker)
        assert saved["status"] == "completed"
        assert saved["baseline_table_sha256"] == frozen["table_sha256"]
        assert saved["table_sha256"] == rf.digest(saved["table"])
        assert saved["response_sha256"] == rf.digest(saved["response_raster"])
        for path, sha in saved["input_hashes"].items():
            assert rf.digest(path) == sha, path
        return saved

    checks = preflight()
    manifest, support, lc, legacy_mdn, base = source.old.masks()
    names = [spec["variable"] for spec in manifest["predictors"]]
    assert names == frozen["candidates"] and len(names) == 22
    raw_manifest = Path(frozen["source_manifest"])
    assert rf.digest(raw_manifest) == frozen["source_manifest_sha256"]
    sources = manifest["predictors"] + [support[name] for name in
        ["land_cover", "stable_vegetation_mask", "MDN", "climate_zone"]]
    hashes = {}
    for spec in sources:
        assert FORBIDDEN not in spec["path"]
        path = str(spec["path"])
        if path not in hashes:
            hashes[path] = rf.digest(path)
        assert hashes[path] == spec["sha256"], spec["variable"]

    ids = np.flatnonzero(base)
    rows, cols = np.divmod(ids, 7200)
    frame = pd.DataFrame(dict(
        pixel_id=((rows + 200) * 7200 + cols).astype("int32"),
        longitude=-180 + (cols + .5) * .05,
        latitude=80 - (rows + .5) * .05,
        vegetation=np.where(lc.ravel()[ids] <= 5, "Forest", "Non-forest"),
        legacy_MDN=legacy_mdn.ravel()[ids],
    ))
    frame["climate_zone"] = prep.package.mat(support["climate_zone"]).ravel()[ids]
    for spec in manifest["predictors"]:
        name = spec["variable"]
        if name in ["kNDVI", "dkNDVI"]:
            continue
        values = prep.package.mat(spec).ravel()[ids].astype("float32")
        values[(~np.isfinite(values)) | (np.abs(values) > 1e30) | np.isin(values, [-9999, -32768])] = np.nan
        frame[name] = values

    green = ORIGINAL / "data/kNDVI/greenness_2000_2025.tif"
    hashes[str(green)] = rf.digest(green)
    with rasterio.open(green) as reader:
        assert reader.shape == prep.SHAPE and reader.transform.almost_equals(prep.GRID)
        g = reader.read([1, 2, 5], masked=True).filled(np.nan).reshape(3, -1)[:, ids]
    frame["kNDVI"], frame["dkNDVI"], frame["QA01_missing_fraction"] = g
    complete = np.isfinite(frame[names]).all(axis=1).to_numpy()
    missing_ok = np.isfinite(g[2]) & (g[2] >= 0) & (g[2] < .5)

    evidence = rf.read(ORIGINAL / "data/kNDVI/complete.json")
    assert evidence["status"] == "completed" and evidence["TAC_endpoints"] == YEARS.tolist()
    values = np.empty((22, len(ids)), dtype="float32")
    input_inventory = []
    for j, year in enumerate(YEARS):
        path = ORIGINAL / f"data/kNDVI/lag1_5year/kNDVI_TAC_lag1_5year_{year}_01_01_QA01.tif"
        digest = rf.digest(path)
        assert evidence["outputs"][str(path)]["sha256"] == digest
        hashes[str(path)] = digest
        with rasterio.open(path) as reader:
            assert reader.shape == prep.SHAPE
            assert reader.crs.to_epsg() == 4326 and reader.transform.almost_equals(prep.GRID)
            assert reader.tags()["window_end_exclusive"] == f"{year}-01-01"
            values[j] = reader.read(1, masked=True).filled(np.nan).ravel()[ids]
        input_inventory.append(dict(endpoint=int(year), reference_year=int(year - 1), path=str(path), sha256=digest))
    values[np.abs(values) > 1] = np.nan
    old_slope, old_p, old_n = prep.fit_pixels(values[:19], YEARS[:19])
    response19 = ORIGINAL / "data/kNDVI/TAC_trend_RF_endpoints_2005_2023.tif"
    hashes[str(response19)] = rf.digest(response19)
    with rasterio.open(response19) as reader:
        original19 = reader.read(masked=True).filled(np.nan).reshape(3, -1)[:, ids]
    for calculated, saved in zip([old_slope, old_p, old_n], original19):
        np.testing.assert_array_equal(calculated.astype("float32"), saved)
    checks["first19_response_raster_exactly_reproduced"] = True

    slope, pvalue, count = prep.fit_pixels(values, YEARS)
    del values
    response = DATA / "TAC_trend_RF_endpoints_2005_2026_QA01.tif"
    if response.exists():
        raise FileExistsError(response)
    map_values = np.full((3, *prep.SHAPE), -9999, dtype="float32")
    for band, arr in enumerate([slope, pvalue, count]):
        map_values[band].ravel()[ids] = np.where(np.isfinite(arr), arr, -9999)
    temporary = Path(str(response) + ".partial.tif")
    with rasterio.open(temporary, "w", **prep.profile(3)) as writer:
        writer.write(map_values)
        for band, label in enumerate(["delta_TAC", "OLS_pvalue", "valid_window_count"], 1):
            writer.set_band_description(band, label)
        writer.update_tags(first_endpoint="2005-01-01", last_endpoint="2026-01-01",
            reference_years="2004-2025", qa="SummaryQA in [0,1]", range_filter="-1 <= TAC <= 1",
            minimum_valid_windows="3", regression="OLS with intercept",
            significance="ordinary two-sided OLS; pixel-wise HAC not applied",
            boundary_effects="full record includes edge-affected residuals; not removed")
    temporary.replace(response)
    del map_values
    frame["y_continuous"], frame["tac_pvalue"], frame["valid_TAC_windows"] = slope.astype("float32"), pvalue.astype("float32"), count.astype("float32")
    frame["y"] = (slope >= 0).astype("int8")
    valid = np.isfinite(slope) & np.isfinite(pvalue)
    significant = valid & (pvalue.astype("float32") < .05)
    old_valid = np.isfinite(old_slope) & np.isfinite(old_p)
    old_sig = old_valid & (original19[1] < .05)
    keep = significant & complete & missing_ok
    old_keep = old_sig & complete & missing_ok
    old_table = pd.read_parquet(original_table)
    assert len(old_table) == old_keep.sum()
    original_positions = pd.Index(frame.pixel_id).get_indexer(old_table.pixel_id)
    assert (original_positions >= 0).all()
    assert np.all(old_keep[original_positions])
    for name in names + ["longitude", "latitude", "climate_zone", "QA01_missing_fraction"]:
        np.testing.assert_array_equal(frame[name].to_numpy()[original_positions], old_table[name].to_numpy())
    np.testing.assert_array_equal(old_slope[original_positions].astype("float32"), old_table.y_continuous)
    checks["baseline_cohort_and_all_X_exactly_reproduced"] = True

    counts, transitions = [], []
    weights = np.cos(np.deg2rad(frame.latitude.to_numpy()))
    for veg in ["Forest", "Non-forest"]:
        group = frame.vegetation.eq(veg).to_numpy()
        both = group & old_keep & keep
        changed = both & ((old_slope >= 0) != (slope >= 0))
        for period, selection, y, sig, n in [
            ("19_endpoints", old_keep, old_slope, old_sig, old_n),
            ("22_endpoints", keep, slope, significant, count),
        ]:
            chosen = group & selection
            counts.append(dict(period=period, vegetation=veg, original_base=int(group.sum()),
                valid_trend=int((group & (old_valid if period == "19_endpoints" else valid)).sum()),
                significant=int((group & sig).sum()), n_eligible=int(chosen.sum()),
                decline=int((chosen & (y >= 0)).sum()), increase=int((chosen & (y < 0)).sum()),
                decline_fraction=float((y[chosen] >= 0).mean()), zero_trends=int((chosen & (y == 0)).sum()),
                minimum_valid_TAC_windows=int(n[chosen].min()), maximum_valid_TAC_windows=int(n[chosen].max()),
                all_period_windows_valid=int((chosen & (n == (19 if period == "19_endpoints" else 22))).sum())))
        transitions.append(dict(vegetation=veg, common_eligible=int(both.sum()),
            only_19_eligible=int((group & old_keep & ~keep).sum()),
            only_22_eligible=int((group & keep & ~old_keep).sum()), label_changed=int(changed.sum()),
            label_changed_percent=100 * float(changed.sum()) / max(1, both.sum()),
            area_weighted_label_changed_percent=100 * float(weights[changed].sum()) / max(1e-20, weights[both].sum()),
            common_slope_correlation=float(np.corrcoef(old_slope[both], slope[both])[0, 1])))

    frame[["pixel_id", "vegetation", "QA01_missing_fraction", "y_continuous", "tac_pvalue", "valid_TAC_windows"]].to_parquet(DATA / "all_base_diagnostics_22.parquet", index=False)
    paired = frame[["pixel_id", "longitude", "latitude", "vegetation"]].copy()
    paired["slope19"], paired["p19"], paired["slope22"], paired["p22"] = old_slope, old_p, slope, pvalue
    paired["eligible19"], paired["eligible22"] = old_keep, keep
    paired.to_parquet(DATA / "response19_vs22_all_base.parquet", index=False)
    retained = frame.loc[keep].reset_index(drop=True)
    assert retained.pixel_id.is_unique and (retained.tac_pvalue < .05).all()
    assert (retained.QA01_missing_fraction < .5).all() and np.isfinite(retained[names]).all().all()
    np.testing.assert_array_equal(retained.y, (retained.y_continuous >= 0).astype("int8"))
    target = DATA / "analysis_endpoints_2005_2026_QA01_MDNlt0p5.parquet"
    retained.to_parquet(target, index=False)
    pd.testing.assert_frame_equal(retained, pd.read_parquet(target))
    pd.DataFrame(counts).to_csv(DATA / "cohort19_vs22.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(transitions).to_csv(DATA / "cohort_transitions.csv", index=False, encoding="utf-8-sig")
    prep.dump(DATA / "preflight.json", dict(passed=True, checks=checks, input_inventory=input_inventory))
    saved = dict(status="completed", qa="SummaryQA in [0,1]", candidates=names,
        table=str(target), table_sha256=rf.digest(target), response_raster=str(response), response_sha256=rf.digest(response),
        TAC_endpoints=YEARS.tolist(), TAC_reference_years=(YEARS - 1).tolist(),
        baseline_table=str(original_table), baseline_table_sha256=frozen["table_sha256"],
        baseline_models=str(source.MODELS), preprocessing="Unchanged unified 2000-2025 residual preprocessing",
        only_scientific_change="OLS response trend uses all22 instead of first19, with corresponding significance eligibility",
        predictor_period_policy="All X fixed to approved first19 input values; kNDVI mean/trend remain 2000-2023",
        missingness_policy="Unchanged QA0/1 missing fraction over 2000-2023 <0.5 plus original stable/legacy masks",
        response="OLS over endpoints 2005-2026, TAC within [-1,1], >=3 valid windows, ordinary OLS P<0.05",
        boundary_effects_eliminated=False, spatially_independent_validation=False,
        original_models_preserved=True, forbidden_data_read=False,
        input_hashes=hashes, counts=counts, transitions=transitions, script_sha256=rf.digest(__file__))
    prep.dump(marker, saved)
    for row in counts:
        if row["period"] == "22_endpoints":
            prep.log("Full22 cohort: " + str(row))
    return saved


def worker(job):
    veg, rep = job
    rf.THREADS = 6
    manifest = rf.read(DATA / "source_manifest.json")
    rf.run_group(veg, MODELS, Path(manifest["table"]), manifest["candidates"], run_indices=[rep])
    return job


def verify_models(manifest):
    frame = pd.read_parquet(manifest["table"])
    checks = []
    for veg in ["Forest", "Non-forest"]:
        group = frame.loc[frame.vegetation.eq(veg)].reset_index(drop=True)
        for rep in range(30):
            folder = MODELS / veg / "runs" / f"{rep:02d}"
            marker, design = rf.read(folder / "complete.json"), rf.read(folder / "design.json")
            assert marker["seed"] == rf.FIRST_SEED + rep
            assert rf.digest(folder / "model.joblib") == marker["model_sha256"]
            assert rf.digest(folder / "importance.csv") == marker["importance_sha256"]
            with np.load(folder / "split_indices.npz") as split:
                train0, train, test = split["training_before_iqr"], split["training"], split["test"]
                assert not np.intersect1d(train0, test).size and np.isin(train, train0).all()
                np.testing.assert_array_equal(np.sort(np.r_[train0, test]), np.arange(len(group)))
            assert design["max_VIF"] < 10 and design["test_filtered"] is False
            prediction = pd.read_parquet(folder / "predictions.parquet")
            np.testing.assert_array_equal(prediction.pixel_id, group.iloc[test].pixel_id)
            np.testing.assert_array_equal(prediction.y, group.iloc[test].y)
            metrics = rf.read(folder / "metrics.json")
            accuracy = float(((prediction.probability_decline >= .5).astype(int) == prediction.y).mean())
            assert abs(accuracy - metrics["accuracy"]) < 1e-12
            with np.load(folder / "per_tree_OOB.npz") as saved:
                delta = saved["delta"]
                assert delta.shape == (300, 3, len(design["selected_features"])) and np.isfinite(delta).all()
                calculated = (delta.mean(axis=0) / delta.std(axis=0, ddof=1)).mean(axis=0)
                importance = pd.read_csv(folder / "importance.csv").set_index("variable").loc[design["selected_features"]]
                np.testing.assert_allclose(importance.standardized_importance, calculated, rtol=1e-10, atol=1e-12)
            checks.append(dict(vegetation=veg, repeat=rep, n_training=len(train), n_test=len(test),
                removed=", ".join(design["removed_features"]), verified=True))
    pd.DataFrame(checks).to_csv(OUT / "RF_verification_all60.csv", index=False, encoding="utf-8-sig")
    prep.dump(OUT / "RF_verified.json", dict(status="verified", models=60,
        table_sha256=rf.digest(manifest["table"]), RF_endpoints=[2005, 2026], RF_reference_years=[2004, 2025],
        input_policy="Only response period changed; X and missingness unchanged", candidates=manifest["candidates"]))


def comparison(manifest):
    folder = OUT / "comparison"
    folder.mkdir(exist_ok=True)
    metrics, rankings = [], []
    lines = ["# Full22 versus first19 RF", "",
        "Full22 endpoints: 2005-01-01 through 2026-01-01; reference years: 2004-2025.",
        "The original first19 response has endpoints 2005-2023 (reference years 2004-2022).",
        "Only response period and resulting significance-selected eligibility change. All X and missingness values are fixed.",
        "Forest/non-forest each have 30 RF fits, 300 trees, independent per-group TPE retuning and training-only VIF <10.",
        "Validation is random stratified 70/30, NOT spatially independent. Refit ranges are 2.5-97.5 percentiles, NOT external confidence intervals.",
        "Full22 includes end-of-record moving-mean effects; the experiment does not remove them.", ""]
    for period, root in [("19_endpoints", source.MODELS), ("22_endpoints", MODELS)]:
        for veg in ["Forest", "Non-forest"]:
            summary = root / veg / "summary"
            performance = pd.read_csv(summary / "performance_summary.csv")
            performance.insert(0, "vegetation", veg)
            performance.insert(0, "period", period)
            metrics.append(performance)
            rank = pd.read_csv(summary / "importance_summary.csv")
            rank.loc[rank.retention_count.eq(0), "rank_by_median"] = np.nan
            rank.insert(0, "vegetation", veg)
            rank.insert(0, "period", period)
            rankings.append(rank)
            row = performance.set_index("metric").loc["accuracy"]
            top = rank.loc[rank.retention_count.gt(0)].head(6)
            params = rf.read(root / veg / "tuning/selection.json")["chosen"]["params"]
            cohort = next(row for row in manifest["counts"] if row["period"] == period and row["vegetation"] == veg)
            lines += [f"## {period}, {veg}", f"Eligible pixels: {cohort['n_eligible']:,}; decline fraction: {cohort['decline_fraction']:.2%}.",
                f"Accuracy: {row['median']:.2%} [{row['q025']:.2%}, {row['q975']:.2%}].",
                "Top six: " + ", ".join(top.variable) + ".", "Parameters: " + str(params) + ".", ""]
    pd.concat(metrics, ignore_index=True).to_csv(folder / "performance19_vs22.csv", index=False, encoding="utf-8-sig")
    ranking = pd.concat(rankings, ignore_index=True)
    ranking.to_csv(folder / "importance19_vs22.csv", index=False, encoding="utf-8-sig")
    for veg in ["Forest", "Non-forest"]:
        one = ranking.loc[ranking.vegetation.eq(veg)].set_index(["period", "variable"])
        for variable in ["P_NIPV", "P_SI"]:
            old, new = one.loc[("19_endpoints", variable)], one.loc[("22_endpoints", variable)]
            lines.append(f"- {veg} {variable}: {old.standardized_importance:.3f} (rank {old.rank_by_median:g}) -> {new.standardized_importance:.3f} (rank {new.rank_by_median:g}).")
    (folder / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    prep.log("Full22 RF comparison tables completed")


def train(manifest):
    MODELS.mkdir(parents=True, exist_ok=True)
    config = dict(rf.CONFIG, version="QA01_full22_20261010", preprocessing_period="2000-2025",
        RF_endpoints=[2005, 2026], RF_reference_years=[2004, 2025], threads_per_model=6,
        processes=4, tuning_threads_per_model=12,
        missing_mask="Unchanged QA0/1 2000-2023 missingness <0.5 plus original stable/legacy masks",
        predictors="Unchanged first19 X; 22 candidates followed by per-training-set VIF",
        performance_scope="Random holdout internal validation; seed0 test untouched during tuning")
    protocol = dict(configuration=config, source_manifest_sha256=rf.digest(DATA / "source_manifest.json"),
        table_sha256=manifest["table_sha256"], script_sha256=rf.digest(__file__),
        training_helper_sha256=rf.digest(rf.__file__), original_models_preserved=True)
    if (MODELS / "protocol.json").exists():
        assert rf.read(MODELS / "protocol.json") == protocol, "Protocol changed; do not mix experiments"
    else:
        prep.dump(MODELS / "protocol.json", protocol)
    started = time.monotonic()
    prep.dump(OUT / "status.json", dict(status="running", stage="Bayesian retuning", models_planned=60))
    rf.THREADS = 12
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(rf.run_group, veg, MODELS, Path(manifest["table"]), manifest["candidates"], run_indices=[])
            for veg in ["Forest", "Non-forest"]]
        for future in futures:
            future.result()
    pending = [(veg, rep) for rep in range(30) for veg in ["Forest", "Non-forest"]
        if not (MODELS / veg / "runs" / f"{rep:02d}" / "complete.json").exists()]
    completed = 60 - len(pending)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, job) for job in pending]
        for future in as_completed(futures):
            job = future.result()
            completed += 1
            prep.dump(OUT / "status.json", dict(status="running", stage="RF and OOB importance",
                completed_models=completed, planned_models=60, last_completed=list(job)))
            prep.log(f"Full22 RF {completed}/60 completed")
    for veg in ["Forest", "Non-forest"]:
        rf.summarize_group(veg, MODELS / veg, manifest["candidates"])
    rf.figures(MODELS)
    for extension in ["png", "pdf"]:
        (MODELS / f"figures/Fig_RF_QA0_OOB_top10_30runs.{extension}").replace(
            MODELS / f"figures/Fig_RF_QA01_full22_OOB_top10_30runs.{extension}")
    verify_models(manifest)
    comparison(manifest)
    prep.dump(OUT / "status.json", dict(status="completed", models=60, RF_endpoints=[2005, 2026],
        RF_reference_years=[2004, 2025], verification="All60 splits, VIF, prediction labels and per-tree OOB checked",
        elapsed_seconds=time.monotonic() - started))
    prep.log("Full22 RF ALL60 COMPLETED AND VERIFIED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare()
    if not args.prepare_only:
        try:
            train(manifest)
        except Exception as error:
            prep.dump(OUT / "status.json", dict(status="failed", error=repr(error)))
            raise


if __name__ == "__main__":
    main()

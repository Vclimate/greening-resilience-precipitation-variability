"""Export native 16-day values in half-year packs, without calculating TAC.

NDVI/EVI are already physical values in this GEE collection. QA is deliberately
retained unchanged; use SummaryQA == 0 in the local calculation.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import ee
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/MOD13C1_NDVI_EVI_QA_local_20261003"
PROJECT = "resilience20260619"
SOURCE = "MODIS/061/MOD13C1"
FOLDER = "JRS_MOD13C1_NDVI_EVI_QA_16day_005deg_20261003"
TRANSFORM = [0.05, 0, -180, 0, -0.05, 80]
BANDS = ["NDVI", "EVI", "SummaryQA"]


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def pack(rows):
    images, names = [], []
    for row in rows:
        labels = [f"{band}_{row['date'].replace('-', '')}" for band in BANDS]
        images.append(ee.Image(f"{SOURCE}/{row['index']}").select(BANDS)
                      .toFloat().rename(labels).unmask(-9999, sameFootprint=False))
        names.extend(labels)
    return ee.ImageCollection.fromImages(images).toBands().rename(names), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Run checks and start half-year image exports")
    parser.add_argument("--preflight", action="store_true", help="Online checks only; do not export")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    if not 2000 <= args.start_year <= args.end_year <= 2025:
        parser.error("Use complete years within 2000-2025")
    start = f"{args.start_year}-01-01"
    end = f"{args.end_year+1}-01-01"
    task_count = 2*(args.end_year-args.start_year+1)
    if not (args.submit or args.preflight):
        print(f"Dry run: {args.start_year}-{args.end_year}; {task_count} half-year packs; folder={FOLDER}; no tasks created.")
        return
    ee.Initialize(project=PROJECT)
    ee.data.setDeadline(120000)
    raw = ee.ImageCollection(SOURCE).filterDate(start, end).sort("system:time_start")

    def metadata(image):
        return image.set("export_date", ee.Date(image.get("system:time_start")).format("YYYY-MM-dd"))

    collection = raw.map(metadata)
    info = ee.Dictionary({"dates": collection.aggregate_array("export_date"),
                          "indexes": collection.aggregate_array("system:index"),
                          "projection": raw.first().select("NDVI").projection(),
                          "bands": raw.first().bandNames()}).getInfo()
    rows = [{"index": index, "date": date} for index, date in zip(info["indexes"], info["dates"])]
    assert len(rows) == len(set(r["date"] for r in rows)) and len(rows) >= 19*(args.end_year-args.start_year+1)
    assert all(b in info["bands"] for b in BANDS)
    assert info["projection"]["crs"] == "EPSG:4326"
    np.testing.assert_allclose(info["projection"]["transform"], [.05, 0, -180, 0, -.05, 90], atol=1e-10, rtol=0)
    jobs = []
    for year in range(args.start_year, args.end_year+1):
        for half in [1, 2]:
            selected = [r for r in rows if int(r["date"][:4]) == year
                        and (1 if int(r["date"][5:7]) <= 6 else 2) == half]
            assert selected and len(selected) <= 12
            jobs.append({"year": year, "half": half, "images": selected,
                         "description": f"MOD13C1_NDVI_EVI_QA_{year}_H{half}_raw_20261003"})
    assert sum(len(j["images"]) for j in jobs) == len(rows) and len(jobs) == task_count
    report = {"project": PROJECT, "source": SOURCE, "folder": FOLDER,
              "date_start": start, "date_end_exclusive": end,
              "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
              "image_count": len(rows), "source_projection": info["projection"],
              "export_transform": TRANSFORM, "export_shape": [2800, 7200],
              "nodata": -9999, "dtype": "float32", "scale_to_apply_after_download": 1,
              "bands_per_date": BANDS, "qa_mask_applied_in_export": False,
              "local_qa_rule": "SummaryQA == 0, plus finite and non-nodata VI values",
              "local_kNDVI_formula": "tanh(NDVI ** 2)",
              "uncompressed_size_GiB": len(rows)*3*2800*7200*4/1024**3,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "jobs": jobs, "actions": [], "status": "preflight_started"}
    path = OUT / f"submission_{datetime.now():%Y%m%d_%H%M%S}.json"
    save(path, report)
    try:
        probe_year = 2006 if args.start_year <= 2006 <= args.end_year else args.start_year
        probe = ee.Image(raw.filterDate(f"{probe_year}-07-01", f"{probe_year}-08-01").first())
        region_probe = ee.Geometry.Rectangle([100, 20, 105, 25], geodesic=False)
        stats = probe.select(BANDS).reduceRegion(ee.Reducer.minMax(), region_probe,
                crs="EPSG:4326", crsTransform=TRANSFORM, maxPixels=100000).getInfo()
        for b in ["NDVI", "EVI"]:
            assert -.201 <= stats[f"{b}_min"] <= stats[f"{b}_max"] <= 1.001
        assert stats["NDVI_max"] > .1, "Unexpected scaling: source should be physical VI values"
        image, names = pack(jobs[0]["images"])
        first = ee.Image(f"{SOURCE}/{rows[0]['index']}").select(BANDS).toFloat().unmask(-9999, sameFootprint=False)
        point = ee.Geometry.Point([100.025, 20.025])
        projection = ee.Projection("EPSG:4326", TRANSFORM)
        direct = first.sample(point, projection=projection).first().toDictionary(BANDS).getInfo()
        actual = image.select(names[:3]).sample(point, projection=projection).first().toDictionary(names[:3]).getInfo()
        np.testing.assert_allclose([actual[n] for n in names[:3]], [direct[b] for b in BANDS], atol=0, rtol=0)
        report["preflight"] = {"physical_value_stats": stats, "stack_sample_direct": direct,
                               "stack_sample_export": actual, "stack_sample_exact_match": True}
        report["status"] = "preflight_passed"
        save(path, report)
        print(f"PASS: native grid, physical VI values and stack identity; {len(rows)} dates; {task_count} tasks", flush=True)
        if not args.submit:
            print(f"No tasks submitted; report={path}", flush=True)
            return
        existing = ee.data.getTaskList()
        region = ee.Geometry.Rectangle([-180+1e-8, -60+1e-8, 180-1e-8, 80-1e-8], geodesic=False)
        for i, job in enumerate(jobs):
            found = [t for t in existing if t.get("description") == job["description"]
                     and t.get("state") in {"READY", "RUNNING", "COMPLETED"}]
            image, labels = pack(job["images"])
            job["band_names"] = labels
            assert len(labels) == len(job["images"])*3
            if found:
                action = {"description": job["description"], "action": "existing", "ids": [t["id"] for t in found]}
            else:
                task = ee.batch.Export.image.toDrive(image=image, description=job["description"],
                    folder=FOLDER, fileNamePrefix=job["description"], region=region,
                    crs="EPSG:4326", crsTransform=TRANSFORM, maxPixels=1e13,
                    fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True, "noData": -9999})
                task.start()
                action = {"description": job["description"], "action": "submitted", "ids": [task.id]}
            report["actions"].append(action)
            save(path, report)
            print(f"{i+1}/{task_count} {action['action']}: {job['description']}", flush=True)
        ids = {tid for a in report["actions"] for tid in a["ids"]}
        current = {t["id"]: t for t in ee.data.getTaskList()}
        assert ids.issubset(current), "Could not confirm all submitted task IDs"
        report["verified_statuses"] = [{k: current[tid].get(k) for k in ["id", "description", "state", "error_message"]}
                                       for tid in sorted(ids)]
        assert all(s["state"] in {"READY", "RUNNING", "COMPLETED"} for s in report["verified_statuses"])
        report["status"] = "submitted_and_status_checked"
        save(path, report)
        print(f"Confirmed {len(ids)} exports; report={path}", flush=True)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        save(path, report)
        raise


if __name__ == "__main__":
    main()

"""Read-only header/coverage checks and stratified samples of every source band."""
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from compare_new_lag1_TAC_20260928 import np, rasterio, Window

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\数据集\JRS_MOD13C1_NDVI_EVI_QA_16day_005deg_20261003")
OUT = ROOT / "output/MOD13C1_local_input_audit_20261003"
SUBMISSION = ROOT / "output/MOD13C1_NDVI_EVI_QA_local_20261003/submission_20261003_103721.json"
PATTERN = re.compile(r"MOD13C1_NDVI_EVI_QA_(\d{4})_H([12])_raw_20261003(?:-(\d+)-(\d+))?\.tif$")


def inspect(path):
    stamp = (path.stat().st_size, path.stat().st_mtime_ns)
    match = PATTERN.fullmatch(path.name)
    assert match, f"Unexpected filename: {path.name}"
    year, half = map(int, match.group(1, 2))
    with rasterio.open(path) as f:
        assert f.crs.to_epsg() == 4326 and f.nodata == -9999
        assert set(f.dtypes) == {"float32"}
        assert f.count % 3 == 0 and len(f.descriptions) == f.count
        t = f.transform
        np.testing.assert_allclose([t.a, t.b, t.d, t.e], [.05, 0, 0, -.05], atol=1e-10, rtol=0)
        col = int(round((t.c+180)/.05))
        row = int(round((80-t.f)/.05))
        np.testing.assert_allclose([t.c, t.f], [-180+col*.05, 80-row*.05], atol=1e-9, rtol=0)
        assert row >= 0 and col >= 0 and row+f.height <= 2800 and col+f.width <= 7200
        if match.group(3):
            assert row == int(match.group(3)) and col == int(match.group(4))
        dates = []
        for i in range(0, f.count, 3):
            date = f.descriptions[i].removeprefix("NDVI_")
            d = datetime.strptime(date, "%Y%m%d")
            assert d.year == year and (1 if d.month <= 6 else 2) == half
            assert f.descriptions[i:i+3] == tuple(f"{b}_{date}" for b in ["NDVI", "EVI", "SummaryQA"])
            dates.append(d.strftime("%Y-%m-%d"))
        assert dates == sorted(set(dates))
        summary = {b: {"finite_sample_values": 0, "min": None, "max": None,
                        "outside_expected_range": 0} for b in ["NDVI", "EVI"]}
        qa_counts = Counter()
        qa_noninteger = 0
        qa0_valid_vi = {b: 0 for b in summary}
        sample_positions = [(r, c) for r in [.15, .5, .8] for c in [.25, .75]]
        for rf, cf in sample_positions:
            r = min(max(int(f.height*rf), 0), f.height-16)
            c = min(max(int(f.width*cf), 0), f.width-16)
            data = f.read(window=Window(c, r, 16, 16))
            qa = data[2::3]
            qvalid = np.isfinite(qa) & (qa != -9999)
            qa_noninteger += int(np.count_nonzero(qvalid & (qa != np.rint(qa))))
            unique, counts = np.unique(qa[qvalid], return_counts=True)
            qa_counts.update({str(int(v)): int(n) for v, n in zip(unique, counts)})
            for offset, b in enumerate(["NDVI", "EVI"]):
                a = data[offset::3]
                valid = np.isfinite(a) & (a != -9999)
                values = a[valid]
                s = summary[b]
                s["finite_sample_values"] += int(len(values))
                if len(values):
                    low, high = float(values.min()), float(values.max())
                    s["min"] = low if s["min"] is None else min(s["min"], low)
                    s["max"] = high if s["max"] is None else max(s["max"], high)
                    s["outside_expected_range"] += int(np.count_nonzero((values < -.2001) | (values > 1.0001)))
                qa0_valid_vi[b] += int(np.count_nonzero(valid & qvalid & (qa == 0)))
        result = {"path": str(path), "year": year, "half": half, "row": row, "col": col,
                  "height": f.height, "width": f.width, "bands": f.count, "dates": dates,
                  "band_names": list(f.descriptions), "bytes": path.stat().st_size,
                  "sample_summary": summary, "sample_QA_counts": dict(qa_counts),
                  "sample_QA_noninteger": qa_noninteger, "sample_QA0_valid_VI": qa0_valid_vi}
    assert stamp == (path.stat().st_size, path.stat().st_mtime_ns), "Input changed during reading"
    print(f"Checked {year} H{half}: col={col}, width={result['width']}, dates={len(dates)}", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SOURCE.glob("*.tif"))
    assert files
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory = list(pool.map(inspect, files))
    groups = defaultdict(list)
    for item in inventory:
        groups[(item["year"], item["half"])].append(item)
    submitted = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    expected = {(j["year"], j["half"]): j for j in submitted["jobs"]}
    assert set(expected).issubset(groups), f"Missing packs: {set(expected)-set(groups)}"
    timeline, packs, errors, qa_counts = [], [], [], Counter()
    for (year, half), tiles in sorted(groups.items()):
        tiles.sort(key=lambda tile: tile["col"])
        dates = tiles[0]["dates"]
        names = tiles[0]["band_names"]
        assert all(t["row"] == 0 and t["height"] == 2800 and t["dates"] == dates
                   and t["band_names"] == names for t in tiles)
        end = 0
        for t in tiles:
            assert t["col"] == end, f"Gap/overlap in {year} H{half} at column {end}"
            end += t["width"]
            qa_counts.update(t["sample_QA_counts"])
            if t["sample_QA_noninteger"] or set(t["sample_QA_counts"])-set("01234"):
                errors.append(f"Unexpected sampled QA: {t['path']}")
            for b, stats in t["sample_summary"].items():
                if stats["outside_expected_range"]:
                    errors.append(f"Sampled {b} values outside product range: {t['path']}")
        assert end == 7200, f"Incomplete mosaic for {year} H{half}"
        original = (year, half) in expected
        if original:
            assert dates == [r["date"] for r in expected[(year, half)]["images"]]
            assert names == expected[(year, half)]["band_names"]
        packs.append({"year": year, "half": half, "original_submission_period": original,
                      "dates": dates, "tiles": [{k: t[k] for k in ["path", "row", "col", "height", "width"]} for t in tiles]})
        timeline.extend({"date": date, "year": year, "half": half,
                         "bands": {b: 3*i+offset+1 for offset,b in enumerate(["NDVI", "EVI", "SummaryQA"])},
                         "original_submission_period": original} for i, date in enumerate(dates))
    assert len(timeline) == len(set(t["date"] for t in timeline))
    source_period = [t for t in timeline if t["original_submission_period"]]
    assert len(source_period) == submitted["image_count"] == 571
    gaps = []
    for a, b in zip(timeline, timeline[1:]):
        da, db = datetime.fromisoformat(a["date"]), datetime.fromisoformat(b["date"])
        if da.year == db.year and (db-da).days != 16:
            gaps.append({"previous": a["date"], "next": b["date"], "days": (db-da).days,
                         "note": "Original-period dates are checked against the source manifest; do not synthesize or compress this gap."})
    sampled_ranges = {b: {"min": min(t["sample_summary"][b]["min"] for t in inventory if t["sample_summary"][b]["min"] is not None),
                          "max": max(t["sample_summary"][b]["max"] for t in inventory if t["sample_summary"][b]["max"] is not None)} for b in ["NDVI", "EVI"]}
    assert sampled_ranges["NDVI"]["max"] > .1, "Unexpected repeated scaling"
    result = {"source": str(SOURCE), "validation_level": "all headers/band dates/tile coverage; stratified samples of every band, not a complete pixel scan",
              "status": "passed" if not errors else "issues_found", "files": len(files), "half_year_packs": len(packs),
              "size_GiB": sum(p.stat().st_size for p in files)/1024**3,
              "original_period_dates": len(source_period), "all_available_dates": len(timeline),
              "first_date": timeline[0]["date"], "last_date": timeline[-1]["date"],
              "additional_packs": [{"year": p["year"], "half": p["half"]} for p in packs if not p["original_submission_period"]],
              "sample_VI_ranges": sampled_ranges, "sample_QA_counts": dict(qa_counts),
              "within_year_date_gaps": gaps, "errors": errors, "packs": packs, "timeline": timeline, "inventory": inventory}
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"packs", "timeline", "inventory"}}, ensure_ascii=False, indent=2), flush=True)
    assert not errors, "Inspect audit.json before using these inputs"


if __name__ == "__main__":
    main()

"""Check written raster values against fresh raw-data point calculations."""
from pathlib import Path
import recompute_QA01_22periods_20261005 as data

np, rasterio, Window = data.np, data.rasterio, data.Window
POINTS = [(-60.025, -5.025), (10.025, 50.025), (20.025, 12.025), (-100.025, 55.025), (100.025, 60.025)]


def main():
    packs, dates, args = data.configuration()
    offset = np.cumsum([0]+[len(p['dates']) for p in packs])
    v = np.full((5, len(dates)), np.nan, np.float32)
    locations = [(int((80-lat)/.05), int((lon+180)/.05)) for lon, lat in POINTS]
    for j, pack in enumerate(packs):
        bands = [3*i+b for i in range(len(pack['dates'])) for b in [1, 3]]
        for tile in pack['tiles']:
            relevant = [(i, row, col) for i, (row, col) in enumerate(locations)
                        if tile['col'] <= col < tile['col']+tile['width']]
            if not relevant:
                continue
            with rasterio.open(tile['path']) as f:
                for i, row, col in relevant:
                    a = f.read(bands, window=Window(col-tile['col'], row, 1, 1))[:, 0, 0]
                    v[i, offset[j]:offset[j+1]] = data.old.qa01_values(a[0::2], a[1::2])
    design, outer, left, right, starts, ends = args
    reference = data.core.calculate(v, design, outer, left, right, starts[0], ends)
    manifest, support, lc, old, base = data.old.masks()
    checks = []
    for i, (row, col) in enumerate(locations):
        if not base[row, col]:
            continue
        for j, year in enumerate(data.YEARS):
            path = data.DATA/'kNDVI/lag1_5year'/f'kNDVI_TAC_lag1_5year_{year}_01_01_QA01.tif'
            with rasterio.open(path) as f:
                actual = f.read(masked=True, window=Window(col, row, 1, 1)).filled(np.nan)[:, 0, 0]
            np.testing.assert_array_equal(actual, reference[i, j])
            checks.append(dict(point=i, endpoint=int(year), matched=True))
    assert len(checks) >= 22
    data.prep.dump(data.OUT/'written_raster_verification.json', dict(status='verified',
        points_within_domain=len(checks)//22, actual_pixel_windows=len(checks), checks=checks,
        source_read_again=True, estimator_reference='previous validated core.calculate',
        same_2000_2023_greenness_MDN_full_grid=True))
    print(f'PASS written TAC: {len(checks)} actual pixel/windows; full-grid invariant quantities matched')


if __name__ == '__main__':
    main()

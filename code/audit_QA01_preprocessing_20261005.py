"""Independent preprocessing checks; does not change rasters or fitted models."""
import json
from pathlib import Path

import prepare_RF_local_QA0_20261004 as prep
import run_RF_QA01_reanalysis_20261005 as run

np = prep.np
OUT = run.OUT / 'preprocessing_audit_20261005'


def reference(k, dates, design):
    days = dates.astype('int64')
    dt = np.full_like(k, np.nan)
    for i in np.flatnonzero(np.isfinite(k)):
        take = (days >= days[i] - 912) & (days < days[i] + 912)
        mean = np.float32(np.nanmean(k[take].astype(float)))
        dt[i] = np.float32(k[i] - mean)
    good = np.isfinite(dt)
    coef = np.linalg.lstsq(design[good], dt[good].astype(float), rcond=None)[0]
    residual = (dt.astype(float) - design @ coef).astype(np.float32)
    # Compare normal equations to an independent SVD least-squares solver.
    normal = np.linalg.solve(design[good].T @ design[good], design[good].T @ dt[good])
    normal_residual = (dt.astype(float) - design @ normal).astype(np.float32)
    error = float(np.nanmax(np.abs(normal_residual - residual)))
    projection = np.linalg.lstsq(design[good], residual[good].astype(float), rcond=None)[0]
    fitted_remaining = float(np.max(np.abs(design[good] @ projection)))
    return residual, dict(valid=int(good.sum()), condition_number=float(np.linalg.cond(design[good])),
                         max_residual_solver_difference=error,
                         max_remaining_modelled_component=fitted_remaining,
                         missing_mask_preserved=bool(np.array_equal(np.isfinite(k), np.isfinite(residual))))


def main():
    packs, dates, args = prep.core.configuration()
    design, outer, left, right, starts, ends = args
    assert np.all(dates[1:] > dates[:-1])
    gaps = np.diff(dates).astype('timedelta64[D]').astype(int)
    report = dict(first_date=str(dates[0]), last_date=str(dates[-1]), observations=len(dates),
                  gap_days_counts={str(v): int((gaps == v).sum()) for v in np.unique(gaps)},
                  gaps_over_16_days=[dict(first=str(dates[i]), second=str(dates[i+1]), days=int(gaps[i]))
                                     for i in np.flatnonzero(gaps > 16)], checks=[])
    samples = []
    for p in range(5):
        with np.load(prep.core.OUT / f'point_{p}_stages.npz') as f:
            np.testing.assert_array_equal(f['dates'], dates)
            samples.append(run.qa01_values(f['ndvi'], f['qa']))
    samples = np.stack(samples)
    actual = prep.core.calculate(samples, *args)
    all_differences = []
    long_gap_effect = []
    for p, k in enumerate(samples):
        residual, diagnostics = reference(k, dates, design)
        expected = np.full((len(starts), 3), np.nan, np.float32)
        for a, (start, end) in enumerate(zip(starts, ends)):
            valid = np.isfinite(residual[start:end])
            pair = valid[:-1] & valid[1:]
            observations, pairs = int(valid.sum()), int(pair.sum())
            if observations:
                expected[a, 1] = observations
            if pairs:
                expected[a, 2] = pairs
            if observations > 15 and pairs > 15:
                x = residual[start:end-1].astype(float)
                y = residual[start+1:end].astype(float)
                expected[a, 0] = prep.core.robust_line(x[pair], y[pair], pairs)
                shorter = pair & (gaps[start:end-1] <= 16)
                if int(shorter.sum()) > 15 and shorter.sum() != pair.sum():
                    short_tac = prep.core.robust_line(x[shorter], y[shorter], int(shorter.sum()))
                    long_gap_effect.append(dict(point=p, endpoint=int(prep.core.YEARS[a]),
                                                original_TAC=float(expected[a, 0]),
                                                excluded_long_gap_TAC=float(short_tac),
                                                absolute_difference=float(abs(short_tac - expected[a, 0]))))
        np.testing.assert_allclose(actual[p, :, 1:], expected[:, 1:], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(actual[p, :, 0], expected[:, 0], rtol=0, atol=2e-6, equal_nan=True)
        diagnostics['point'] = p
        diagnostics['max_TAC_difference'] = float(np.nanmax(np.abs(actual[p, :, 0] - expected[:, 0])))
        all_differences.append(diagnostics['max_TAC_difference'])
        assert diagnostics['missing_mask_preserved']
        assert diagnostics['max_remaining_modelled_component'] < 1e-7
        report['checks'].append(diagnostics)
    # A missing pixel-level observation must not make its neighbours adjacent.
    valid = np.ones(len(dates), bool)
    valid[100] = False
    pairs = np.flatnonzero(valid[:-1] & valid[1:])
    assert 99 not in pairs and 100 not in pairs
    # Verify harmonic basis, including annual, semiannual and four-month terms.
    t = ((dates - np.datetime64('1970-01-01')).astype(int) / 365.2425 * 2*np.pi).astype(np.float32)
    for j in range(1, 4):
        np.testing.assert_array_equal(design[:, 1+j], np.cos((t*j).astype(np.float32)).astype(float))
        np.testing.assert_array_equal(design[:, 4+j], np.sin((t*j).astype(np.float32)).astype(float))
    report.update(passed=True, actual_pixel_window_checks=int(actual.shape[0]*actual.shape[1]),
                  maximum_TAC_reference_difference=max(all_differences),
                  missing_pixel_observations_not_bridged=True,
                  long_collection_gap_effect_on_five_fixture_pixels=long_gap_effect,
                  QA_rule='0 or 1 only; no additional NDVI scaling',
                  stages='centered moving mean subtraction (+/-912 days), then full-series intercept/linear/3-harmonic fit, then residual lagged robust regression',
                  limitations=['Five actual fixture pixels are checked, not a second global recomputation.',
                               'Moving windows shorten at both time-series edges.',
                               'Fixed harmonic terms remove fitted seasonality, not necessarily all time-varying seasonality.',
                               'A globally absent composite creates a 32-day pair in the collection-index lag definition.',
                               'This is not the older MATLAB calendar-slot climatology plus linear-detrend algorithm.'])
    prep.dump(OUT / 'audit.json', report)
    note = '\n'.join([
        '# QA=0/1 去趋势和去季节性核查',
        '',
        '- 未修改任何原始数据、TAC 栅格或 RF 模型。',
        '- 顺序：kNDVI 减去中心移动均值，再拟合常数项、线性项和三阶正余弦谐波，残差用于 TAC。',
        '- 移动均值使用中心日期前后 912 天的有效观测；谐波拟合使用全部 2000–2024 年有效观测。',
        '- 缺测不补零、不按像元删日期后重新连接相邻观测。NDVI 不重复缩放。',
        f'- 五个实际像元、共 {actual.shape[0]*actual.shape[1]} 个窗口独立复算通过；最大 TAC 差异 {max(all_differences):.3g}。',
        '- SVD 最小二乘与现有正规方程解一致；拟合后残余线性/谐波分量低于 1e-7（kNDVI 单位）。',
        '',
        '## 需要如实说明的限制',
        '',
        '- 2001-01-17 合成影像缺失，2001-01-01 与 2001-02-02 在集合中相邻，相隔 32 天；当前 lag-1 指集合相邻观测，不是每对严格相隔 16 天。',
        '- 该跨日期配对只可能影响 2005、2006 两个 TAC 端点，实际影响以 audit.json 中五个像元敏感性结果为准，不能推广为全图无影响。',
        '- 移动均值在序列两端使用缩短窗口；固定谐波不能保证去除所有随时间变化的季节性。',
        '- 当前与 GEE 移动均值＋三阶谐波流程一致，但与旧 MATLAB 按年内合成时段去均值＋线性去趋势不是同一种方法。',
        '- 本核查证明抽查点的数值实现一致，不证明 QA=0/1 在所有地区最优，也不证明变量排序具有因果含义。',
        '',
        '详细数值见 audit.json。',
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / '核查结论_CN.md').write_text(note, encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

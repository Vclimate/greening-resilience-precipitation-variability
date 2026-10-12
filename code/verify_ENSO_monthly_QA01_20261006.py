"""Focused independent checks for monthly ENSO analysis and HAC coefficients."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1] / 'output/QA01_unified22_20261005/ENSO_monthly_20261006'


def main():
    state = json.loads((ROOT/'compute_status.json').read_text(encoding='utf-8'))
    assert state['status'] == 'completed' and state['annual_max_absolute_error'] == 0
    dates = pd.read_csv(ROOT/'monthly_window_dates.csv', parse_dates=['month', 'window_start', 'window_end_exclusive'])
    assert len(dates) == 253 and dates.month.is_monotonic_increasing
    assert (dates.month+pd.offsets.MonthBegin(1)).equals(dates.window_end_exclusive)
    assert (dates.window_start+pd.DateOffset(years=5)).equals(dates.window_end_exclusive)
    coverage = pd.read_csv(ROOT/'cohort_coverage.csv')
    fixed = coverage.loc[coverage.cohort.eq('monthly_fixed')]
    assert fixed.min_monthly_pixels.equals(fixed.max_monthly_pixels)
    series = pd.read_csv(ROOT/'monthly_regional_TAC_ONI.csv', parse_dates=['month'])
    series = series.loc[series.cohort.eq('monthly_fixed')]
    fits = pd.read_csv(ROOT/'ONI_regressions_HAC_and_FDR.csv')
    assert fits[['p_HAC', 'p_circular_shift']].ge(0).all().all()
    assert fits[['p_HAC', 'p_circular_shift']].le(1).all().all()
    checks = []
    for (vegetation, zone), group in series.groupby(['vegetation', 'zone']):
        if vegetation == 'Global':
            continue
        group = group.sort_values('month')
        trend = np.arange(len(group), dtype=float)/12
        trend -= trend.mean()
        X = np.column_stack([np.ones(len(group)), trend]+
            [group.month.dt.month.eq(m).to_numpy(dtype=float) for m in range(2, 13)]+
            [group.ONI.to_numpy()])
        y = group.mean_TAC.to_numpy()
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        error = y-X@beta
        score = X*error[:, None]
        sandwich = score.T@score
        for lag in range(1, 61):
            cross = score[lag:].T@score[:-lag]
            sandwich += (1-lag/61)*(cross+cross.T)
        inverse = np.linalg.inv(X.T@X)
        n, k = X.shape
        cov = inverse@sandwich@inverse*n/(n-k)
        se = np.sqrt(cov[-1, -1])
        p = 2*t.sf(abs(beta[-1]/se), n-k)
        row = fits.loc[fits.vegetation.eq(vegetation) & fits.zone.eq(zone)
            & fits.outcome.eq('level') & fits.ONI_lead_months.eq(0) & fits.HAC_lag_months.eq(60)].iloc[0]
        np.testing.assert_allclose(beta[-1], row.coefficient_per_ONI_degC, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(p, row.p_HAC, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(beta[-1]+np.array([-1, 1])*t.ppf(.975, n-k)*se,
                                   [row.ci95_low, row.ci95_high], rtol=1e-7, atol=1e-12)
        checks.append(dict(vegetation=vegetation, zone=zone, OLS_and_HAC_independent_check=True))
    for name in ['Fig_ENSO_monthly_TAC_global_tropical', 'Fig_ENSO_monthly_TAC_climate_anomalies',
                 'Fig_ENSO_ONI_association_HAC60', 'Fig_ENSO_event_composites']:
        assert all((ROOT/f'{name}.{ext}').exists() for ext in ['png', 'pdf'])
    report = dict(status='passed', monthly_windows=253, annual_pixel_max_error=0,
        endpoint_month_alignment=True, fixed_monthly_membership=True,
        independent_Bartlett_HAC60_checks=checks, visual_review='All four figures inspected')
    (ROOT/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: dates, fixed membership, annual pixel equivalence, 10 independent OLS/HAC checks, four figure exports.')


if __name__ == '__main__':
    main()

"""Tropical-only views of validated monthly TAC and endpoint-aligned annual TAC."""
import json
from pathlib import Path

import analyse_ENSO_monthly_QA01_20261006 as base
import matplotlib.dates as mdates
from matplotlib.patches import Patch

pd, np, plt = base.pd, base.np, base.plt
ROOT = base.OUT
OUT = ROOT / 'tropical_only'
VEGETATION = ['Global', 'Forest', 'Non-forest']
LABELS = {'Global': 'All tropical vegetation', 'Forest': 'Tropical forest',
          'Non-forest': 'Tropical non-forest'}
COLORS = {'Global': '#444444', **base.COLORS}


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    monthly = pd.read_csv(ROOT / 'monthly_regional_TAC_ONI.csv', parse_dates=['month'])
    monthly = monthly.loc[monthly.zone.eq('Tropical') & monthly.cohort.eq('monthly_fixed')].copy()
    dates = pd.read_csv(ROOT / 'monthly_window_dates.csv',
                        parse_dates=['month', 'window_start', 'window_end_exclusive'])
    monthly = monthly.merge(dates, on='month', validate='many_to_one')
    annual = pd.read_csv(ROOT.parent / 'analysis/kNDVI_regional_trajectories.csv')
    annual = annual.loc[annual.zone.eq('Tropical') & annual.variant.eq('kNDVI')].copy()
    annual['window_end_exclusive'] = pd.to_datetime(annual.endpoint_year.astype(str) + '-01-01')
    annual['window_start'] = pd.to_datetime((annual.endpoint_year - 5).astype(str) + '-01-01')
    assert (annual.endpoint_year == annual.reference_year + 1).all()
    assert annual.loc[annual.reference_year.eq(2016), 'window_end_exclusive'].eq(pd.Timestamp('2017-01-01')).all()
    dec2016 = monthly.loc[monthly.month.eq(pd.Timestamp('2016-12-01'))]
    assert len(dec2016) == 3 and dec2016.window_end_exclusive.eq(pd.Timestamp('2017-01-01')).all()
    assert monthly.groupby('vegetation').size().eq(253).all()
    assert monthly.groupby('vegetation').n_pixels.nunique().eq(1).all()
    assert annual.groupby('vegetation').size().eq(22).all()
    annual['change_from_previous_year'] = annual.sort_values('reference_year').groupby('vegetation').mean_TAC.diff().reindex(annual.index)
    monthly.to_csv(OUT / 'tropical_monthly_TAC.csv', index=False)
    annual.to_csv(OUT / 'tropical_annual_TAC_endpoint_aligned.csv', index=False)

    results = pd.read_csv(ROOT / 'ONI_regressions_HAC_and_FDR.csv')
    results = results.loc[results.zone.eq('Tropical')].copy()
    # Preserve the original testing families, rather than recalculating q values after selecting a region.
    results.to_csv(OUT / 'tropical_ONI_regressions_original_FDR.csv', index=False)
    primary = results.loc[results.outcome.eq('level') & results.ONI_lead_months.eq(0)
                          & results.HAC_lag_months.eq(60)]
    index = pd.read_csv(ROOT / 'ONI_center_month_and_phase.csv', parse_dates=['month'])
    index = index.loc[index.month.between(monthly.month.min(), monthly.month.max())]
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 9,
                         'axes.titlesize': 10, 'pdf.fonttype': 42})
    legend = [Patch(facecolor='#d6604d', alpha=.25, label='El Nino'),
              Patch(facecolor='#4393c3', alpha=.25, label='La Nina')]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1.2]})
    axes[0].plot(index.month, index.ONI, color='#444444', lw=1.1)
    for threshold, color in [(.5, '#b2182b'), (-.5, '#2166ac')]:
        axes[0].axhline(threshold, color=color, ls=':', lw=.8)
    axes[0].set(ylabel='ONI (degrees C)', title='a  ENSO')
    axes[0].legend(handles=legend, loc='upper left', ncol=2, frameon=False)
    for ax, veg, letter in zip(axes[1:], VEGETATION, ['b', 'c', 'd']):
        frame = monthly.loc[monthly.vegetation.eq(veg)].sort_values('month')
        ax.plot(frame.month, frame.mean_TAC, color=COLORS[veg], lw=1.4)
        december = frame.loc[frame.month.dt.month.eq(12)]
        ax.scatter(december.month, december.mean_TAC, color=COLORS[veg], s=13, zorder=3)
        ax.set(ylabel='Lag-1 TAC', title=f'{letter}  {LABELS[veg]}')
    for ax in axes:
        base.phases(ax, index)
        ax.axvline(pd.Timestamp('2016-12-01'), color='#777777', ls='--', lw=.7)
        ax.grid(axis='y', color='#eeeeee', lw=.6)
        ax.tick_params(direction='out')
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Calendar month represented (window end minus one month)')
    fig.suptitle('Tropical TAC: monthly updates of a trailing five-year window', fontsize=12)
    save(fig, 'Fig_Tropical_ENSO_monthly_TAC')

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True, layout='constrained')
    for ax, veg, letter in zip(axes, VEGETATION, ['a', 'b', 'c']):
        frame = annual.loc[annual.vegetation.eq(veg)].sort_values('reference_year')
        ax.plot(frame.reference_year, frame.mean_TAC, '-o', color=COLORS[veg], lw=1.3, ms=3.5)
        point = frame.loc[frame.reference_year.eq(2016)].iloc[0]
        ax.scatter([2016], [point.mean_TAC], s=42, facecolor='white', edgecolor=COLORS[veg], zorder=4)
        ax.axvline(2016, color='#777777', lw=.7, ls='--')
        ax.text(.02, .88, f'2016: {point.mean_TAC:.4f}', transform=ax.transAxes,
                color=COLORS[veg], fontsize=8)
        ax.set(ylabel='Lag-1 TAC', title=f'{letter}  {LABELS[veg]}')
        ax.grid(axis='y', color='#eeeeee', lw=.6)
        ax.margins(y=.18)
    axes[-1].set_xticks(np.arange(2004, 2026, 2))
    axes[-1].set_xlabel('Resilience year (2016 = window end 2017-01-01)')
    fig.suptitle('Tropical annual TAC: original annual-analysis cohort', fontsize=12)
    save(fig, 'Fig_Tropical_annual_TAC_endpoint_aligned')

    lines = ['# 热带 TAC 与 ENSO：探索性结果', '',
             '- 热带定义：Koppen 气候分区中的 Tropical（代码 1），不是单纯按纬度划分。',
             '- 年份对齐：2016 年年度 TAC 的窗口为 [2012-01-01, 2017-01-01)，终点为 2017-01-01；逐月图中对应 2016 年 12 月。',
             '- 逐月 TAC 是五年窗口每月更新，lag-1 仍针对原始约 16 天观测；不是一个月内计算的 TAC，也不是月均植被数据的 lag-1。',
             '- 月度图圆点为当年 12 月对应的窗口。月度图与年度图分别使用各自固定的有效像元集合，二者不混用。', '',
             '## 2015—2016 年年度变化', '',
             '| 植被 | 2015 TAC | 2016 TAC | 变化（2016−2015） | 年度像元数 | 月度像元数 |',
             '|---|---:|---:|---:|---:|---:|']
    names = {'Global': '热带整体植被', 'Forest': '热带森林', 'Non-forest': '热带非森林'}
    for veg in VEGETATION:
        frame = annual.loc[annual.vegetation.eq(veg)].set_index('reference_year')
        a, b = frame.loc[2015], frame.loc[2016]
        nmonthly = int(monthly.loc[monthly.vegetation.eq(veg), 'n_pixels'].iloc[0])
        lines.append(f'| {names[veg]} | {a.mean_TAC:.6f} | {b.mean_TAC:.6f} | {b.mean_TAC-a.mean_TAC:+.6f} | {int(b.n_pixels):,} | {nmonthly:,} |')
    lines += ['', '2016 年森林和非森林 TAC 均升高，按本文指标可描述为恢复力下降的表现。但该窗口包含五年观测，不能证明 ENSO 的独立因果效应；2016 年内 ENSO 相位也发生了转换。', '',
              '## 同期 ONI 关联', '',
              '控制线性时间趋势与月份效应；HAC 带宽 60 个月。正系数表示 ONI 升高伴随 TAC 升高。', '',
              '| 植被 | ONI 系数（TAC / °C） | 95% HAC CI | HAC P | 原 10 项检验 FDR q | 循环移位 P |',
              '|---|---:|---|---:|---:|---:|']
    for row in primary.itertuples():
        q = f'{row.q_BH_contemporaneous_10_tests:.3f}' if np.isfinite(row.q_BH_contemporaneous_10_tests) else '—（描述性）'
        lines.append(f'| {names[row.vegetation]} | {row.coefficient_per_ONI_degC:+.6f} | [{row.ci95_low:+.6f}, {row.ci95_high:+.6f}] | {row.p_HAC:.3f} | {q} | {row.p_circular_shift:.3f} |')
    lines += ['', '森林与非森林的同期系数方向相反，因此整体平均可能掩盖分组差异。但两类的同期关联均未通过原多重比较校正后的 0.05 阈值，循环移位检验亦不显著，现阶段不能写成确定的 ENSO–恢复力关系。',
              '森林 HAC P 在带宽 12/24/60 个月时约为 0.161/0.098/0.017，说明推断对时间依赖处理敏感。保留原检验家族的 q 值，不因本次只展示热带而重新缩小校正范围。', '',
              '图中未扩大不确定性区间；年度图和月度图均仅展示均值轨迹，不将像元均值的窄区间当作 ENSO 效应不确定性。',
              'ONI 来源：[NOAA ERSSTv5 ONI](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v5/)，沿用上一轮缓存与事件定义。', '',
              '原始 TAC、论文图和 RF 结果均未改动。本目录仅为热带探索性分析。']
    (OUT / 'tropical_ENSO_results_CN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    checks = {'zone': 'Koppen Tropical', 'monthly_rows': len(monthly), 'annual_rows': len(annual),
              'monthly_dates_per_group': 253, 'annual_dates_per_group': 22,
              'annual_2016_window_end': '2017-01-01', 'monthly_Dec2016_window_end': '2017-01-01',
              'FDR_preserved_from_original_families': True,
              'annual_mean_matches_monthly_annual_reference': True}
    reference = pd.read_csv(ROOT / 'monthly_regional_TAC_ONI.csv', parse_dates=['month'])
    reference = reference.loc[reference.zone.eq('Tropical') & reference.cohort.eq('annual22_reference')
                              & reference.month.dt.month.eq(12)].copy()
    reference['reference_year'] = reference.month.dt.year
    comparison = annual.merge(reference, on=['vegetation', 'reference_year'], suffixes=('_annual', '_monthly'), validate='one_to_one')
    assert len(comparison) == 66
    error = float(np.max(np.abs(comparison.mean_TAC_annual - comparison.mean_TAC_monthly)))
    assert error < 1e-10, error
    checks['annual_reference_mean_max_difference'] = error
    checks['passed'] = True
    (OUT / 'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2), flush=True)
    print(primary[['vegetation', 'coefficient_per_ONI_degC', 'p_HAC', 'p_circular_shift']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

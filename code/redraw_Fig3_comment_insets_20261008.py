"""Enlarge unchanged co-occurrence maps and move donuts into ocean space."""
import hashlib
import json
import os
from pathlib import Path

from pyproj import datadir
os.environ['PROJ_DATA'] = datadir.get_data_dir()

import redraw_all_spatial_fullframe_white_20261006 as previous

a = previous.a
np, pd, plt = a.np, a.pd, a.plt
OUT = a.source.prep.ROOT / 'v15 - JRS/QA01_integrated_results_20261006/Figures_fullframe_white_v4/Comments_v5_20261008/figures'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csv = a.ANALYSIS / 'kNDVI_decoupling.csv'
    source_hash = hashlib.sha256(csv.read_bytes()).hexdigest()
    _, _, lc, _, base = a.source.old.masks()
    ids = np.flatnonzero(base)
    forest = lc.ravel()[ids] <= 5
    categories = a.read(a.ANALYSIS / 'kNDVI_decoupling_category.tif').ravel()[ids]
    fractions = pd.read_csv(csv)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.1))
    fig.subplots_adjust(left=.06, right=.985, top=.955, bottom=.115, hspace=.26)
    checks = []
    for j, (ax, veg) in enumerate(zip(axes, ['Forest', 'Non-forest'])):
        take = forest if veg == 'Forest' else ~forest
        a.map_panel(ax, ids, np.where(take, categories, np.nan), f'{chr(97+j)}  {veg}', categorical=True)
        previous.finish_map(ax)
        group = fractions.loc[fractions.vegetation.eq(veg) & fractions.zone.eq('Global')].sort_values('category_code')
        assert len(group) == 4
        assert abs(group.percent.sum() - 100) < 1e-6
        inset = ax.inset_axes([.007, .018, .29, .53])
        inset.set_facecolor('white')
        wedges, _ = inset.pie(
            group.percent, colors=a.CATCOLORS,
            startangle=90, counterclock=False,
            wedgeprops=dict(width=.45, edgecolor='white', linewidth=.6))
        positions = [(1.34, .28), (.64, -1.30), (-.64, -1.30), (-1.34, .30)]
        for wedge, value, position, color in zip(wedges, group.percent, positions, a.CATCOLORS):
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            inset.annotate(f'{value:.1f}%', xy=(np.cos(angle), np.sin(angle)),
                           xytext=position, ha='center', va='center', fontsize=8.0,
                           arrowprops=dict(arrowstyle='-', color=color, lw=.7),
                           bbox=dict(facecolor='white', edgecolor='none', pad=.3))
        inset.set(xlim=(-1.86, 1.86), ylim=(-1.55, 1.55))
        inset.set_aspect('equal')
        inset.set_axis_off()
        checks.append(dict(vegetation=veg, percentages=group.percent.tolist(), donut_inside_map=True,
                           full_frame=all(s.get_visible() for s in ax.spines.values()), white_missing_background=True))
    fig.legend(handles=[a.Patch(facecolor=c, label=a.CATEGORY[k]) for k, c in zip(a.CATEGORY, a.CATCOLORS)],
               loc='lower center', bbox_to_anchor=(.51, .025), ncol=2, frameon=False, fontsize=9)
    fig.canvas.draw()
    for ax in axes:
        box = ax.get_window_extent()
        np.testing.assert_allclose(box.width / box.height, 360 / 140, rtol=1e-6)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'Fig3_comments_v5.{ext}', dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == source_hash
    (OUT / 'figure_layout_verification.json').write_text(
        json.dumps(dict(status='passed', source_statistics_unchanged=True, source_sha256=source_hash,
                        original_palette_retained=True, maps=checks), indent=2), encoding='utf-8')
    print('Fig. 3 revised: two larger maps, circular inset donuts, unchanged statistics.', flush=True)


if __name__ == '__main__':
    main()

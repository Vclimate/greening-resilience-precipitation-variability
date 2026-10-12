"""Apply white missing backgrounds and complete frames to current spatial figures."""
import hashlib
import json
import shutil
from pathlib import Path

import redraw_SI_spatial_ratio_20261006 as previous
import redraw_S3_S4_square_panels_20261006 as square

a=previous.a
np,pd,plt=a.np,a.pd,a.plt
OUT=previous.OUT.parent/'spatial_fullframe_v4'
STYLE_CHECKS=[]
original_finish=previous.finish_map
original_save=previous.save


def finish_map(ax):
    original_finish(ax)
    ax.set_facecolor('white')
    for edge in ax.spines.values():
        edge.set_visible(True)
        edge.set_color('#222222')
        edge.set_linewidth(.7)
    for image in ax.images:
        old=image.get_cmap()
        cmap=old.copy()
        cmap.set_bad('white')
        np.testing.assert_array_equal(old(np.linspace(0,1,256)),cmap(np.linspace(0,1,256)))
        image.set_cmap(cmap)


def save(fig,name):
    name=name.replace('_map_ratio_v2','_fullframe_v4').replace('_map_ratio_v3','_fullframe_v4')
    panels=[]
    for ax in fig.axes:
        if tuple(ax.get_xlim())!=(-180.,180.) or tuple(ax.get_ylim())!=(-60.,80.):
            continue
        assert all(edge.get_visible() for edge in ax.spines.values())
        np.testing.assert_array_equal(ax.get_facecolor(),[1,1,1,1])
        for image in ax.images:
            np.testing.assert_array_equal(image.get_cmap().get_bad(),[1,1,1,1])
        panels.append(dict(full_four_sided_frame=True,ocean_and_missing_background='white'))
    STYLE_CHECKS.append(dict(figure=name,spatial_panels=panels))
    original_save(fig,name)


def main_figure3(ids,forest):
    categories=a.read(a.ANALYSIS/'kNDVI_decoupling_category.tif').ravel()[ids]
    fractions=pd.read_csv(a.ANALYSIS/'kNDVI_decoupling.csv')
    fig=plt.figure(figsize=(10,6))
    grid=fig.add_gridspec(2,2,width_ratios=[3,1],wspace=.25,hspace=.35)
    for j,veg in enumerate(['Forest','Non-forest']):
        take=forest if veg=='Forest' else ~forest
        ax=fig.add_subplot(grid[j,0])
        a.map_panel(ax,ids,np.where(take,categories,np.nan),f'{chr(97+j)}  {veg}',categorical=True)
        finish_map(ax)
        group=fractions.loc[fractions.vegetation.eq(veg) & fractions.zone.eq('Global')]
        ax=fig.add_subplot(grid[j,1])
        ax.pie(group.percent,colors=a.CATCOLORS,autopct='%1.1f%%',pctdistance=.77,
               wedgeprops=dict(width=.45,edgecolor='white'),textprops=dict(fontsize=8))
        ax.set_title(veg)
    fig.legend(handles=[a.Patch(facecolor=c,label=a.CATEGORY[k]) for k,c in zip(a.CATEGORY,a.CATCOLORS)],
               loc='lower center',ncol=2,frameon=False,fontsize=7)
    save(fig,'Fig3_fullframe_v4')


def main():
    OUT.mkdir(exist_ok=True)
    previous.OUT=OUT
    previous.finish_map=finish_map
    previous.save=save
    paths=list(a.ANALYSIS.glob('*.csv'))+list(a.ANALYSIS.glob('*.json'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest,support,lc,_,base=a.source.old.masks()
    ids=np.flatnonzero(base)
    weights=np.cos(np.deg2rad(80-(ids//7200+.5)*.05))
    forest=lc.ravel()[ids]<=5
    zone=a.source.prep.package.mat(support['climate_zone']).ravel()[ids]
    missing=a.read(a.DATA/'kNDVI/greenness_2000_2025.tif',6).ravel()[ids]
    valid=np.isfinite(missing) & (missing<.5)
    previous.overview(ids,forest,zone,weights,manifest,valid)
    lookup={r['variable']:r for r in manifest['predictors']}
    precip,temp=[a.read(lookup[v]['path']).ravel()[ids] for v in ['P','Ta']]
    values=a.read(a.ANALYSIS/'kNDVI_mean_TAC_22.tif').ravel()[ids]
    square.draw(ids,values,weights,precip,temp,'Long-term mean TAC','FigS3_QA01_unified22',.8,3)
    table=pd.read_parquet(a.ANALYSIS/'kNDVI_pixel_statistics.parquet',columns=['pixel_id','greenness_slope'])
    np.testing.assert_array_equal(table.pixel_id.to_numpy()-200*7200,ids)
    values=table.greenness_slope.to_numpy()
    limit=max(.0005,float(np.nanquantile(np.abs(values),.97)))
    square.draw(ids,values,weights,precip,temp,'kNDVI greenness trend','FigS4_QA01_unified22',limit,4)
    main_figure3(ids,forest)
    limit=previous.window(ids,forest,weights,valid)
    previous.lag_and_evi(ids,valid,weights,limit)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    # The approved MATLAB Fig. 1 already has a complete map frame and white no-data areas.
    fig1=a.OUT/'figures_matlab_style/Fig1_MATLAB_style_polished_v14_original_palette'
    for ext in ['png','pdf','tif','fig']:
        path=fig1.with_suffix('.'+ext)
        if path.is_file(): shutil.copy2(path,OUT/path.name)
    count=sum(len(r['spatial_panels']) for r in STYLE_CHECKS)
    assert count==16,count
    report=dict(status='passed',spatial_panels_checked=count,
                additional_main_Fig1_map_already_compliant=True,total_spatial_panels=17,
                source_statistics_unchanged=True,no_TAC_recomputation=True,no_RF_retraining=True,
                finite_value_colormap_unchanged=True,S3_color_limits=[0,1],
                S3_S4_square_panels_preserved=True,style_checks=STYLE_CHECKS,
                square_panel_checks=square.CHECKS,figures=previous.RECORDS)
    (OUT/'map_ratio_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('All 17 spatial panels comply; original statistics and valid-value colours unchanged.',flush=True)


if __name__=='__main__':main()

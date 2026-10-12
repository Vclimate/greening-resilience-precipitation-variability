"""Adjust only S3 colour limits and square S3/S4 lower panels."""
import hashlib
import json
from pathlib import Path

import redraw_SI_spatial_ratio_20261006 as previous
from matplotlib.colors import Normalize

a=previous.a
np,pd,plt=a.np,a.pd,a.plt
OUT=previous.OUT.parent/'spatial_ratio_v3'
CHECKS=[]


def draw(ids,values,weights,precip,temp,title,stem,limit,number):
    fig=plt.figure(figsize=(8.5,7.8),layout='constrained')
    grid=fig.add_gridspec(2,1,height_ratios=[1.08,1],hspace=.055)
    map_ax=fig.add_subplot(grid[0])
    image=previous.map_panel(map_ax,ids,values,'a  '+title,limit)
    is_mean=number==3
    lower_count=int((np.isfinite(values) & (values<0)).sum()) if is_mean else 0
    if is_mean:
        image.set_norm(Normalize(vmin=0,vmax=1))
    map_cb=fig.colorbar(image,ax=map_ax,orientation='horizontal',fraction=.035,
                       pad=.085,shrink=.52,aspect=30,
                       extend='min' if lower_count else 'neither')
    map_cb.set_label('TAC' if is_mean else r'yr$^{-1}$',labelpad=3)
    map_cb.ax.tick_params(length=3,pad=2,labelsize=8.5)
    if is_mean:
        map_cb.set_ticks(np.linspace(0,1,6))

    # A dedicated colourbar column keeps both data panels the same square size.
    bottom=grid[1].subgridspec(1,3,width_ratios=[1,1,.055],wspace=.12)
    lat_ax=fig.add_subplot(bottom[0])
    bins=pd.read_csv(a.ANALYSIS/f'{stem}_latitude.csv')
    for veg in ['Forest','Non-forest']:
        g=bins.loc[bins.vegetation.eq(veg)]
        lat_ax.plot(g['mean'],g.latitude,color=a.COLORS[veg],lw=1.3,label=veg)
        lat_ax.fill_betweenx(g.latitude,g['mean']-g.ci95_half,g['mean']+g.ci95_half,
                             color=a.COLORS[veg],alpha=.2)
    lat_ax.axvline(0,color='#777777',lw=.5)
    lat_ax.set(ylabel='Latitude (degrees)',xlabel='TAC' if is_mean else r'kNDVI trend (yr$^{-1}$)',
               ylim=(-60,80))
    lat_ax.set_title('b  Latitudinal profile',loc='left',fontweight='bold')
    lat_ax.legend(frameon=False,fontsize=8,loc='best')
    lat_ax.set_box_aspect(1)

    climate_ax=fig.add_subplot(bottom[1])
    bins2,mean,_=a.climate_grid(values,precip,temp,weights)
    lo,hi=(0,1) if is_mean else (-limit,limit)
    mesh=climate_ax.pcolormesh(*bins2,mean.T,cmap='PiYG_r',vmin=lo,vmax=hi,shading='flat')
    climate_ax.set(xlabel=r'Precipitation (mm yr$^{-1}$)',ylabel='Temperature (degrees C)')
    climate_ax.set_title('c  Climate gradients',loc='left',fontweight='bold')
    climate_ax.set_box_aspect(1)
    climate_cb=fig.colorbar(mesh,cax=fig.add_subplot(bottom[2]),
                           extend='min' if is_mean and np.any(mean[np.isfinite(mean)]<0) else 'neither')
    climate_cb.ax.tick_params(labelsize=8.5,length=3,pad=2)
    if is_mean:
        climate_cb.set_ticks(np.linspace(0,1,6))

    fig.canvas.draw()
    boxes=[ax.get_window_extent() for ax in [lat_ax,climate_ax]]
    ratios=[box.width/box.height for box in boxes]
    np.testing.assert_allclose(ratios,[1,1],rtol=1e-7)
    np.testing.assert_allclose(boxes[0].width,boxes[1].width,rtol=1e-7)
    if is_mean:
        assert image.get_clim()==mesh.get_clim()==(0.,1.)
    CHECKS.append(dict(figure=number,panel_b_and_c_box_aspect_ratios=ratios,
                      panels_b_and_c_same_physical_size=True,
                      color_limits_a_and_c=[lo,hi],negative_mean_TAC_pixel_count=lower_count,
                      negative_TAC_not_removed=True,below_zero_indicated_by_colorbar_extension=bool(lower_count)))
    previous.save(fig,f'FigS{number}_map_ratio_v3')


def main():
    OUT.mkdir(exist_ok=True)
    previous.OUT=OUT
    files=[a.ANALYSIS/n for n in ['kNDVI_mean_TAC_22.tif','kNDVI_pixel_statistics.parquet',
                                 'FigS3_QA01_unified22_latitude.csv','FigS4_QA01_unified22_latitude.csv']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest,_,_,_,base=a.source.old.masks()
    ids=np.flatnonzero(base)
    weights=np.cos(np.deg2rad(80-(ids//7200+.5)*.05))
    lookup={r['variable']:r for r in manifest['predictors']}
    precip,temp=[a.read(lookup[v]['path']).ravel()[ids] for v in ['P','Ta']]
    values=a.read(a.ANALYSIS/'kNDVI_mean_TAC_22.tif').ravel()[ids]
    draw(ids,values,weights,precip,temp,'Long-term mean TAC','FigS3_QA01_unified22',.8,3)
    table=pd.read_parquet(a.ANALYSIS/'kNDVI_pixel_statistics.parquet',columns=['pixel_id','greenness_slope'])
    np.testing.assert_array_equal(table.pixel_id.to_numpy()-200*7200,ids)
    values=table.greenness_slope.to_numpy()
    limit=max(.0005,float(np.nanquantile(np.abs(values),.97)))
    draw(ids,values,weights,precip,temp,'kNDVI greenness trend','FigS4_QA01_unified22',limit,4)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    record=dict(status='passed',source_statistics_unchanged=True,no_TAC_recomputation=True,
                no_RF_retraining=True,checks=CHECKS,figures=previous.RECORDS)
    (OUT/'map_ratio_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(CHECKS,indent=2),flush=True)


if __name__=='__main__':main()

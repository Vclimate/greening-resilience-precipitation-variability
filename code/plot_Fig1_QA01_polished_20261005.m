% Fig. 1 only: same QA01 arrays, compact manuscript layout, explicit display scales.
clear; close all;
scriptDir = fileparts(mfilename('fullpath'));
outDir = fullfile(scriptDir,'..','output','QA01_unified22_20261005','figures_matlab_style');
inputPath = fullfile(outDir,'plot_inputs.mat');
D = load(inputPath);
assert(isequal(double(D.years(:)),(2004:2025)'));
intervalPath = fullfile(outDir,'Fig1_original_method_intervals.mat');
J = load(intervalPath);
colors = [.48 .48 .48; D.cmap(1,:); D.cmap(end,:)];
limit = .012;
mapColors = D.cmap(round(linspace(1,size(D.cmap,1),12)),:);
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial', ...
    'defaultAxesFontSize',7,'defaultTextFontSize',7,'defaultAxesLineWidth',.55, ...
    'defaultFigureColor','w');
f = figure('Visible','off','Units','centimeters','Position',[1 1 17.8 12.5], ...
    'PaperUnits','centimeters','PaperSize',[17.8 12.5], ...
    'PaperPosition',[0 0 17.8 12.5],'PaperPositionMode','manual','Renderer','painters');

%% a: global map, with the same smoothing and valid-pixel mask as before.
axMap = makeAxes(f,[.055 .525 .695 .385]);
imagesc(axMap,D.longitude,D.latitude,D.trend_map,'AlphaData',isfinite(D.trend_map));
set(axMap,'YDir','normal','XTick',[],'YTick',[],'Box','on', ...
    'XColor',[.2 .2 .2],'YColor',[.2 .2 .2],'LineWidth',.6);
xlim(axMap,[-180 180]); ylim(axMap,[-60 80]);
colormap(axMap,mapColors); clim(axMap,[-limit limit]);
for k = 1:numel(D.coast)
    c = D.coast{k};
    plot(axMap,c(:,1),c(:,2),'Color',[.55 .55 .55],'LineWidth',.28);
end
heading(axMap,'a','Spatial TAC trends');

%% b: original 101-row smoothing; interval half-widths are not enlarged.
latitudeSource = intervalPath;
latitudeWindow = 101;
latitudeDisplayMultiplier = 1;
axLat = makeAxes(f,[.835 .525 .135 .385]);
xline(axLat,0,'Color',[.65 .65 .65],'LineWidth',.55,'HandleVisibility','off');
latBounds = [];
for i = 1:2
    a = squeeze(J.lat_values_original(:,i,:));
    good = all(isfinite(a),2);
    x = a(:,2)*1000; e = a(:,3)*1000*latitudeDisplayMultiplier;
    horizontalBand(axLat,a(:,1),x-e,x+e,colors(i+1,:));
    x(~good) = nan;
    plot(axLat,x,a(:,1),'Color',colors(i+1,:),'LineWidth',1.15);
    latBounds = [latBounds; x(good)-e(good); x(good)+e(good)]; %#ok<AGROW>
end
bound = ceil(max(abs(latBounds))*1.03*2)/2;
xlim(axLat,[-bound bound]); ylim(axLat,[-60 80]);
latitudeTickStep = 5*ceil(bound/10);
xticks(axLat,[-latitudeTickStep 0 latitudeTickStep]); yticks(axLat,[-60 -20 20 60 80]);
xlabel(axLat,{'\deltaTAC','(10^{-3} year^{-1})'},'FontSize',6.5);
ylabel(axLat,'Latitude (^\circ)','FontSize',7);
heading(axLat,'b','Latitude');

%% Shared map/climate color scale. Triangles show the explicitly clipped tails.
colorbarPosition = [.1825 .475 .44 .016];
axScale = axes(f,'Position',colorbarPosition,'FontName','Arial', ...
    'FontSize',6.5,'Box','off','TickDir','out','LineWidth',.4);
hold(axScale,'on');
edges = linspace(-limit*1000,limit*1000,13);
for k = 1:12
    patch(axScale,[edges(k) edges(k+1) edges(k+1) edges(k)], ...
        [0 0 1 1],mapColors(k,:),'EdgeColor','none');
end
colormap(axScale,mapColors); clim(axScale,[-limit limit]*1000);
tail = .65;
patch(axScale,[-limit*1000-tail -limit*1000 -limit*1000],[.5 0 1], ...
    mapColors(1,:),'EdgeColor','none');
patch(axScale,[limit*1000+tail limit*1000 limit*1000],[.5 0 1], ...
    mapColors(end,:),'EdgeColor','none');
xlim(axScale,[-limit*1000-tail limit*1000+tail]); ylim(axScale,[0 1]);
yticks(axScale,[]); xticks(axScale,-12:4:12);
axScale.YColor = 'none';
xlabel(axScale,'\deltaTAC (10^{-3} year^{-1})','FontSize',6.5);

%% c: retain original precipitation/temperature bins; concise labels.
squareHeight = .27;
squareWidth = squareHeight*12.5/17.8;
axClimate = makeAxes(f,[.2475-squareWidth/2 .115 squareWidth squareHeight]);
pbaspect(axClimate,[1 1 1]);
xc = (D.climate_precip_edges(1:end-1)+D.climate_precip_edges(2:end))/2;
yc = (D.climate_temp_edges(1:end-1)+D.climate_temp_edges(2:end))/2;
imagesc(axClimate,xc,yc,D.climate_grid,'AlphaData',isfinite(D.climate_grid));
set(axClimate,'YDir','normal','Box','on','LineWidth',.6);
colormap(axClimate,mapColors); clim(axClimate,[-limit limit]);
xlim(axClimate,[0 5000]); ylim(axClimate,[-10 30]);
xticks(axClimate,[0 2500 5000]); yticks(axClimate,[-10 0 10 20 30]);
axClimate.XTickLabelRotation = 0;
axClimate.XAxis.Exponent = 0;
xlabel(axClimate,{'Annual precipitation','(mm year^{-1})'},'FontSize',7);
ylabel(axClimate,'Annual temperature (^\circC)','FontSize',7);
heading(axClimate,'c','Climate space');

%% d: original intervals, explicitly enlarged fivefold for visualisation only.
displayMultiplier = 5;
axTime = makeAxes(f,[.505 .095 .465 .29]);
yline(axTime,0,':','Color',[.7 .7 .7],'LineWidth',.5,'HandleVisibility','off');
h = gobjects(3,1);
for i = 1:3
    value = J.tac_change_original(:,i)*1000;
    half = J.tac_half_original(:,i)*1000*displayMultiplier;
    band(axTime,D.years,value-half,value+half,colors(i,:));
    h(i) = plot(axTime,D.years,value,'Color',colors(i,:),'LineWidth',1.25);
end
xlim(axTime,D.years([1 end]));
xticks(axTime,[2004 2008 2012 2016 2020 2025]);
dBounds = [J.tac_change_original(:)-displayMultiplier*J.tac_half_original(:); ...
    J.tac_change_original(:)+displayMultiplier*J.tac_half_original(:)]*1000;
dRange = range(dBounds);
dLimits = 2*[floor((min(dBounds)-.08*dRange)/2) ceil((max(dBounds)+.30*dRange)/2)];
ylim(axTime,dLimits); yticks(axTime,-20:5:20);
axTime.YAxis.Exponent = 0;
xlabel(axTime,'Year','FontSize',7);
ylabel(axTime,'TAC change from 2004 (10^{-3})','FontSize',7);
heading(axTime,'d','Regional TAC');
for i = 2:3
    slope = D.hac4(1,i,1);
    pHAC = D.hac4(1,i,4);
    assert(isfinite(pHAC) && pHAC>=0 && pHAC<=1);
    fit = polyfit(double(D.years(:)),J.tac_change_original(:,i),1);
    assert(abs(slope-fit(1))<1e-12);
    text(axTime,.025,.96-(i-2)*.115, ...
        sprintf('%s: \\deltaTAC = %.2f \\times10^{-4} year^{-1}; P_{HAC} = %.3f', ...
            D.vegetation{i},slope*1e4,pHAC), ...
        'Units','normalized','VerticalAlignment','top','Color',colors(i,:), ...
        'FontSize',6.5,'FontWeight','normal');
end
text(axTime,.025,.735,'OLS trend; two-sided HAC test (lag = 4)', ...
    'Units','normalized','VerticalAlignment','top','Color',[.35 .35 .35],'FontSize',6);
lg = legend(axTime,h,D.vegetation,'Orientation','horizontal','NumColumns',3, ...
    'FontSize',7,'Box','off');
lg.ItemTokenSize = [13 7];
lg.Units = 'normalized'; lg.Position = [.49 .955 .48 .025];
annotation(f,'textbox',[.07 .004 .9 .018], ...
    'String','Panel b: original interval widths. Panel d: 95% interval half-widths expanded 5-fold for display only.', ...
    'FontName','Arial','FontSize',5.8,'LineStyle','none','Margin',0,'HorizontalAlignment','center');

drawnow;
name = 'Fig1_MATLAB_style_polished_v14_original_palette';
print(f,fullfile(outDir,[name '.png']),'-dpng','-r600');
print(f,fullfile(outDir,[name '.tif']),'-dtiff','-r600');
print(f,fullfile(outDir,[name '.pdf']),'-dpdf','-r600','-painters');
savefig(f,fullfile(outDir,[name '.fig']));
metadata = struct('same_input_arrays',true,'source',inputPath, ...
    'interval_source',intervalPath,'original_interval_formulas_restored',true, ...
    'map_frame',true,'climate_frame',true,'color_levels',12,'climate_panel_width_height_ratio',1, ...
    'palette','Original green-neutral-magenta palette, identical to v9 colorbar', ...
    'palette_reference','Fig1_MATLAB_style_polished_v9_colorbar.fig', ...
    'map_RGB_colors',mapColors,'trajectory_RGB_colors',colors, ...
    'colorbar_position',colorbarPosition,'colorbar_label_font_size',6.5, ...
    'map_climate_color_limits',[-limit limit],'colorbar_triangles','Clipped values outside displayed limits', ...
    'map_clipped_fraction',sum(abs(D.trend_map(isfinite(D.trend_map)))>limit)/sum(isfinite(D.trend_map(:))), ...
    'climate_clipped_fraction',sum(abs(D.climate_grid(isfinite(D.climate_grid)))>limit)/sum(isfinite(D.climate_grid(:))), ...
    'latitudinal_axis_display_units','10^-3 year^-1','latitudinal_axis_limits',[-bound bound], ...
    'regional_anomaly_display_units','10^-3','regional_baseline_reference_year',2004, ...
    'TAC_year_label_rule','Window-end calendar year minus one; 2025-01-01 represents 2024', ...
    'TAC_year_labels',double(D.years(:))','window_end_years',double(D.years(:))'+1, ...
    'regional_axis_limits',dLimits,'latitude_interval_display_multiplier',latitudeDisplayMultiplier, ...
    'latitude_source',latitudeSource,'latitude_smoothing_rows',latitudeWindow, ...
    'latitude_smoothing_center_span_degrees',(latitudeWindow-1)*.05, ...
    'uncertainty_multiplier',displayMultiplier,'actual_interval_multiplier',1, ...
    'display_note','Panel b interval widths unchanged; panel d shading fivefold enlarged for display only', ...
    'forest_slope',D.hac4(1,2,1),'nonforest_slope',D.hac4(1,3,1), ...
    'forest_HAC4_p',D.hac4(1,2,4),'nonforest_HAC4_p',D.hac4(1,3,4), ...
    'trend_significance_method','Two-sided t test, Newey-West HAC, lag 4, Bartlett kernel, finite-sample correction', ...
    'HAC_applies_to','OLS slope standard errors and P values, not plotted pixel-mean shading', ...
    'old_figures_overwritten',false);
fid = fopen(fullfile(outDir,'Fig1_polished_v14_display_notes.json'),'w','n','UTF-8');
assert(fid>0); fprintf(fid,'%s',jsonencode(metadata,PrettyPrint=true)); fclose(fid);
close(f);
fprintf('Polished Fig1 only; input data unchanged; color clipping %.2f%% map / %.2f%% climate.\n', ...
    100*metadata.map_clipped_fraction,100*metadata.climate_clipped_fraction);

function ax = makeAxes(f,pos)
ax = axes(f,'Position',pos,'Box','off','TickDir','out','Layer','top', ...
    'XColor',[.2 .2 .2],'YColor',[.2 .2 .2],'Color','w','TickLength',[.015 .015]);
hold(ax,'on');
end

function heading(ax,letter,label)
text(ax,-.07,1.045,letter,'Units','normalized','FontWeight','bold', ...
    'FontSize',9,'VerticalAlignment','bottom');
labelX = .035;
if strcmp(letter,'b'), labelX = .16; end
text(ax,labelX,1.052,label,'Units','normalized','FontWeight','normal', ...
    'FontSize',8,'VerticalAlignment','bottom');
end

function band(ax,x,low,high,color)
x = x(:); low = low(:); high = high(:);
assert(all(low<=high));
fill(ax,[x;flipud(x)],[low;flipud(high)],color, ...
    'FaceAlpha',.15,'EdgeColor','none','HandleVisibility','off');
end

function horizontalBand(ax,y,low,high,color)
good = isfinite(y)&isfinite(low)&isfinite(high);
starts = find(diff([false;good])==1);
stops = find(diff([good;false])==-1);
for k = 1:numel(starts)
    index = starts(k):stops(k);
    if numel(index)<2, continue; end
    fill(ax,[low(index);flipud(high(index))],[y(index);flipud(y(index))],color, ...
        'FaceAlpha',.22,'EdgeColor','none','HandleVisibility','off');
end
end

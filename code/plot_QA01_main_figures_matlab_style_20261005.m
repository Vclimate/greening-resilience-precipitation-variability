% Original manuscript visual conventions; all numbers come from saved QA01 results.
clear; close all;
scriptDir = fileparts(mfilename('fullpath'));
outDir = fullfile(scriptDir, '..', 'output', 'QA01_unified22_20261005', 'figures_matlab_style');
D = load(fullfile(outDir, 'plot_inputs.mat'));
set(groot, 'defaultAxesFontName', 'Arial', 'defaultTextFontName', 'Arial', ...
    'defaultAxesFontSize', 7, 'defaultTextFontSize', 7, ...
    'defaultAxesLineWidth', .5, 'defaultFigureColor', 'w');
vegetationColors = [.6 .6 .6; D.cmap(1,:); D.cmap(end,:)];

%% Figure 1: map + latitude + climate space + time series.
f = canvas(17,14);
ax = panel(f,[.065 .55 .71 .37],'a');
mapPlot(ax,D,D.trend_map,D.cmap,[-.016 .016]);
title(ax,'kNDVI TAC trend','FontWeight','normal');
cb = colorbar(ax,'southoutside');
cb.Position = [.14 .505 .52 .014];
cb.Ticks = [-.016 -.008 0 .008 .016];
cb.Label.String = '\deltaTAC (year^{-1})';
ax = panel(f,[.835 .55 .145 .37],'b');
hold(ax,'on');
xline(ax,0,'k-','LineWidth',.5,'HandleVisibility','off');
for i = 1:2
    a = squeeze(D.lat_values(:,i,:));
    a = a(all(isfinite(a),2),:);
    fill(ax,[a(:,2)-a(:,3);flipud(a(:,2)+a(:,3))], ...
        [a(:,1);flipud(a(:,1))],vegetationColors(i+1,:), ...
        'FaceAlpha',.18,'EdgeColor','none','HandleVisibility','off');
    plot(ax,a(:,2),a(:,1),'Color',vegetationColors(i+1,:),'LineWidth',1, ...
        'DisplayName',D.vegetation{i+1});
end
xlim(ax,[-.03 .03]); ylim(ax,[-60 80]);
xticks(ax,[-.03 0 .03]); yticks(ax,[-60 -20 20 60 80]);
xlabel(ax,'\deltaTAC (year^{-1})'); ylabel(ax,'Latitude (^\circ)');
legend(ax,'Location','northwest','FontSize',6,'Box','off');
ax = panel(f,[.11 .105 .30 .285],'c');
xc = (D.climate_precip_edges(1:end-1)+D.climate_precip_edges(2:end))/2;
yc = (D.climate_temp_edges(1:end-1)+D.climate_temp_edges(2:end))/2;
imagesc(ax,xc,yc,D.climate_grid,'AlphaData',isfinite(D.climate_grid));
set(ax,'YDir','normal'); colormap(ax,D.cmap); clim(ax,[-.016 .016]);
xlim(ax,[0 5000]); ylim(ax,[-10 30]); xticks(ax,[0 2500 5000]); yticks(ax,[-10 0 10 20 30]);
xlabel(ax,{'Average annual precipitation','(mm year^{-1})'});
ylabel(ax,{'Average annual temperature','(^\circC)'});
ax = panel(f,[.51 .105 .47 .285],'d');
hold(ax,'on');
h = gobjects(3,1);
for i = 1:3
    y = D.tac_mean(:,1,i); e = D.tac_half(:,1,i);
    y = y-mean(y);
    band(ax,D.years,y-e,y+e,vegetationColors(i,:));
    h(i) = plot(ax,D.years,y,'Color',vegetationColors(i,:),'LineWidth',1.2);
end
configureYears(ax,D.years); ylabel(ax,'TAC anomaly'); title(ax,'Regional TAC','FontWeight','normal');
legend(ax,h,D.vegetation,'Location','northwest','Box','off','FontSize',6);
exportFigure(f,outDir,'Fig1_MATLAB_style');

%% Figure 2: original five stacked climate-zone panels, forest/non-forest columns.
f = canvas(17,15.5);
for v = 1:2
    for z = 1:5
        y = .81-(z-1)*.158;
        ax = panel(f,[.115+(v-1)*.49 y .365 .116],char('a'+(v-1)*5+z-1));
        hold(ax,'on');
        value = D.tac_mean(:,z,v+1); e = D.tac_half(:,z,v+1);
        band(ax,D.years,value-e,value+e,D.climate_colors(z,:));
        plot(ax,D.years,value,'Color',D.climate_colors(z,:),'LineWidth',1.15);
        configureYears(ax,D.years);
        ylabel(ax,{D.zones{z},'TAC'});
        if z ~= 5
            ax.XTickLabel = [];
            xlabel(ax,'');
        end
        if z == 1
            title(ax,D.vegetation{v+1},'FontWeight','bold','FontSize',8);
        end
    end
end
exportFigure(f,outDir,'Fig2_MATLAB_style');

%% Figure 3: maps, global donut insets, and four climate-zone donuts.
f = canvas(17,14);
for v = 1:2
    y = .56-(v-1)*.43;
    ax = panel(f,[.035 y .74 .33],char('a'+v-1));
    ax.Children(1).Position(1) = 0;
    if v == 1, values = D.category_forest; else, values = D.category_nonforest; end
    mapPlot(ax,D,values,D.category_colors,[.5 4.5]);
    title(ax,D.vegetation{v+1},'FontWeight','normal');
    inset = axes(f,'Position',[.09 y+.005 .17 .16]);
    donut(inset,squeeze(D.fractions(1,v,:)),D.category_colors,6.5);
    for z = 2:5
        inset = axes(f,'Position',[.82 y+.255-(z-2)*.085 .16 .083]);
        donut(inset,squeeze(D.fractions(z,v,:)),D.category_colors,5.5);
        text(inset,0,1.65,D.zones{z}, ...
            'HorizontalAlignment','center','FontSize',6);
    end
end
labels = {'Greening + resilience decline','Browning + resilience decline', ...
          'Browning + resilience increase','Greening + resilience increase'};
for k = 1:4
    x = .06+mod(k-1,2)*.49; y = .062-floor((k-1)/2)*.027;
    annotation(f,'rectangle',[x y .017 .016],'FaceColor',D.category_colors(k,:), 'EdgeColor','none');
    annotation(f,'textbox',[x+.022 y-.003 .45 .022],'String',labels{k}, ...
        'LineStyle','none','FontName','Arial','FontSize',6.5,'Margin',0);
end
exportFigure(f,outDir,'Fig3_MATLAB_style');

%% Figure 4: original top-ten gradient bars and six PDPs, two vegetation groups.
f = canvas(17,21);
for v = 1:2
    base = .59-(v-1)*.42;
    ax = panel(f,[.16 base .19 .315],char('a'+(v-1)*7));
    hold(ax,'on');
    r = D.rank_values{v};
    if v == 1, colors = D.cmap20(1:10,:); else, colors = flipud(D.cmap20(11:20,:)); end
    b = barh(ax,1:10,r(:,1),'FaceColor','flat','BarWidth',.55);
    b.CData = colors;
    errorbar(ax,r(:,1),1:10,r(:,1)-r(:,2),r(:,3)-r(:,1),'horizontal', ...
        'LineStyle','none','Color',[.15 .15 .15],'CapSize',2,'LineWidth',.6);
    set(ax,'YDir','reverse','YTick',1:10,'YTickLabel',D.rank_labels{v});
    xlim(ax,[0 max(r(:,3))*1.12]); ylim(ax,[.3 10.7]);
    xlabel(ax,{'Standardized OOB','importance'});
    title(ax,D.vegetation{v+1},'FontWeight','bold','FontSize',8);
    p = D.performance(v,:);
    text(ax,.5,-.205,sprintf('Accuracy %.1f%% [%.1f, %.1f]',p), ...
        'Units','normalized','HorizontalAlignment','center','FontSize',6);
    for j = 1:6
        row = floor((j-1)/3); col = mod(j-1,3);
        ax = panel(f,[.425+col*.194 base+.22-row*.22 .143 .105],char('a'+(v-1)*7+j));
        hold(ax,'on');
        a = D.curve_values{v,j};
        unit = D.curve_units{v,j};
        if strcmp(D.curve_labels{v,j},'\deltakNDVI')
            a(:,1) = a(:,1)*1000;
            unit = '(10^{-3} year^{-1})';
        end
        band(ax,a(:,1),a(:,3),a(:,4),D.curve_colors(2,:));
        band(ax,a(:,1),100-a(:,4),100-a(:,3),D.curve_colors(1,:));
        plot(ax,a(:,1),a(:,2),'-','Color',D.curve_colors(2,:),'LineWidth',1.1);
        plot(ax,a(:,1),100-a(:,2),'--','Color',D.curve_colors(1,:),'LineWidth',1.1);
        bounds = [a(:,3);a(:,4);100-a(:,3);100-a(:,4)];
        pad = max(.2,.10*range(bounds));
        ylim(ax,[max(0,min(bounds)-pad) min(100,max(bounds)+pad)]);
        xlim(ax,a([1 end],1)); ytickformat(ax,'%.1f%%'); ax.YAxis.Exponent = 0;
        ax.XTick = linspace(a(1,1),a(end,1),3);
        if max(abs(a(:,1))) >= 100
            xticklabels(ax,compose('%.0f',ax.XTick));
        else
            xticklabels(ax,compose('%.2g',ax.XTick));
        end
        ax.XTickLabelRotation = 0;
        ax.XAxis.Exponent = 0;
        ax.YTick = linspace(ax.YLim(1),ax.YLim(2),3);
        title(ax,D.curve_labels{v,j},'FontWeight','normal','FontSize',7,'Units','normalized','Position',[.5 1.38 0]);
        score = squeeze(D.importance_values(v,j,:));
        text(ax,.5,1.05,sprintf('I = %.2f [%.2f, %.2f]',score), ...
            'Units','normalized','HorizontalAlignment','center','FontSize',5.5);
        xlabel(ax,unit,'FontSize',6);
        if col == 0, ylabel(ax,{'Classification','probability'},'FontSize',6); end
        ax.FontSize = 6;
    end
end
annotation(f,'line',[.12 .16],[.062 .062],'Color',D.curve_colors(1,:),'LineStyle','--');
annotation(f,'textbox',[.17 .049 .26 .026],'String','Resilience increase','LineStyle','none','Margin',0,'FontSize',7);
annotation(f,'line',[.52 .56],[.062 .062],'Color',D.curve_colors(2,:));
annotation(f,'textbox',[.57 .049 .3 .026],'String','Resilience decline','LineStyle','none','Margin',0,'FontSize',7);
annotation(f,'textbox',[.09 .025 .88 .025],'String','Shading and brackets: 2.5th-97.5th percentiles across 30 refits.', ...
    'LineStyle','none','Margin',0,'FontSize',6,'HorizontalAlignment','center');
exportFigure(f,outDir,'Fig4_MATLAB_style');
fprintf('Completed Fig. 1-4 from unchanged QA01 results: %s\n',outDir);

function f = canvas(w,h)
f = figure('Visible','off','Units','centimeters','Position',[1 1 w h], ...
    'PaperUnits','centimeters','PaperSize',[w h],'PaperPosition',[0 0 w h], ...
    'PaperPositionMode','manual','Renderer','painters');
end

function ax = panel(f,pos,letter)
ax = axes(f,'Position',pos,'Box','on','TickDir','out','Layer','top');
hold(ax,'on');
text(ax,-.075,1.035,letter,'Units','normalized','FontWeight','bold', ...
    'FontSize',8,'VerticalAlignment','bottom');
end

function mapPlot(ax,D,a,cmap,limits)
imagesc(ax,D.longitude,D.latitude,a,'AlphaData',isfinite(a));
set(ax,'YDir','normal','XTick',[],'YTick',[],'Color','w');
xlim(ax,[-180 180]); ylim(ax,[-60 80]); colormap(ax,cmap); clim(ax,limits);
hold(ax,'on');
for k = 1:numel(D.coast)
    c = D.coast{k}; plot(ax,c(:,1),c(:,2),'Color',[.35 .35 .35],'LineWidth',.3);
end
end

function band(ax,x,low,high,color)
x = x(:); low = low(:); high = high(:);
assert(all(low <= high));
fill(ax,[x;flipud(x)],[low;flipud(high)],color, ...
    'FaceAlpha',.18,'EdgeColor','none','HandleVisibility','off');
end

function configureYears(ax,years)
xlim(ax,years([1 end])); xticks(ax,[2004 2008 2012 2016 2020 2025]); xlabel(ax,'Reference year');
ax.YAxis.Exponent = 0;
end

function donut(ax,values,colors,fontSize)
assert(abs(sum(values)-100) < 1e-6);
hold(ax,'on'); start = pi/2;
for k = 1:4
    finish = start-values(k)/100*2*pi;
    theta = linspace(start,finish,max(3,ceil(values(k)*3)));
    patch(ax,[cos(theta) .55*cos(fliplr(theta))], ...
        [sin(theta) .55*sin(fliplr(theta))],colors(k,:), ...
        'EdgeColor','w','LineWidth',.4);
    mid = (start+finish)/2;
    positions = [1.22 .98; 1.22 -1.18; -1.22 -1.18; -1.22 .98];
    label = positions(k,:);
    plot(ax,[cos(mid) label(1)*.78],[sin(mid) label(2)*.78], ...
        'Color',colors(k,:)*.65,'LineWidth',.4);
    text(ax,label(1),label(2),sprintf('%.1f%%',values(k)), ...
        'HorizontalAlignment','center','FontSize',fontSize);
    start = finish;
end
axis(ax,'off'); xlim(ax,[-2.15 2.15]); ylim(ax,[-1.5 1.9]);
% Keep rings circular while allowing room for four non-overlapping labels.
daspect(ax,[1 1 1]);
end

function exportFigure(f,outDir,name)
drawnow;
print(f,fullfile(outDir,[name '.png']),'-dpng','-r600');
print(f,fullfile(outDir,[name '.tif']),'-dtiff','-r600');
print(f,fullfile(outDir,[name '.pdf']),'-dpdf','-r600','-painters');
close(f);
end

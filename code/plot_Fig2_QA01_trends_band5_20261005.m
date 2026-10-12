% Five-region FIG2: original paired intervals, fivefold display width, actual OLS slopes.
clear; close all;
scriptDir = fileparts(mfilename('fullpath'));
outDir = fullfile(scriptDir,'..','output','QA01_unified22_20261005','figures_matlab_style');
D = load(fullfile(outDir,'plot_inputs.mat'));
assert(isequal(double(D.years(:)),(2004:2025)'));
J = load(fullfile(outDir,'Fig2_original_method_intervals.mat'));
displayMultiplier = 5;
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial', ...
    'defaultAxesFontSize',7,'defaultTextFontSize',7,'defaultAxesLineWidth',.5, ...
    'defaultFigureColor','w');
f = figure('Visible','off','Units','centimeters','Position',[1 1 17.8 16], ...
    'PaperUnits','centimeters','PaperSize',[17.8 16],'PaperPosition',[0 0 17.8 16], ...
    'PaperPositionMode','manual','Renderer','painters');
for v = 1:2
    for z = 1:5
        ax = axes(f,'Position',[.115+(v-1)*.49 .81-(z-1)*.154 .365 .114], ...
            'Box','off','TickDir','out','Layer','top');
        hold(ax,'on');
        value = J.tac_change(:,z,v+1)*1000;
        half = J.tac_half(:,z,v+1)*1000*displayMultiplier;
        fill(ax,[D.years(:);flipud(D.years(:))], ...
            [value-half;flipud(value+half)],D.climate_colors(z,:), ...
            'FaceAlpha',.20,'EdgeColor','none');
        plot(ax,D.years,value,'Color',D.climate_colors(z,:),'LineWidth',1.15);
        xlim(ax,D.years([1 end])); xticks(ax,[2004 2008 2012 2016 2020 2025]);
        bounds = [value-half;value+half];
        span = max(range(bounds),1);
        ylim(ax,[min(bounds)-.08*span max(bounds)+.35*span]);
        ax.YAxis.Exponent = 0;
        slope = D.hac4(z,v+1,1);
        pHAC = D.hac4(z,v+1,4);
        assert(isfinite(pHAC) && pHAC>=0 && pHAC<=1);
        if pHAC < .001
            pLabel = 'P_{HAC} < 0.001';
        else
            pLabel = sprintf('P_{HAC} = %.3f',pHAC);
        end
        fit = polyfit(double(D.years(:)),J.tac_change(:,z,v+1),1);
        assert(abs(slope-fit(1))<1e-12);
        text(ax,.02,.95,sprintf('\\deltaTAC = %.2f \\times10^{-4} year^{-1}; %s',slope*1e4,pLabel), ...
            'Units','normalized','VerticalAlignment','top','FontSize',6.5);
        text(ax,-.075,1.03,char('a'+(v-1)*5+z-1),'Units','normalized', ...
            'FontWeight','bold','FontSize',8,'VerticalAlignment','bottom');
        ylabel(ax,{D.zones{z},'\DeltaTAC (10^{-3})'},'FontSize',7);
        if z == 1
            title(ax,D.vegetation{v+1},'FontWeight','bold','FontSize',8);
        end
        if z ~= 5
            ax.XTickLabel = [];
        else
            xlabel(ax,'Year','FontSize',7);
        end
    end
end
annotation(f,'textbox',[.07 .082 .9 .025], ...
    'String','TAC year = window-end year - 1 (2025-01-01 represents 2024). Baseline: 2004.', ...
    'FontName','Arial','FontSize',6,'LineStyle','none','Margin',0,'HorizontalAlignment','center');
annotation(f,'textbox',[.07 .053 .9 .025], ...
    'String','Shading: calculated 95% interval half-widths expanded 5-fold for display only.', ...
    'FontName','Arial','FontSize',6,'LineStyle','none','Margin',0,'HorizontalAlignment','center');
annotation(f,'textbox',[.07 .024 .9 .025], ...
    'String','Significance: two-sided HAC test (lag = 4); OLS slopes over 2004-2025.', ...
    'FontName','Arial','FontSize',6,'LineStyle','none','Margin',0,'HorizontalAlignment','center');
drawnow;
name = 'Fig2_MATLAB_style_v5_HAC4';
print(f,fullfile(outDir,[name '.png']),'-dpng','-r600');
print(f,fullfile(outDir,[name '.tif']),'-dtiff','-r600');
print(f,fullfile(outDir,[name '.pdf']),'-dpdf','-r600','-painters');
savefig(f,fullfile(outDir,[name '.fig']));
metadata = struct('source','Fig2_original_method_intervals.mat', ...
    'display_multiplier',displayMultiplier,'actual_interval_multiplier',1, ...
    'global_matches_fig1',true,'OLS_slopes',D.hac4(:,2:3,1), ...
    'HAC4_p_values',D.hac4(:,2:3,4), ...
    'trend_significance_method','Two-sided t test, Newey-West HAC, lag 4, Bartlett kernel, finite-sample correction', ...
    'HAC_applies_to','OLS slope standard errors and P values, not plotted pixel-mean shading', ...
    'reference_years',[2004 2025],'relative_to_reference_year',2004, ...
    'TAC_year_label_rule','Window-end calendar year minus one; 2025-01-01 represents 2024', ...
    'TAC_year_labels',double(D.years(:))','window_end_years',double(D.years(:))'+1, ...
    'old_figures_overwritten',false,'display_note','Fivefold enlarged shading is not an actual 95% interval');
fid = fopen(fullfile(outDir,'Fig2_HAC4_display_notes.json'),'w','n','UTF-8');
assert(fid>0); fprintf(fid,'%s',jsonencode(metadata,PrettyPrint=true)); fclose(fid);
close(f);
fprintf('FIG2 exported with checked OLS slopes and explicitly labelled fivefold shading.\n');

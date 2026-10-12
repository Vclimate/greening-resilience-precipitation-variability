clc;clear; close all
%% DRIVERS
cd('F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005')
load("KNDVI_AC1-01-01_2000_2023_5year_window_stp","Xtrend_smooth","Xtrend")

%掩膜，去掉非森林草原地区、以及发生土地利用类型变化区域
load('005IBGP_SLC.mat', 'IBGP_SLC')
IGBP = IBGP_SLC;

% filenameIn = ['F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\IGBP_2022.mat'];
% load(filenameIn, 'IGBP')

[HFP_MASK,IGBP_MASK,W,pfts_CF,ifl_MASK,koppen_MASK,Tmed,Pmed,MDNw] = importmaps_qa_005(0.05,'005');
MASKmdn = MDNw < 0.5; %缺失KNDVI数值比例掩膜

%设置字体大小
Fsize = 6;
% Compute area计算画图面积大小
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%设置画布大小
figure('Units', 'centimeters', 'Position', [1, 9, 17, 14]);

% setting external limits 设置外部限值
xbox = 480;  %宽度
ybox = (xbox*(60+80))/360; %高度（宽度*相应比例）
spacex = 30; %子图X方向间距
spacey = 22; %子图Y方向间距
fig_pos = [20 240]; %起始绘图的左下角位置像素

cellsize = 0.05;
lat(:,1) = [-60+cellsize/2:cellsize:80-cellsize/2];
lon(1,:) = [-180+cellsize/2:cellsize:180-cellsize/2];
[LON,LAT] = meshgrid(lon,flipud(lat));
latmax = 80;
latmin = -60;
%设置图例间隔与图例
num_intervals = 256;
cmap = getPyPlot_cMap('PiYG_r', num_intervals);
cmap(110:146,:) = [];
% Setting plot position
xstart = fig_pos(1);
ystart = fig_pos(2);
coords = [xstart ystart xbox ybox];
ax_pos = coords;

% 加载数据
% Show map
%第1张图  -----------------------------------------------------------------------------------
%读取数据进行绘图

Xmap = Xtrend_smooth;
MASK = ((MASKmdn==1)&(isfinite(Xmap))&(isfinite(Tmed))&(isfinite(Pmed))&(IGBP~=0));

Xmap(MASK==0) = nan;
% 只保留80°N-60°S区域
Xmap = Xmap(201:3000,:);
% -------------------------------------------------------------------------------------------
%设置图例范围
clim = [-0.016, 0.016];
ax1 = axes('Unit','pixels','Position',ax_pos,'XColor','k','YColor','k');
pcolor(LON,LAT,Xmap); shading flat; axis auto;
hold on; geoshow('landareas.shp', 'facecolor', 'none')
set(gca,'xtick',[], 'ytick',[],'xticklabel','', 'yticklabel','')
axis([-180, 180, latmin, latmax]);
set(gca,'layer','top')
set(gca,'Fontsize',6)
colormap(gca,cmap)
caxis(clim)
title(ax1,'KNDVI TAC trend','Fontsize',6)
xlims = get(gca,'Xlim');
ylims = get(gca,'Ylim');
text(xlims(1),ylims(2),'a','HorizontalAlignment','left',...
    'VerticalAlignment','bottom',...
    'FontWeight','bold','Fontsize',6)

cbh = colorbar;
L = [-clim(2): clim(2)/2: clim(2)];

set(cbh, 'Ytick',L,'Fontsize',6,...
    'Position',get(cbh,'Position')-[0.65, -0.02, 0.02, 0.2],...
    'YAxisLocation','right')
title(cbh,'          \deltaTAC [year^{-1}]','Fontsize',6)
%第1张图附属子图  -----------------------------------------------------------------------------------
% Setting plot position
xstart = fig_pos(1)+xbox+spacex;
ystart = fig_pos(2);
xboxnew = xbox/5;
coords = [xstart ystart xboxnew ybox];
ax_pos = coords;

ax2 = axes('Unit','pixels','Position',ax_pos,...
    'XColor','k','YColor','k');

%设置颜色
%cmap = brewermap(8,'Set1');
col_1 = cmap(1,:);
col_2 = cmap(end,:);

col_lat = [col_1; col_2];

plot([0,0],[latmin, latmax],'-k','Linewidth',1)

IGBP = IGBP(201:3000,:);
for  i = 1:2
    Xtrend(MASK==0) = nan;
    Xmapi = Xtrend(201:3000,:); %用原数据绘图
    if i==1 % 森林区域
        Xmapi(IGBP>5) = nan;
    elseif i==2 % 草地
        Xmapi(IGBP<=5) = nan;
    elseif i==3 % trend in CSDnet
        Xmapi = Xmapi; %不做任何处理
    end

    Xmap_lat = nanmean(Xmapi,2);
    Xmap_lat_mov = movmean(Xmap_lat,101);
    x = Xmap_lat_mov;

    % 计算置信区间
    Xmap_std_lat = nanstd(Xmapi,1,2);
    Xmap_num_lat = nansum(isfinite(Xmapi),2);
    Xmap_se_lat = 1.96*Xmap_std_lat./(Xmap_num_lat.^(0.5)); % 0.95 Confidence interbals
    Xmap_se_mov = movmean(Xmap_se_lat,101);
    Xmap_se_mov(isnan(Xmap_lat_mov)) = nan;
    xse = Xmap_se_mov;

    hold on
    lat_flip = flipud(lat);
    ind_lat = find(isfinite(x));
    boundedline(x(min(ind_lat):max(ind_lat)), lat_flip(min(ind_lat):max(ind_lat)), ...
        xse(min(ind_lat):max(ind_lat)), 'orientation', 'horiz','Color',col_lat(i,:),'alpha','Linewidth',1)

end
%做其他调整
xlims = [-0.03,0.03];
xlim(xlims);
ylim([latmin, latmax])
xlabel('\deltaTAC [year^{-1}]','Fontsize',6)
ylabel('Latitude (°)','Fontsize',6)
set(gca,'Fontsize',6)

xlims = get(gca,'Xlim');
ylims = get(gca,'Ylim');

text(xlims(1),ylims(2),'b','HorizontalAlignment','left',...
    'VerticalAlignment','bottom',...
    'FontWeight','bold','Fontsize',6)

% Plot legend 画图例
text(-0.018, 70,' Forest','Fontsize',6,...
    'HorizontalAlignment','left',...
    'VerticalAlignment','middle')
plot([-0.028,-0.018],[70,70],'-','Color',col_lat(1,:),'Linewidth',1.5)

text(-0.018, 65,' Non-forest','Fontsize',6,...
    'HorizontalAlignment','left',...
    'VerticalAlignment','middle')
plot([-0.028,-0.018],[65,65],'-','Color',col_lat(2,:),'Linewidth',1.5)

%%
%分箱图
xstart = fig_pos(1)+10;
ystart = fig_pos(2)-210;
coords = [xstart ystart 180 180];
ax_pos = coords;
ax3 = axes('Unit','pixels','Position',ax_pos,...
    'XColor','k','YColor','k');

cd('F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005')
load('importmaps_qa函数用数据.mat');
% 绘制气温降水空间图
Xtrend(MASK==0) = nan;
Xmap = Xtrend; %用原数据绘图
%设置图例最大最小值
%简单查看一下数值情况
qls = quantile(Xmap(:),[0.05, 0.95])
p = 0.016;
KGZname = {'Tropical', 'Arid', 'Temperate', 'Boreal','Globe'};

% Loop on climate zones
for k=5:5
    %对热带（k=1），干旱区（k=2），温带（k=3），北方（k=4）进行气候分区筛选
    if k<5
        MASKk = koppen_MASK==k;
    else
        MASKk = koppen_MASK>0;%全球地区
    end
    %% Define mask and weigths
    MASK = (isfinite(MASKk))&(isfinite(Tmed))&(isfinite(Pmed));
    
    if sum(MASK(:))>0 % not empty domain

        % Compute map
        Xmap(MASK==0) = nan;

        % Compute average value
        Xavg = wmean(Xmap(MASK==1),W(MASK==1),'omitnan');
        %%计算指定区域内2000-2020年期间的ACF空间上平均值，在这里考虑到了纬度面积大小带来的权重问题
        % Compute weighted standard deviation
        V = (var(Xmap(MASK==1),W(MASK==1),"omitnan"))^(0.5);
        %%计算指定区域内2000-2020年期间的ACF空间上所有值的标准差（var是方差，但这里做了^(0.5)，计算为标准差）
        SEM = V/sqrt(length(Xmap(MASK==1)));               % Standard Error
        %%计算标准误
        ts = tinv([0.025  0.975],length(Xmap(MASK==1))-1);      % T-Score
        %该格点数量下的学生的t反累积分布函数
        % Compute 0.95 confidence interval
        % 基于空间权重平均值，t得分，标准误计算95%的置信区间
        X_ci = Xavg + ts*SEM; %平均值95%置信区间

        % Compute climate space
        xx = Pmed(MASK==1); %筛选出MASK中的相应位置的年降水
        yy = Tmed(MASK==1); %筛选出MASK中的相应位置的年潜在蒸散发
        zz = Xmap(MASK==1); %筛选出MASK中的相应位置的ACF值
        ww = W(MASK==1); %筛选出MASK中的相应位置的纬度权重值
        targetSizeCell = 50;
        xlimset = [0, 5000; -10,30];
        [Xmat,Ymat,Z,Znum,HB,PB] = bioclimaW(xx,yy,zz,ww,xlimset,targetSizeCell,'excl');

    else % empty domain
        Xmap = []; Xavg = []; X_ci = []; Xmat = []; Ymat = []; Z = [];
        Znum = []; HB = []; PB = [];
    end

end
%% CLIMATE SPACE绘图
%获取图例
Z(Znum<10) = nan; % 掩膜仓的代表性很差的仓
pcolor(Xmat,Ymat,Z'); shading flat; axis square;
axis([min(Xmat(:)) max(Xmat(:)) min(Ymat(:)) max(Ymat(:))])
set(gca,'layer','top')

caxis([-p p]) %图例设置

xlabel('Average annual precipitation [mm]','Fontsize',Fsize);
ylabel('Average annual temperature (°C)','Fontsize',Fsize)
set(gca,'XTick',[0,  2500 5000])
set(gca,'YTick',[-10, 0, 10, 20, 30])
set(gca,'Fontsize',Fsize)
colormap(gca,cmap)

% Overlay significance 绘制显著性点
IND = find((HB'==1)&(isfinite(Z')));
Xind = Xmat(IND)+0.5*abs((Xmat(1,1)-Xmat(1,2)));
Yind = Ymat(IND)+0.5*abs((Ymat(1,1)-Ymat(2,1)));
hold on; plot(Xind, Yind,'.k','MarkerSize',3)

% Legend 图例标注
hold on; plot(2500,-5,'.k','MarkerSize',8) 
text(2600,-5,' p-value<0.05','Fontsize',10,'HorizontalAlignment','left',...
    'VerticalAlignment','middle');

xlims = get(gca,'Xlim');
ylims = get(gca,'Ylim');
text(xlims(1),ylims(2),'c','HorizontalAlignment','left',...
    'VerticalAlignment','bottom',...
    'FontWeight','bold','Fontsize',Fsize)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
% 绘制时间上的变化曲线----------------------------------------------------------------
% Setting plot position
xstart = fig_pos(1) + 210;
ystart = fig_pos(2)-210;
xboxnew = 395;
coords = [xstart ystart xboxnew 180];
ax_pos = coords;

ax4 = axes('Unit','pixels','Position',ax_pos,...
    'XColor','k','YColor','k');

cd('F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005\code\FIG1')
load('F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005\code\FIG1\output\KNDVI_AC1_Xavg_glo_k5.mat')
bl_low = acf_avg-acf_ci(:,1);
bl_up = acf_ci(:,2)-acf_avg;
hold on
line1 = boundedline([2005:2024],acf_avg,1*([bl_low,bl_up]),'linewidth',1,'color',[0.6,0.6,0.6],'alpha'); 
[slope, p] = calculate_linear_slope(2005:2023,acf_avg(1:end-1));

load('F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005\code\FIG1\output\KNDVI_AC1_Xavg_forest_k5.mat')
bl_low = acf_avg-acf_ci(:,1);
bl_up = acf_ci(:,2)-acf_avg;
line2 = boundedline([2005:2024],acf_avg,1*([bl_low,bl_up]),'linewidth',1,'color',col_1,'alpha');
[slope_forest, p_forest] = calculate_linear_slope(2005:2023,acf_avg(1:end-1));

load('F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005\code\FIG1\output\KNDVI_AC1_Xavg_Non_forest_k5.mat')
bl_low = acf_avg-acf_ci(:,1);
bl_up = acf_ci(:,2)-acf_avg;
line3 = boundedline([2005:2024],acf_avg,1*([bl_low,bl_up]),'linewidth',1,'color',col_2,'alpha');
[slope_Non_forest, p_Non_forest] = calculate_linear_slope(2005:2023,acf_avg(1:end-1));

set(gca,'layer','top')
set(gca,'Fontsize',6)
box on
% 自定义 x 轴刻度为年份，并去掉中文字符
ax = gca;
ax.YTickLabel = []; % 只显示年份
ax.YTick = []; % 设置 x 轴刻度

xlims = get(gca,'Xlim');
xlim([2005 2024]);
ylims = get(gca,'Ylim');

box on
% 自定义 x 轴刻度为年份，并去掉中文字符
ax = gca;

%图例---------------------------------------------------------------------------------
y1 = 0.025; y2 = 0.02; y3 = 0.015; % 一行内给 y1, y2, y3 分别赋值

text(2008, y1,[' Global slope = 3.2 x 10^{-4} *'],'Fontsize',6,'FontName','Arial',...
    'HorizontalAlignment','left',...
    'VerticalAlignment','middle')
plot([2006,2008],[y1,y1],'-','Color','[0.6,0.6,0.6]','Linewidth',1.5)

text(2008, y2,[' Forest slope = -8.2 x 10^{-5}'],'Fontsize',6,'FontName','Arial',...
    'HorizontalAlignment','left',...
    'VerticalAlignment','middle')
plot([2006,2008],[y2,y2],'-','Color',col_1,'Linewidth',1.5)

text(2008, y3,[' Non-forest slope = 5.1 x 10^{-4} *'],'Fontsize',6,'FontName','Arial',...
    'HorizontalAlignment','left',...
    'VerticalAlignment','middle')
plot([2006,2008],[y3,y3],'-','Color',col_2,'Linewidth',1.5)
%---------------------------------------------------------------------------------
xlabel('year','Fontsize',6);
ylabel('Anomalies','Fontsize',6);

title('TAC','Fontsize',6);

xlims = get(gca,'Xlim');
ylims = get(gca,'Ylim');
text(xlims(1),ylims(2),'d','HorizontalAlignment','left',...
    'VerticalAlignment','bottom',...
    'FontWeight','bold','Fontsize',6)

filenameOut = 'FIG1 KNDVI TAC time space trend';
set(gcf,'renderer','painters')
%这行代码设置当前figure（gcf）的渲染器（renderer）为painters
print('-dtiff', '-r600-', filenameOut)
% print('-dpdf', '-r600-', '-painters', filenameOut)
clc;clear;close all

cd('E:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005\code\FIG2\output')
load("变化方向(显著)_AC1_KNDVI_数据准备.mat","TCmap","TCmapkcz");
TCmap_sw = smooth_data(TCmap, 1);
Xmap = TCmap_sw(201:3000,:);

Xmap = TCmap(201:3000,:);
filenameIn = [ 'E:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\tipping_point\input\005IBGP_SLC.mat'];
load(filenameIn,"IBGP_SLC");
IBGP_SLC = double(IBGP_SLC);
% 将0的值设为NaN
IBGP_SLC(IBGP_SLC == 0) = NaN;
% 将1到5的值设为1
IBGP_SLC(IBGP_SLC >= 1 & IBGP_SLC <= 5) = 1;
% 将6到10的值设为0
IBGP_SLC(IBGP_SLC >= 6 & IBGP_SLC <= 10) = 0;
IBGP_SLC = IBGP_SLC(201:3000,:);
% 绘图 自定义图例展示 ---------------------------------------------
%设置字体大小
Fsize = 10;
% Compute area计算画图面积大小
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%设置画布大小
%figure('Units', 'pixels', 'Position', [100, 100, 900, 320]); % 1200x900像素

figure1 = figure('Units', 'pixels', 'Position', [100, 100, 860, 650]); % 1200x900像素

%% 
% 第一排绘图
% setting external limits 设置外部限值
xbox = 720;  %宽度
ybox = (xbox*(60+80))/360; %高度（宽度*相应比例）
spacex = 30; %子图X方向间距
spacey = 22; %子图Y方向间距
fig_pos = [20 20+340]; %起始绘图的左下角位置像素

cellsize = 0.05;
lat(:,1) = [-60+cellsize/2:cellsize:80-cellsize/2];
lon(1,:) = [-180+cellsize/2:cellsize:180-cellsize/2];
[LON,LAT] = meshgrid(lon,flipud(lat));
latmax = 80;
latmin = -60;
%设置图例间隔与图例
cmap = flipud(brewermap(6,'Paired'));
cmap(3:4,:) = [];
% 设置子图位置
xstart = fig_pos(1);
ystart = fig_pos(2);
coords = [xstart ystart xbox ybox];
ax_pos = coords;
ax1 = axes('Unit','pixels','Position',ax_pos,'XColor','k','YColor','k');
%森林绘图
Xmap_1 = Xmap;
Xmap_1(IBGP_SLC ~= 1) = NaN;
pcolor(LON,LAT,Xmap_1); shading flat; axis auto;
hold on; geoshow('landareas.shp', 'facecolor', 'none')
set(gca,'xtick',[], 'ytick',[],'xticklabel','', 'yticklabel','')
axis([-180, 180, latmin, latmax]);
set(gca,'layer','top')
set(gca,'Fontsize',Fsize)
colormap(gca,cmap)
clim = [1, 5];
caxis([clim]);  % 手动设置颜色范围
% title(ax1,'KNDVI TAC trend','Fontsize',6)
xlims = get(gca,'Xlim');
ylims = get(gca,'Ylim');

hold on;

%rectangle('Position', [760, 360, 80, 270], 'EdgeColor', 'k', 'LineWidth', 1);
%axes(ax0);  % 切换到整个图形的坐标系（即主图）

% 附属饼图1 森林
TCmapcz = TCmapkcz(5,:,1);

xstart = fig_pos(1);
ystart = fig_pos(2);
coords = [xstart+20 ystart+20 80 80];
ax_pos = coords;
ax2 = axes('Unit','pixels','Position',ax_pos,'XColor','k','YColor','k');

data = TCmapcz;
labels = {'类别1', '类别2', '类别3', '类别4'};
% 将数据转换为百分数格式（保留小数点后一位）
total = sum(data); % 计算总和，用于转换为百分数
for i = 1:length(data)
    labels{i} = sprintf('%.1f%%', data(i) / total * 100);
    %labels{i} = sprintf('%.1f%', data(i) / total * 100);
end

colors = cmap;
hollowDonutChart(data, labels, colors,Fsize);

title(ax2,'Forest','Fontsize',Fsize)

% 不同气候区饼图
%如何把这里的四个KGZname作为每一行（四行）的y轴标签添加到图片当中
KGZname = {'Tropical', 'Arid', 'Temperate', 'Boreal'};
for i = 1:4
    TCmapcz = TCmapkcz(i,:,1);

    xstart = 790;
    ystart = 230+345 - (i-1)*(ybox + spacey);
    xbox = 60;
    ybox = 60;
    spacex = 10;
    spacey = 10;

    % 主饼图1：Forest
    coords = [xstart ystart xbox ybox];
    ax_pos = coords;
    ax2 = axes('Unit', 'pixels', 'Position', ax_pos, 'XColor', 'k', 'YColor', 'k');
    
    data = TCmapcz;
    labels = {};
    % 将数据转换为百分数格式（保留小数点后一位）
    total = sum(data); % 计算总和，用于转换为百分数
    for d = 1:length(data)
        labels{d} = sprintf('%.1f%%', data(d) / total * 100);
        %labels{d} = sprintf('%.1f%', data(d) / total * 100);
    end

    colors = cmap;
    hold on;
    hollowDonutChart(data, labels, colors, 8);

    % if i == 1
    %     title(ax2, 'Forest', 'Fontsize', Fsize);
    % end
    
        % 在左侧添加气候区名称作为标签的轴
    axLabel = axes('Units', 'pixels', 'Position', [xstart-40, ystart, xbox, ybox], ...
                   'XColor', 'none', 'YColor', 'none'); % 新增左侧标签轴
    text(0.5, 0.5, KGZname{i}, 'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'FontSize', Fsize, 'FontName', 'Arial', 'Rotation', 90);
    set(axLabel, 'Color', 'none'); % 设置为透明背景
    axis off;
    hold off;
end

hold on;    % 保持当前图形
% 创建 rectangle
annotation(figure1,'rectangle',...
    [750/850 360/650 95/850 280/650]);
hold off;   % 结束保持


%% 
% 第二排绘图
fig_pos = [20 50]; %起始绘图的左下角位置像素
% 设置子图位置
xstart = fig_pos(1);
ystart = fig_pos(2);
xbox = 720;  %宽度
ybox = (xbox*(60+80))/360; %高度（宽度*相应比例）
coords = [xstart ystart xbox ybox];
ax_pos = coords;
ax1 = axes('Unit','pixels','Position',ax_pos,'XColor','k','YColor','k');
%非森林绘图
Xmap_2 = Xmap;
Xmap_2(IBGP_SLC ~= 0) = NaN;

pcolor(LON,LAT,Xmap_2); shading flat; axis auto;
hold on; geoshow('landareas.shp', 'facecolor', 'none')
set(gca,'xtick',[], 'ytick',[],'xticklabel','', 'yticklabel','')
axis([-180, 180, latmin, latmax]);
set(gca,'layer','top')
set(gca,'Fontsize',Fsize)
colormap(gca,cmap)
clim = [1, 5];
caxis([clim]);  % 手动设置颜色范围
% title(ax1,'KNDVI TAC trend','Fontsize',6)
xlims = get(gca,'Xlim');
ylims = get(gca,'Ylim');

% 附属饼图2 非森林
TCmapcz = TCmapkcz(5,:,2);

xstart = fig_pos(1);
ystart = fig_pos(2);
coords = [xstart+20 ystart+20 80 80];
ax_pos = coords;
ax2 = axes('Unit','pixels','Position',ax_pos,'XColor','k','YColor','k');

data = TCmapcz;
labels = {};
% 将数据转换为百分数格式（保留小数点后一位）
total = sum(data); % 计算总和，用于转换为百分数
for i = 1:length(data)
    labels{i} = sprintf('%.1f%%', data(i) / total * 100);
    %labels{i} = sprintf('%.1f%', data(i) / total * 100);
end

colors = cmap;
hollowDonutChart(data, labels, colors,Fsize);

title(ax2,'Non-Forest','Fontsize',Fsize)

% 不同气候区饼图
%如何把这里的四个KGZname作为每一行（四行）的y轴标签添加到图片当中
KGZname = {'Tropical', 'Arid', 'Temperate', 'Boreal'};
for i = 1:4
    xstart = 790;
    ystart = 265 - (i-1)*(ybox + spacey);
    xbox = 60;
    ybox = 60;
    spacex = 10;
    spacey = 10;

    % 附属饼图2：Non-Forest
    TCmapcz = TCmapkcz(i,:,2);

    coords = [xstart ystart xbox ybox];
    ax_pos = coords;
    ax2 = axes('Unit', 'pixels', 'Position', ax_pos, 'XColor', 'k', 'YColor', 'k');
    
    data = TCmapcz;
    labels = {};
    % 将数据转换为百分数格式（保留小数点后一位）
    total = sum(data); % 计算总和，用于转换为百分数
    for d = 1:length(data)
        labels{d} = sprintf('%.1f%%', data(d) / total * 100);
        %labels{d} = sprintf('%.1f%', data(d) / total * 100);
    end

    colors = cmap;
    hold on
    hollowDonutChart(data, labels, colors, 8);

        % 在左侧添加气候区名称作为标签的轴
    axLabel = axes('Units', 'pixels', 'Position', [xstart-40, ystart, xbox, ybox], ...
                   'XColor', 'none', 'YColor', 'none'); % 新增左侧标签轴
    text(0.5, 0.5, KGZname{i}, 'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'FontSize', Fsize, 'FontName', 'Arial', 'Rotation', 90);
    set(axLabel, 'Color', 'none'); % 设置为透明背景
    axis off;
    % if i == 1
    %     title(ax2, 'Non-Forest', 'Fontsize', Fsize);
    % end
end

hold on;    % 保持当前图形
% 创建 rectangle
annotation(figure1,'rectangle',...
    [750/850 50/650 95/850 280/650]);
hold off;   % 结束保持


filenameOut = 'KNDVI_TAC（显著）变化方向绘图';
set(gcf, 'renderer', 'painters');  % 设置当前figure（gcf）的渲染器为painters
%print('-dtiff', '-r600', filenameOut);

clc;clear
KGZname = {'Forest','Non-Forest','All'};

for i =1:1
    close all
    infolder = sprintf(['D:\\paper4_恢复力在干旱和湿润时期的变化\\恢复力数据\\随机森林分析\\' ...
        'output\\MODIS\\RF_koppen_trend_class\\class_land_cover_%d'], i);
    cd(infolder);
    filenameOut = sprintf('RF_PDP_%d_%s', i, char(KGZname{i}));

    cmap = getPyPlot_cMap('PiYG_r', 20); %设置颜色
    if i == 1
        cmap = flipud(cmap(1:10,:));
    elseif i == 2
        cmap = (cmap(11:20,:));
    elseif i == 3
        cmap = (cmap(1:2:20,:));
    end

    load('kndvi_RFC_ALL_ind_exnan.mat','Predictors_names','varimp_all','accuracy_all')

    varimp_mean = mean(varimp_all,2);
    varimp_std = std(varimp_all, 0, 2);

    [varimp_sort, idxvarimp] = sort(varimp_mean);
    varimp_std_sort = varimp_std(idxvarimp);
    Predictor_sort = Predictors_names(idxvarimp);

    accuracy_mean = mean(accuracy_all);
    accuracy_std = std(accuracy_all);
    clear varimp_all Predictors_names varimp_mean accuracy_all

    % 定义名称修改对应关系的字典
    name_map = {
        'tp.ts', 'P';
        't2m.ts', 'T_a';
        'def.ts', 'Water Deficit';
        'ssrd.ts', 'R_s';
        'sm0.28cm.ts', 'SM';
        'VPD.ts', 'VPD';
        'biomass.005.ts', 'AGB';
        'ecosystem.heterogeneity.005.ts','Hterogeneity';
        'for.ts', 'Forest density';
        'gra.ts', 'Grass density';
        'GLW.2015.ts', 'Livestock';
        'wtd.ts', 'WTD';
        'Elevation.ts', 'Elevation';
        'Slope.ts', 'Slope';
        'Aspect.ts', 'Aspect';
        'BA.TOT.GFED.025d.ts', 'Burned area';
        'FRP.ts', 'FRP';
        'HFP.ts', 'HFP';
        'SoilGrid.clay.mean.ts', 'Clay';
        'SoilGrid.sand.mean.ts', 'Sand';
        'SoilGrid.soc.mean.ts', 'SOC';
        'SoilGrid.nitrogen.mean.ts', 'Nitrogen';
        'Species.sum.ts', 'Species';
        'kndvi.ts', 'kNDVI';
        'tp.trend', '\delta P'
        't2m.trend', '\delta T_a';
        'for.trend', '\delta Forest density';
        'kndvi.trend', '\delta kNDVI';
        'tp.nipv', 'P.NIPV';
        'tp.mean.si', 'P.SI';
        };

    % 遍历Predictor_sort并修改名称
    for j = 1:length(Predictor_sort)
        % 获取当前名称
        old_name = Predictor_sort{j};
        % 查找对应的新名称
        idx = find(strcmp(name_map(:, 1), old_name));
        if ~isempty(idx)
            % 修改名称
            Predictor_sort{j} = name_map{idx, 2};
        end
    end
    % 我需要依据随机森林的重要性varimp_sort(23x1 double)计算重要性的百分百，然后把重要性排前14个的因子的百分比绘制为柱状图，
    % 并且在Y轴上为每个柱子标准对应的Predictor_sort(23x1 cell)中的对应名称，注意越重要的因子的柱状图颜色越在cmap中对应颜色靠前，请帮我完成MATLAB代码，你可以参考我提供给你的pdf中fig3 a的作图风格
    figure('Units', 'pixels', 'Position', [100, 100, 850, 350]); % 1200x900像素
    title(KGZname{i},'Fontsize',12,'FontName','Arial')
    axis off;
    % setting external limits 设置外部限值
    xbox = 140;  %宽度
    ybox = 120; %高度（宽度*相应比例）
    spacex = 40; %子图X方向间距
    spacey = 40; %子图Y方向间距
    fig_pos = [80 40]; %起始绘图的左下角位置像素

    xstart = fig_pos(1);
    ystart = fig_pos(2);
    coords = [xstart ystart xbox ybox*2+spacey];
    ax_pos = coords;
    ax1 = axes('Unit','pixels','Position',ax_pos,'XColor','k','YColor','k');

    varimp_percentage = varimp_sort / sum(varimp_sort) * 100;
    varimp_std_percentage = varimp_std_sort / sum(varimp_sort) * 100;
    % 按重要性排序，选择前10个
    [~, idx] = sort(varimp_percentage, 'descend');
    top10_idx = idx(1:10);
    top10_varimp = varimp_percentage(top10_idx);
    top10_varimp_std = varimp_std_percentage(top10_idx);
    top10_labels = Predictor_sort(top10_idx);

    % 创建柱状图
    b = barh(top10_varimp, 'FaceColor', 'flat','BarWidth', 0.5);
    colormap(cmap);

    % 设置每个柱子的颜色
    for j = 1:length(top10_varimp)
        % 根据重要性越大颜色越靠前的原则设置颜色
        bar_color_idx = 11 - j; % 颜色从cmap中选择
        b.CData(j, :) = cmap(bar_color_idx, :);
    end

    % 添加误差棒
    hold on; % 保持图形
    % 获取柱状图的y坐标
    y_positions = b.YData;
    x_positions = b.XData;
    % 绘制误差棒，x为值，y为y轴位置，误差为std
    errorbar(y_positions, x_positions, top10_varimp_std,'horizontal',"CapSize",3,'Color',[0.1,0.1,0.1], 'LineStyle', 'none', 'LineWidth', 1);
    hold off;

    % 设置Y轴标签
    set(gca, 'YTick', 1:10, 'YTickLabel', top10_labels, 'YDir', 'reverse');
    xlabel('Predictor importance (%)');
    %title('前10个因子的随机森林重要性');

    % 设置图形样式
    set(gca, 'FontSize', 8,'FontName','Arial');
    box on;
    %显示模型精确度数值
    xlims = get(gca,'Xlim');
    ylims = get(gca,'Ylim');

    % 格式化标注字符串，显示 "均值±标准差%"
    accuracy_text = sprintf('Accuracy = %.1f ± %.1f%%', accuracy_mean*100, accuracy_std*100);

    % 在图中添加标注
    text((xlims(2)+xlims(1))/2, ylims(2)*0.98, accuracy_text, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'baseline', ...
        'FontSize', 8, 'FontName', 'Arial');
    %固定轴范围
    xlim(xlims);
    ylim(ylims);

    %%
    %对名字加入单位
    name_map = {
        'P', 'P (mm)';
        'T_a', 'T_a (°C)';
        'Water Deficit', 'Water Deficit (mm)';
        'R_s', 'R_s (J/m^2)';
        'SM', 'SM (m^3/m^3)';
        'VPD', 'VPD (hPa)';
        'AGB', 'AGB (Mg/ha)';
        'Hterogeneity','Hterogeneity (-)';
        'Forest density', 'Forest density (-)';
        'Grass density', 'Grass density (-)';
        'Livestock', 'Livestock (M birds/heads)';
        'WTD', 'WTD (m)';
        'Elevation', 'Elevation (m)';
        'Slope', 'Slope (°)';
        'Aspect', 'Aspect (°)';
        'Burned area', 'Burned area (%)';
        'FRP', 'FRP (GW per O.05° grid)';
        'HFP', 'HFP (-)';
        'Clay', 'Clay (g/kg)';
        'Sand','Sand (g/kg)'
        'SOC', 'SOC (dg/kg)';
        'Nitrogen', 'Nitrogen (cg/kg)';
        'Species', 'Species (-)';
        'kNDVI', 'kNDVI (-)';
        '\delta P', '\delta P (mm/year)'
        '\delta T_a', '\delta T_a (°C/year)';
        '\delta Forest density', '\delta Forest density (-)';
        '\delta kNDVI', '\delta kNDVI (-)';
        'P.NIPV', 'P.NIPV (-)';
        'P.SI', 'P.SI (-)';
        };

    % 遍历Predictor_sort并修改名称
    for j = 1:length(Predictor_sort)
        % 获取当前名称
        old_name = Predictor_sort{j};
        % 查找对应的新名称
        idx = find(strcmp(name_map(:, 1), old_name));
        if ~isempty(idx)
            % 修改名称
            Predictor_sort{j} = name_map{idx, 2};
        end
    end

    cmap = getPyPlot_cMap('PiYG_r', 10); %设置颜色
    cmap1 = cmap(2,:);
    cmap2 = cmap(9,:);

    for n = 1:3
        xstart = fig_pos(1)+n*(spacex+xbox*1.15);
        ystart = fig_pos(2)+spacey + ybox;
        coords = [xstart ystart xbox ybox];
        ax_pos = coords;
        ax2 = axes('Unit','pixels','Position',ax_pos,'XColor','k','YColor','k');

        %PDP结果矩阵（30次循环, 100个分段, 26个变量， 2个分类）
        load('kndvi_RFC_ALL_ind_exnan.mat','X_PDP_all','Y_PDP_all')
        % 对 X_PDP_all 的第三维度依据 idxvarimp 进行排序
        X_PDP_sorted = X_PDP_all(:, :, idxvarimp, :);
        Y_PDP_sorted = Y_PDP_all(:, :, idxvarimp, :);


        num = size(varimp_percentage,1)-n+1;

        X1 = squeeze(X_PDP_sorted(:,:,num,1));
        Y1 = squeeze(Y_PDP_sorted(:,:,num,1));

        X2 = squeeze(X_PDP_sorted(:,:,num,2));
        Y2 = squeeze(Y_PDP_sorted(:,:,num,2));

        X1_avg = mean(X1,1);
        Y1_avg = mean(Y1,1);
        % Loop on year
        for j=1:size(Y1,2)
            Yt = Y1(:,j);
            % 计算加权标准差
            V = var(Yt)^(0.5); %计算加权方差的平方根，即标准差
            SEM = V/sqrt(length(Yt));               % 计算标准误差
            ts = tinv([0.025  0.975],length(Yt)-1);      % 根据 t 分布计算 t 值，使用双侧 95% 置信区间的临界值
            % Compute 0.95 confidence interval
            Y1_ci(j,:) = Y1_avg(j) + ts*SEM;
        end

        Y1_low = Y1_avg'-Y1_ci(:,1);
        Y1_up = Y1_ci(:,2)-Y1_avg';
        %设置图片可见
        hold on
        line1 = boundedline(X1_avg,Y1_avg,10*([Y1_low,Y1_up]),'linewidth',1.5,'color',cmap1,'alpha');

        X2_avg = mean(X2,1);
        Y2_avg = mean(Y2,1);
        % Loop on year
        for j=1:size(Y2,2)
            Yt = Y2(:,j);
            % 计算加权标准差
            V = var(Yt)^(0.5); %计算加权方差的平方根，即标准差
            SEM = V/sqrt(length(Yt));               % 计算标准误差
            ts = tinv([0.025  0.975],length(Yt)-1);      % 根据 t 分布计算 t 值，使用双侧 95% 置信区间的临界值
            % Compute 0.95 confidence interval
            Y2_ci(j,:) = Y2_avg(j) + ts*SEM;
        end

        Y2_low = Y2_avg'-Y2_ci(:,1);
        Y2_up = Y2_ci(:,2)-Y2_avg';
        %设置图片可见
        hold on
        line2 = boundedline(X2_avg,Y2_avg,10*([Y2_low,Y2_up]),'linewidth',1.5,'color',cmap2,'alpha');
        box on

        title(Predictor_sort(num));
        set(gca, 'FontSize', 8,'FontName','Arial');

        xlims = get(gca,'Xlim');
        ylims = get(gca,'Ylim');

        % 格式化字符串，显示 "百分比 ± 标准差%"
        varimp_text = sprintf('%.1f ± %.1f%%', varimp_percentage(num), varimp_std_percentage(num));

        % 在图中添加标注
        text((xlims(2) + xlims(1)) / 2, ylims(2), varimp_text, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 8, 'FontName', 'Arial');


        %固定轴范围
        xlim(xlims);
        ylim(ylims);
    end

    %%
    for n = 4:6
        xstart = fig_pos(1)+(n-3)*(spacex+xbox*1.15);
        ystart = fig_pos(2);
        coords = [xstart ystart xbox ybox];
        ax_pos = coords;
        ax2 = axes('Unit','pixels','Position',ax_pos,'XColor','k','YColor','k');

        %PDP结果矩阵（30次循环, 100个分段, 26个变量， 2个分类）
        load('kndvi_RFC_ALL_ind_exnan.mat','X_PDP_all','Y_PDP_all')
        % 对 X_PDP_all 的第三维度依据 idxvarimp 进行排序
        X_PDP_sorted = X_PDP_all(:, :, idxvarimp, :);
        Y_PDP_sorted = Y_PDP_all(:, :, idxvarimp, :);


        num = size(varimp_percentage,1)-n+1;

        X1 = squeeze(X_PDP_sorted(:,:,num,1));
        Y1 = squeeze(Y_PDP_sorted(:,:,num,1));

        X2 = squeeze(X_PDP_sorted(:,:,num,2));
        Y2 = squeeze(Y_PDP_sorted(:,:,num,2));

        X1_avg = mean(X1,1);
        Y1_avg = mean(Y1,1);
        % Loop on year
        for j=1:size(Y1,2)
            Yt = Y1(:,j);
            % 计算加权标准差
            V = var(Yt)^(0.5); %计算加权方差的平方根，即标准差
            SEM = V/sqrt(length(Yt));               % 计算标准误差
            ts = tinv([0.025  0.975],length(Yt)-1);      % 根据 t 分布计算 t 值，使用双侧 95% 置信区间的临界值
            % Compute 0.95 confidence interval
            Y1_ci(j,:) = Y1_avg(j) + ts*SEM;
        end

        Y1_low = Y1_avg'-Y1_ci(:,1);
        Y1_up = Y1_ci(:,2)-Y1_avg';
        %设置图片可见
        hold on
        line1 = boundedline(X1_avg,Y1_avg,10*([Y1_low,Y1_up]),'linewidth',1.5,'color',cmap1,'alpha');

        X2_avg = mean(X2,1);
        Y2_avg = mean(Y2,1);
        % Loop on year
        for j=1:size(Y2,2)
            Yt = Y2(:,j);
            % 计算加权标准差
            V = var(Yt)^(0.5); %计算加权方差的平方根，即标准差
            SEM = V/sqrt(length(Yt));               % 计算标准误差
            ts = tinv([0.025  0.975],length(Yt)-1);      % 根据 t 分布计算 t 值，使用双侧 95% 置信区间的临界值
            % Compute 0.95 confidence interval
            Y2_ci(j,:) = Y2_avg(j) + ts*SEM;
        end

        Y2_low = Y2_avg'-Y2_ci(:,1);
        Y2_up = Y2_ci(:,2)-Y2_avg';
        %设置图片可见
        hold on
        line2 = boundedline(X2_avg,Y2_avg,10*([Y2_low,Y2_up]),'linewidth',1.5,'color',cmap2,'alpha');
        box on

        title(Predictor_sort(num));
        set(gca, 'FontSize', 8,'FontName','Arial');

        xlims = get(gca,'Xlim');
        ylims = get(gca,'Ylim');
        % 格式化字符串，显示 "百分比 ± 标准差%"
        varimp_text = sprintf('%.1f ± %.1f%%', varimp_percentage(num), varimp_std_percentage(num));

        % 在图中添加标注
        text((xlims(2) + xlims(1)) / 2, ylims(2), varimp_text, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 8, 'FontName', 'Arial');


        %固定轴范围
        xlim(xlims);
        ylim(ylims);
    end

    cd ..
    cd('Fig\')
    %print('-dtiff', '-r330-', filenameOut)
    %print('-dpdf', '-r330-', filenameOut)
end
clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERAL SETTING
pathIn = 'F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005\code\FIG1\output\';
yearstart = 2003;
yearend = 2024;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% setting external limits
scrsz = [1, 1, 1920, 1080];
xbox = 250;
ybox = 85;
spacey = 20;
spacex = 50;
spacexx = 0;
spaceax = 40;
fig_pos = [60 450];

LCclass = {'forest','Non_forest','glo'};
LCclass_title = {'forest','Non forest','glo'};
for i = 1:3
    disp(LCclass{i})

    h = figure('position',[1 1 scrsz(3) scrsz(4)]);
    set(h,'unit','centimeters','position',[0, 0, 9, 14.5],...
        'PaperUnits','centimeters',...
        'PaperSize',[9 14.5],...
        'PaperPosition',[0, 0,  9, 14.5],...
        'PaperPositionMode','auto','Visible', 'on');

    KGZname = {'Tropical','Arid','Temperate','Boreal','Globe'};
    kord = [5,1,2,3,4];
    cmap_kcz = (1/255)*[0, 176, 0;
        255, 87, 20;
        235, 215, 0;
        105, 219, 255;
        150, 150, 150];
    tit = {'a','b','c','d','e'};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PLOT TIME SERIES
    % Loop on climate zone
    for kt=1:length(KGZname)
        k = kord(kt);
            % Setting plot position
            xstart = fig_pos(1);
            ystart = fig_pos(2)-(kt-1)*(ybox+spacey);

            coords = [xstart ystart xbox ybox];
            ax_pos = coords;
            ax1 = axes('Unit','pixels','Position',ax_pos,...
                'XColor','k','YColor','k');

            %KNDVI_AC1_Xavg_forest_k1.mat 案例输入文件名
            filenameIn = [pathIn,'KNDVI_AC1_Xavg_',LCclass{i},'_k',num2str(k)];
            load(filenameIn,'acf_avg','acf_ci')

            bl_low = acf_avg-acf_ci(:,1);
            bl_up = acf_ci(:,2)-acf_avg;
            acf_avg_rc = acf_avg;
            boundedline([yearstart:yearend],acf_avg_rc,10*([bl_low,bl_up]),'color',cmap_kcz(k,:))

            set(gca,'layer','top')
            set(gca,'Fontsize',6)
            box on
            xlim([yearstart,yearend])
            ylim([min(acf_avg_rc-10*bl_low),max(acf_avg_rc+10*bl_up)])

            xlims = get(gca,'Xlim');
            ylims = get(gca,'Ylim');

            text(xlims(1),ylims(2),tit{kt},'HorizontalAlignment','right',...
                'VerticalAlignment','bottom',...
                'FontWeight','bold','Fontsize',8)
            if k==5
                title(LCclass_title{i},'Fontsize',8)
            end
            if k~=4
                set(gca,'Xtick',[yearstart+1:4:yearend],'XtickLabel','')
            else
                set(gca,'Xtick',[yearstart+1:4:yearend])
            end

            ylabel(['\fontsize{7pt}\bf{',KGZname{k},'}' newline ...
                '\fontsize{6pt}\rm{TAC}'])

            if k==4
                xlabel('Year','Fontsize',6)
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        set(gcf,'renderer','painters')
        filenameOut = ['F:\paper4_恢复力在干旱和湿润时期的变化\恢复力数据\KNDVI_AC1_mon_005\code\FIG1\', 'Change_time_',LCclass{i}];
        print('-dtiff', '-r300-', filenameOut)
    end
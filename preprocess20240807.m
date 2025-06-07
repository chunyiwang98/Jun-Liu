close all; clear all;clc


analysis ='TFR'; 
subjects=[1183]
s = 1:length(subjects)

direct = sprintf('D:/02252025msa', subjects);
cd (direct);  

%load files and resamplefs
cfg            = [];
%cfg.dataset = '21001_ec_1.mff'
cfg.dataset = (sprintf('%d.edf',subjects))
hdr            = ft_read_header(cfg.dataset);
cfg.continuous = 'yes';
cfg.channel    = 'all';
data_mff       = ft_preprocessing(cfg);
cfg.resamplefs = 500;
data_mff = ft_resampledata(cfg, data_mff);

% % visually inspect the data
%     cfg            = [];
%     cfg.viewmode   = 'vertical';
%     ft_databrowser(cfg, data_mff);

% define trial
cfg = [];
cfg.trialfun   = 'ft_trialfun_general';
%cfg.headerfile = '21001_ec_1.mff'
cfg.headerfile = (sprintf('%d.edf',subjects))
cfg            = ft_trialfun_general(cfg);
cfg.trials     = 'all'
cfg.length     = 2;
cfg.trialdef.ntrials  = Inf;

data_mff =  ft_redefinetrial(cfg,data_mff);

cfg.channel   = {'EEG Fp1' ,'EEG Fp2','EEG F3','EEG F4' ,'EEG C3' ,'EEG C4' ,'EEG P3' ,'EEG P4','EEG O1' ,'EEG O2','EEG F7','EEG F8' ,'EEG T7' ,'EEG T8' ,'EEG P7' ,'EEG P8' ,'EEG Fz' ,'EEG FCz' ,'EEG Cz','EEG Pz' ,'EEG FC1','EEG FC2' ,'EEG CP1' ,'EEG CP2','EEG FC5' ,'EEG FC6','EEG CP5','EEG CP6','EEG TP9' ,'EEG TP10','EEG F1' ,'EEG F2' ,'EEG C1'  ,'EEG C2' ,'EEG P1' ,'EEG P2','EEG AF3','EEG AF4','EEG FC3' ,'EEG FC4' ,'EEG CP3' ,'EEG CP4' ,'EEG PO3' ,'EEG PO4','EEG F5','EEG F6','EEG C5','EEG C6','EEG P5' ,'EEG P6' ,'EEG AF7','EEG AF8','EEG FT7','EEG FT8' ,'EEG TP7' ,'EEG TP8','EEG PO7' ,'EEG PO8', 'EEG Fpz','EEG CPz','EEG POz','EEG Oz'};
data_mff = ft_selectdata(cfg, data_mff);

%Filter
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [1 90];
cfg.dftfilter='yes';
cfg.dftfreq=[50];
data_fil = ft_preprocessing(cfg,data_mff);


%Visual Rejection and Repair
cfg         =  [];
cfg.method  = 'summary';
cfg.channel = 'all';
cfg.keepchannel = 'yes';
data_fil    = ft_rejectvisual(cfg,data_fil);

data_fil.label = {'Fp1' ,'Fp2','F3','F4' ,'C3' ,'C4' ,'P3' ,'P4','O1' ,'O2','F7','F8' ,'T7' ,'T8' ,'P7' ,'P8' ,'Fz' ,'FCz' ,'Cz','Pz' ,'FC1','FC2' ,'CP1' ,'CP2','FC5' ,'FC6','CP5','CP6','TP9' ,'TP10','F1' ,'F2' ,'C1'  ,'C2' ,'P1' ,'P2','AF3','AF4','FC3' ,'FC4' ,'CP3' ,'CP4' ,'PO3' ,'PO4','F5','F6','C5','C6','P5' ,'P6' ,'AF7','AF8','FT7','FT8' ,'TP7' ,'TP8','PO7' ,'PO8','Fpz','CPz','POz','Oz'}';


%Bad eletrodes repair      %repairing the bad electrodes with averaged of its neighbours
cfg  = [];
cfg.method  = 'triangulation';                  %'distance', 'triangulation' or'template' (default = 'distance')
cd('C:/Users/Lenovo/Desktop/fieldtrip-20200406/template/electrode')
cfg.layout        = 'standard_1020.elc';
cfg.feedback      = 'no';
neighbours        = ft_prepare_neighbours(cfg);

cfg               = [];
cd (direct)
cfg.channel = data_fil.label;
cfg.badchannel    = input('write badchannels: ')     %{'E7';'Cz'}
cfg.trials        = 'all';
cfg.layout        = 'standard_1020.elc';
cfg.method        = 'average';
cfg.neighbours    = neighbours;
data_repair       = ft_channelrepair(cfg,data_fil);

%check the data structure;

%% demean
cfg = [];
cfg.demean         = 'yes';          % whether to apply baseline correction (default = 'no')
cfg.baselinewindow = ['all'];        % [begin end] in seconds, the default is the complete trial (default = 'all')
cfg.detrend        = 'yes';          % Trend removal

%% rerefencing
cfg.reref       = 'yes';
cfg.channel     = {'all'};
cfg.refchannel     = {'all'};
data_demean = ft_preprocessing(cfg,data_repair);

%% ICA
cfg = [];
cfg.continuous = 'yes';
cfg.numcomponent = 30;
comp = ft_componentanalysis(cfg,data_demean);

cfg.component = 1:size(comp.topo,2);                       % specify the component(s) that should be plotted
cfg.layout    = 'standard_1020.elc';                    % specify the layout file that should be used for plotting
cfg.comment   = 'no';
figure;
ft_topoplotIC(cfg, comp);
cfg.viewmode = 'component';
cfg.blocksize = 3;
ft_databrowser(cfg, comp);


cfg = [];
cfg.component = input('Which components would you like to remove?');%(i.e. [8 11])
data_ICA = ft_rejectcomponent(cfg, comp);


% % visually inspect the data
cfg            = [];
cfg.viewmode   = 'vertical';
ft_databrowser(cfg, data_ICA);

%% remove most obvious artifacts based on variance
%     cfg = [];
%     cfg.keepchannel = 'yes';
%     data_clean = ft_rejectvisual_auto(cfg,data_ICA);

new_folder = 'D:/预处理20250225新增MSA';
% mkdir(new_folder);
cd(new_folder);
savename1 = sprintf('%d_ec_ICA.mat',subjects(s));
% savename2 = sprintf('%d_ec_clean.mat',subjects(s));
save(savename1, 'data_ICA', 'cfg')
% save(savename2, 'data_clean','cfg')

       
% cfg            = [];
% cfg.viewmode   = 'vertical';
% ft_databrowser(cfg, data_clean);
       
clc;close all;clear all;
addpath(genpath('./'));
load modelparameters.mat

blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

rst_dir  = '/Users/liam/PycharmProjects/curvenight_sim/result/';
%rst_dir = '/Users/liam/PycharmProjects/curvenight/results';
if exist(fullfile(rst_dir,'results.txt'))
    delete fullfile(rst_dir,'results.txt')
end
if exist(fullfile(rst_dir,'.DS_Store'))
    delete fullfile(rst_dir,'.DS_Store')
end
if exist(fullfile(rst_dir,'.gitkeep'))
    delete fullfile(rst_dir,'.gitkeep')
end
if exist(fullfile(rst_dir,'results.txt'))
    delete fullfile(rst_dir,'results.txt')
end

exp_list = dir(rst_dir);
exp_names = cell(1,length(exp_list)-2);
exp_results_m = [];
fp=fopen(fullfile(rst_dir,'results.txt'),'a');
for k =1:length(exp_list)
    if( isequal( exp_list( k ).name, '.' )||...
            isequal( exp_list( k ).name, '..')||...
            ~exp_list( k ).isdir)               % 如果不是目录则跳
        continue;
    end
    dataset_list = dir(fullfile(exp_list( k ).folder, exp_list( k ).name));
    fprintf(fp, '%d. %s \n',k-2, exp_list( k ).name);
    exp_names(1, k-2) = {exp_list( k ).name};
    % a = dir([rst_dir, '/*/*.*']);
    dataset_names = cell(1,length(dataset_list)-2);
    exp_results_v = [];
    for i = 1 : length( dataset_list )
        if( isequal( dataset_list( i ).name, '.' )||...
                isequal( dataset_list( i ).name, '..')||...
                ~dataset_list( i ).isdir)               % 如果不是目录则跳
            continue;
        end
        dataset_names(1,i-2) = {dataset_list( i ).name};
        % dataset_list( i ).name
        % dataset = dataset_list(i).name
        img_list = dir(fullfile(dataset_list(i).folder, dataset_list(i).name));
        
        NIQE = [];
        for j = 1:length( img_list )
            if( img_list( j ).isdir)               % 如果不是目录则跳
                continue;
            end
            
            % 计算niqe并存下来
            %imgname = img_list(j).name
            %fullfile(img_list(j).folder, img_list(j).name);
            imgname = fullfile(img_list(j).folder, img_list(j).name);
            if exist(fullfile(img_list(j).folder,'.DS_Store'))
                delete fullfile(img_list(j).folder,'.DS_Store')
            end
            img=imread(imgname);
            quality = computequality(img,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
                mu_prisparam,cov_prisparam);
            NIQE = [NIQE; quality];
        end
        mNIQE = mean(NIQE);
        exp_results_v = [exp_results_v;mNIQE];
        
        fprintf(fp, '- The average NIQE for %d images in %s is %2.4f \n',length(img_list)-2, dataset_list( i ).name, mNIQE);
        fprintf('- The average NIQE for %d images in %s is %2.4f \n',length(img_list)-2, dataset_list( i ).name, mNIQE);
        
    end
    exp_results_m = [exp_results_m, exp_results_v];
end

results = array2table(exp_results_m,'RowNames',dataset_names, 'VariableNames',exp_names)
writetable(results,fullfile(rst_dir,'resultNIQE.xlsx'),'WriteRowNames',true)

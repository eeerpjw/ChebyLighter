clc;close all;clear all;addpath(genpath('./'));
load modelparameters.mat

blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

%rst_dir  =    './myresult/test11_ssim_2tv_125_85/';
rst_dir = '/Users/liam/PycharmProjects/curvenight/results/nsw_chebyall_fcat3_DAB_ssim_2_a10_200';
dataset_list = dir(rst_dir);

%a = dir([rst_dir, '/*/*.*']);

for i = 1 : length( dataset_list )
    if( isequal( dataset_list( i ).name, '.' )||...
        isequal( dataset_list( i ).name, '..')||...
        ~dataset_list( i ).isdir)               % 如果不是目录则跳
        continue;
    end
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
        img=imread(fullfile(img_list(j).folder, img_list(j).name));
        quality = computequality(img,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
            mu_prisparam,cov_prisparam);
        NIQE = [NIQE; quality];
    end
    mNIQE = mean(NIQE);
    fprintf('- The average NIQE for %d images in %s is %2.4f \n',length(img_list)-2, dataset_list( i ).name, mNIQE);
    
end



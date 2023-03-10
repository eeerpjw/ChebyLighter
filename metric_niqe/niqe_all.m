%% 准备
clc;close all;clear all;
addpath(genpath('./'));
load modelparameters.mat

blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

%% 基本还要求
% 现在的文件组织方式：
% **实验名--数据集名--图片**
% exp_ name--dataset_name--img_name
% 期望的数值写入结果为：
% 数据集为文件名写入.csv文件(dataset_name.csv)，每个文件中row为图片名，col为方法名。

%% 读入文件的准备
% read_path  = '/Users/liam/Desktop/results_traditional';
read_path = '../result';
save_path = '../NIQE_result';
mkdir(save_path)
if exist(fullfile(read_path,'results.txt'))
    delete fullfile(read_path,'results.txt')
end
if exist(fullfile(read_path,'.DS_Store'))
    delete fullfile(read_path,'.DS_Store')
end
if exist(fullfile(read_path,'.gitkeep'))
    delete fullfile(read_path,'.gitkeep')
end

exp_list = get_sub_folder_names(read_path);
dataset_list = get_sub_folder_names(fullfile(read_path, exp_list{1}));
average_table_content = [];

for i=1:length(dataset_list)
    dataset_name = dataset_list{i}
    result_table_content = [];
    for j=1:length(exp_list)
        exp_name = exp_list{j};
        img_list = dir(fullfile(read_path,exp_name,dataset_name));
        img_names = populate_images(fullfile(read_path,exp_name,dataset_name));
        exp_results = [];
        for k=1:length(img_names)
            %img_names{k}
            img_path = fullfile(read_path,exp_name,dataset_name,img_names{k});
            img = imread(img_path);
            quality = computequality(img,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
                mu_prisparam,cov_prisparam);
            exp_results = [exp_results, quality];
        end
        result_table_content = [result_table_content;exp_results];
    end
    results_table = array2table(result_table_content,'RowNames',exp_list, 'VariableNames',img_names)
    writetable(results_table,fullfile(save_path,[dataset_name,'.xlsx']),'WriteRowNames',true)
    average_table_content = [average_table_content, mean(result_table_content,2)];
end
average_table = array2table(average_table_content,'RowNames',exp_list, 'VariableNames',dataset_list)
writetable(average_table,fullfile(save_path,['ablation_num_blocks_avg_NIQE.xlsx']),'WriteRowNames',true)
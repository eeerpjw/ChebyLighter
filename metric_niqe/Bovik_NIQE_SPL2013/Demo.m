clc;close all;clear all;addpath(genpath('./'));
load modelparameters.mat

blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;
% BIMEF
% CRM_ICCV
% Dong
% LIME
% MF
% MSR
% NPE
% SRIE
Original_image_dir  =    './myresult/eval_results_LOLUnpair_rec_per_plus_colorg2_5_210/VV/';
fpath = fullfile(Original_image_dir, '*.jpg');
im_dir  = dir(fpath);
im_num = length(im_dir);
NIQE = [];
for i = 1:im_num
    %% read clean image
    i
    %IMname = regexp(im_dir(i).name, '\.', 'split');
    %IMname = IMname{1};
    im=imread(fullfile(Original_image_dir, im_dir(i).name));
        quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
            mu_prisparam,cov_prisparam);
        %imDis is a RGB colorful image
        NIQE = [NIQE; quality];
end
mNIQE = mean(NIQE);

fprintf('The average NIQE = %2.4f. \n', mNIQE);
save ./myresult/eval_results_LOLUnpair_rec_per_plus_colorg2_5_210/VV/VV_NIQE_myresult1.mat NIQE mNIQE

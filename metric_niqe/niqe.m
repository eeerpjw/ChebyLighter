function q = niqe(img)
% warp niqe function with settled params
    blocksizerow    = 96;
    blocksizecol    = 96;
    blockrowoverlap = 0;
    blockcoloverlap = 0;
    q = computequality(img,blocksizerow,blocksizecol,blockrowoverlap,...
        blockcoloverlap,mu_prisparam,cov_prisparam)
end
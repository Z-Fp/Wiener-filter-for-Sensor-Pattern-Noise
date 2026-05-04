% PCE (Peak-to-Correlation Energy) classifier on Wiener residuals.
%
% PCE is the standard matching metric in the SPN forensics literature
% (Goljan, Fridrich, Filler 2009). Unlike NCC, it computes the full 2D
% cross-correlation and reports
%
%     PCE = peak^2 / (1/(N - |neigh|)) * sum_{not neigh}(C^2)
%
% where C is the circular cross-correlation surface, "peak" is its max,
% and "neigh" is a small exclusion window around the peak to remove its
% own energy from the denominator. This gives a sharpness-aware score
% that is more robust to mild misalignments than plain NCC.
%
% This script reuses the centroids saved by extract_wiener_gray.m.

close all; clear; clc;

% ---- paths ----
testRoot       = "data/test/";
centroidsFile  = "results/Wiener_Centroids_Gray.mat";
neighRadius    = 11;       % half-side of the peak exclusion window

% ---- load centroids ----
load(centroidsFile, "mu", "classNames");
numClasses = numel(classNames);
[H, W, ~] = size(mu);

% ---- testing ----
fprintf("Testing Wiener + PCE (gray)\n");

dsTest = fileDatastore(testRoot, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.mat', ...
    'ReadFcn', @load);

numTest = numel(dsTest.Files);
allPredIdx = zeros(numTest, 1);
allTrueIdx = zeros(numTest, 1);
allPceTop  = zeros(numTest, 1);

for i = 1:numTest
    data = load(dsTest.Files{i});
    Z = double(data.residual);
    Z = mean(Z, 3);
    Z = Z - mean(Z, 'all');

    pceScores = zeros(numClasses, 1);
    for c = 1:numClasses
        pceScores(c) = computePCE(Z, mu(:,:,c), neighRadius);
    end
    [pceTop, predIdx] = max(pceScores);

    [~, labelName] = fileparts(fileparts(dsTest.Files{i}));
    allPredIdx(i) = predIdx;
    allTrueIdx(i) = find(classNames == string(labelName));
    allPceTop(i)  = pceTop;
end

acc = mean(allPredIdx == allTrueIdx);
fprintf("\nGray Wiener PCE accuracy: %.2f%%\n", acc * 100);
fprintf("Mean top-1 PCE on correct calls:   %.2f\n", mean(allPceTop(allPredIdx == allTrueIdx)));
fprintf("Mean top-1 PCE on incorrect calls: %.2f\n", mean(allPceTop(allPredIdx ~= allTrueIdx)));

figure;
confusionchart(allTrueIdx, allPredIdx);
title(sprintf("Gray Wiener PCE Accuracy: %.2f%%", acc * 100));


% ---- helper: PCE between two 2D arrays ----
function pce = computePCE(X, Y, r)
    % Circular 2D cross-correlation via FFT.
    C = real(ifft2(fft2(X) .* conj(fft2(Y))));

    [peak, idx] = max(abs(C(:)));
    [py, px] = ind2sub(size(C), idx);

    % Exclusion window around the peak (with circular wrap).
    [H, W] = size(C);
    [yy, xx] = ndgrid(1:H, 1:W);
    dy = min(abs(yy - py), H - abs(yy - py));
    dx = min(abs(xx - px), W - abs(xx - px));
    keep = (dy > r) | (dx > r);

    noiseEnergy = sum(C(keep).^2) / nnz(keep);
    pce = sign(C(py, px)) * peak^2 / (noiseEnergy + 1e-12);
end

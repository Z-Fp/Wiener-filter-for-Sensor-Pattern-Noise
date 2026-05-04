% Wiener-residual centroid classifier on the gray channel.
% Builds a per-device reference (centroid) from training residuals,
% then classifies test residuals by maximum normalised correlation.

close all; clear; clc;

% ---- paths ----
trainRoot = "data/train/";
testRoot  = "data/test/";
outFile   = "results/Wiener_Centroids_Gray.mat";

epsVal = 1e-12;

% ---- training: build one centroid per class ----
fprintf("Computing Wiener centroids (gray)\n");

labelFolders = dir(trainRoot);
labelFolders = labelFolders([labelFolders.isdir] & ~startsWith({labelFolders.name}, '.'));

numClasses = numel(labelFolders);
classNames = strings(numClasses, 1);
mu = [];

for c = 1:numClasses
    labelName = labelFolders(c).name;
    files = dir(fullfile(trainRoot, labelName, '*.mat'));
    numFiles = numel(files);

    fprintf("  class %2d/%d: %s (%d files)\n", c, numClasses, labelName, numFiles);

    sumResidual = [];
    for k = 1:numFiles
        data = load(fullfile(trainRoot, labelName, files(k).name));
        Z = double(data.residual);
        Z = mean(Z, 3);                 % RGB -> gray
        Z = Z - mean(Z, 'all');         % zero-mean

        if isempty(sumResidual)
            sumResidual = zeros(size(Z));
        end
        sumResidual = sumResidual + Z;
    end

    muC = sumResidual / numFiles;
    muC = muC - mean(muC, 'all');
    muC = muC / sqrt(sum(muC.^2, 'all') + epsVal);

    if isempty(mu)
        mu = zeros([size(muC), numClasses]);
    end
    mu(:,:,c) = muC;
    classNames(c) = string(labelName);
end

if ~exist("results", "dir"); mkdir("results"); end
save(outFile, "mu", "classNames", "-v7.3");

% ---- testing: NCC against every centroid ----
fprintf("\nTesting Wiener correlation (gray)\n");

dsTest = fileDatastore(testRoot, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.mat', ...
    'ReadFcn', @load);

numTest = numel(dsTest.Files);
allPredIdx = zeros(numTest, 1);
allTrueIdx = zeros(numTest, 1);

for i = 1:numTest
    data = load(dsTest.Files{i});
    Z = double(data.residual);
    Z = mean(Z, 3);
    Z = Z - mean(Z, 'all');
    Z = Z / sqrt(sum(Z.^2, 'all') + epsVal);

    corrScores = zeros(numClasses, 1);
    for c = 1:numClasses
        corrScores(c) = sum(Z .* mu(:,:,c), 'all');
    end
    [~, predIdx] = max(corrScores);

    [~, labelName] = fileparts(fileparts(dsTest.Files{i}));
    allPredIdx(i) = predIdx;
    allTrueIdx(i) = find(classNames == string(labelName));
end

acc = mean(allPredIdx == allTrueIdx);
fprintf("\nGray Wiener accuracy: %.2f%%\n", acc * 100);

figure;
confusionchart(allTrueIdx, allPredIdx);
title(sprintf("Gray Wiener Accuracy: %.2f%%", acc * 100));

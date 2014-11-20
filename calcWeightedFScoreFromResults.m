function [ weightedFScore ] = calcWeightedFScoreFromResults( predicted, actual )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    confusionMatrix = confusionmat(predicted, actual);
    weightedFScore = calcWeightedFScore(confusionMatrix);
end


function [ result ] = DT_selector( Xtrain,Ytrain,Xtest,Ytest )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    tree = fitctree(Xtrain, Ytrain);
    prediction = tree.predict(Xtest);
    result = sum(Ytest ~= prediction);
end


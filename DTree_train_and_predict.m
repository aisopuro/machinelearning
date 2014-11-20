function [ prediction ] = DTree_train_and_predict( training_data, training_targets, test_data )
%UNTITLED4 Summary of this function goes here
    tree = fitctree(training_data, training_targets);
    prediction = tree.predict(test_data);
end


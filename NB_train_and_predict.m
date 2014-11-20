function [ prediction ] = NB_train_and_predict( training_data, training_values, test_data )
%Return a column matrix with classes predicted by a Naive Bayes classifier
%   Detailed explanation goes here
    NB_classifier = fitNaiveBayes(training_data, training_values);
    [prediction] = NB_classifier.predict(test_data);
end


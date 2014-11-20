function [ weightedFScore ] = calcWeightedFScore( confusionMatrix )
%Calculate the weighted F-Score for the given confusion matrix.
    true_positives = 0;
    false_positives = 0;
    false_negatives = 0;
    for i = 1:length(confusionMatrix)
        should_have_been_i = confusionMatrix(i,:);
        predicted_as_i = confusionMatrix(:,i);
        true_positives = true_positives + predicted_as_i(i);
        false_positives = false_positives + sum(predicted_as_i) - predicted_as_i(i);
        false_negatives = false_negatives + sum(should_have_been_i) - should_have_been_i(i);
    end
    precision = true_positives / (true_positives + false_positives);
    recall = true_positives / (true_positives + false_negatives);
    weightedFScore = 2 * precision * recall / (precision + recall);
end


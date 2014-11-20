function [ prediction ] = ANN_train_and_predict( training_data, training_targets, test_data, class_count )
%UNTITLED5 Summary of this function goes here
    X = transpose(training_data);
    % The neural net requires classes be given as binary row indexes:
    % A 1 in the row indicates a class of that row's number
    % A 7 X 1 binary column then becomes a 2 X 7 matrix
    T = zeros(length(training_targets), class_count);
    for i = 1:length(T)
        T(i,training_targets(i) + 1) = 1;
    end
    T = T';
    net = patternnet();
    net = configure(net, X, T);
    [trained_net,tr] = train(net, X, T);
    test_output = trained_net(test_data');
    prediction = vec2ind(test_output)' - 1;
end


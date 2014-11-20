function [result]=ANN_selector(Xtrain,Ytrain,Xtest,Ytest)
    net = patternnet(30);
    X = transpose(Xtrain);
    offset = 0;
    if (min(Ytrain) == 0)
        offset = 1;
    end
    T = zeros(length(Ytrain), range(Ytrain) + 1);
    for i = 1:length(T)
        T(i,Ytrain(i) + offset) = 1;
    end
    red_column = Ytrain;
    T = transpose(red_column);
    net = configure(net, X, T);
    [trained_net,tr] = train(net, X, T);
    result = sum(Ytest ~= vec2ind(trained_net(Xtest')'));
end
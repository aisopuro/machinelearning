TIMESTORUN = 10;

training_set = readtable('../data/T-61_3050_training_dataset_modified.csv');
testing_set = readtable('../data/test_dataset_modified.csv');

training_features = training_set{:, 1:11};
training_quality = training_set{:, 12};
training_type = training_set{:, 14};

testing_features = testing_set{:, 1:11};
testing_quality = testing_set{:, 12};
testing_type = testing_set{:, 14};

boosted_training = [training_features training_type];

c2 = cvpartition(training_quality,'k',10);
opts = statset('display','iter');
NB_selector = @(Xtrain,Ytrain,Xtest,Ytest)...
    sum(Ytest ~= predict(NaiveBayes.fit(Xtrain, Ytrain), Xtest));

[fs, history] = sequentialfs(NB_selector, training_features, training_type, 'cv', c2, 'options', opts);
NB_selected_type_features = training_features(:,fs);
NB_selected_type_targets = testing_features(:,fs);

[fs, history] = sequentialfs(NB_selector, training_features, training_quality, 'cv', c2, 'options', opts);
NB_selected_quality_features = training_features(:,fs);
NB_selected_quality_targets = testing_features(:,fs);

[DT_quality_selections, history] = sequentialfs(@DT_selector, training_features, training_quality, 'cv', c2, 'options', opts);
[DT_type_selections, history] = sequentialfs(@DT_selector, training_features, training_type, 'cv', c2, 'options', opts);

% ANN does not successfully select features
% [ANN_type_selections, history] = sequentialfs(@ANN_selector, training_features, training_type, 'cv', c2, 'options', opts);

% Baselines
random_types = zeros(2) + 250;
random_quality = zeros(7) + 20;
fscore_low = calcWeightedFScore(random_quality);
fscore_high = calcWeightedFScore(random_quality + 1);

NB_type_FScores = zeros(TIMESTORUN, 1);
NB_quality_FScores = zeros(TIMESTORUN, 1);
DTree_type_FScores = zeros(TIMESTORUN, 1);
DTree_quality_FScores = zeros(TIMESTORUN, 1);
boosted_DTree_quality_FScores = zeros(TIMESTORUN, 1);
ANN_type_FScores = zeros(TIMESTORUN, 1);
ANN_quality_FScores = zeros(TIMESTORUN, 1);
ANN_boosted_quality_fscores = zeros(TIMESTORUN, 1);

% Naive Bayes
NB_type_predictions = NB_train_and_predict(training_features, training_type, testing_features);
NB_quality_predictions = NB_train_and_predict(training_features, training_quality, testing_features);
NB_type_FScore = calcWeightedFScoreFromResults(NB_type_predictions, testing_type);
NB_quality_FScore = calcWeightedFScoreFromResults(NB_quality_predictions, testing_quality);

% NB with selection
NB_selected_type_predictions = NB_train_and_predict(NB_selected_type_features, training_type, NB_selected_type_targets);
NB_selected_quality_predictions = NB_train_and_predict(NB_selected_quality_features, training_quality, NB_selected_quality_targets);
NB_s_type_FScore = calcWeightedFScoreFromResults(NB_selected_type_predictions, testing_type);
NB_s_quality_FScore = calcWeightedFScoreFromResults(NB_selected_quality_predictions, testing_quality);

% Decision Tree
DTree_types = DTree_train_and_predict(training_features, training_type, testing_features);
DTree_qualitites = DTree_train_and_predict(training_features, training_quality, testing_features);
DTree_type_FScore = calcWeightedFScoreFromResults(DTree_types,testing_type);
DTree_quality_FScore = calcWeightedFScoreFromResults(DTree_qualitites, testing_quality);

% DTree with selection
DTree_selected_type_predictions = DTree_train_and_predict(training_features(:,DT_type_selections), training_type, testing_features(:,DT_type_selections));
DTree_selected_quality_predictions = DTree_train_and_predict(training_features(:,DT_quality_selections), training_quality, testing_features(:,DT_quality_selections));
DTree_s_type_FScore = calcWeightedFScoreFromResults(DTree_selected_type_predictions, testing_type);
DTree_s_quality_FScore = calcWeightedFScoreFromResults(DTree_selected_quality_predictions, testing_quality);

% Boost DTree by including type predictions in training data
boosted_features = [testing_features DTree_types];
DT_boosted_quality_predicitions = DTree_train_and_predict(boosted_training, training_quality, boosted_features);
boosted_DTree_quality_FScore = calcWeightedFScoreFromResults(DT_boosted_quality_predicitions, testing_quality);
for i = 1:TIMESTORUN
    % Adaptive Neural Network
    ANN_types = ANN_train_and_predict(training_features, training_type, testing_features, 2);
    ANN_qualitites = ANN_train_and_predict(training_features, training_quality, testing_features, 7);
    ANN_type_FScores(i) = calcWeightedFScoreFromResults(ANN_types,testing_type);
    ANN_quality_FScores(i) = calcWeightedFScoreFromResults(ANN_qualitites, testing_quality);
    
    ANN_boosted_quality_predicitions = ANN_train_and_predict(boosted_training, training_quality, [testing_features ANN_types], 7);
    ANN_boosted_quality_fscores(i) = calcWeightedFScoreFromResults(ANN_boosted_quality_predicitions, testing_quality);
end

% disp(calcWeightedFScore(random_types));
% disp((fscore_high + fscore_low) / 2);
% disp(NB_type_FScore);
% disp(DTree_type_FScore);
% disp(boosted_DTree_quality_FScore);
ANN_t_mean = mean(ANN_type_FScores);
ANN_q_mean = mean(ANN_quality_FScores);
result = [
    calcWeightedFScore(random_types) (fscore_high + fscore_low) / 2; 
    NB_type_FScore NB_quality_FScore; 
    NB_s_type_FScore NB_s_quality_FScore; 
    DTree_type_FScore DTree_quality_FScore;
    DTree_s_type_FScore DTree_s_quality_FScore;
    0 boosted_DTree_quality_FScore; 
    ANN_t_mean ANN_q_mean;
    0 mean(ANN_boosted_quality_fscores)
];
disp(result);
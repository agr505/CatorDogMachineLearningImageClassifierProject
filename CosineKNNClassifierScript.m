%% CAP 4630 - Intro to AI - FAU - Dr. Marques - Fall 2016 
% Group number 12
% Members: Ernso Jean-Louis, Willene Nazaire, Aaron Reich

%% Download/load Pre-trained Convolutional Neural Network (CNN)

% 1: Location of pre-trained "AlexNet"
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
% Specify folder for storing CNN model 
cnnFolder = './networks';
cnnMatFile = 'imagenet-caffe-alex.mat'; 
cnnFullMatFile = fullfile(cnnFolder, cnnMatFile);

% Check that the code is only downloaded once
if ~exist(cnnFullMatFile, 'file')
    disp('Downloading pre-trained CNN model...');     
    websave(cnnFullMatFile, cnnURL);
end

% Load MatConvNet network into a SeriesNetwork
convnet = helperImportMatConvNet(cnnFullMatFile);

%% Set up image data
% Load simplified dataset and build image store
dataFolder = './data/SomePetImages';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);

% Use the smallest overlap set
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

%% Pre-process Images For CNN
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% Divide data into training and testing sets
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

%% Feature Extraction 
% Get the network weights for the second convolutional layer
w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Use features from one of the deeper layers
featureLayer = 'fc7';

trainingFeaturesFolder = './';
trainingFeaturesFile = 'trainingSomeFeatures.mat'; 
trainingFeaturesFullMatFile = fullfile(trainingFeaturesFolder, trainingFeaturesFile);

% Check that the code is only downloaded once
if ~exist(trainingFeaturesFullMatFile, 'file')
    disp('Building training features... This will take a while...'); 
    trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
      'MiniBatchSize', 32, 'OutputAs', 'columns');
    save(trainingFeaturesFullMatFile, 'trainingFeatures');
else
    load trainingSomeBlurryFeatures.mat
end

% Extract features from images in the test set
testFeaturesFolder = './';
testFeaturesFile = 'testSomeFeatures.mat'; 
testFeaturesFullMatFile = fullfile(testFeaturesFolder, testFeaturesFile);

% Check that the code is only downloaded once
if ~exist(testFeaturesFullMatFile, 'file')
    disp('Extracting features... This will take a while...');     
    % Extract test features using the CNN
    testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);
    % Save features for future use
    save(testFeaturesFullMatFile, 'testFeatures');
else
    load testSomeBlurryFeatures.mat
end

%% Load model
load CosineKNNClassifier.mat

rng(32);
%% Train classifier

Model = CosineKNNClassifier.ClassificationKNN;

predictedLabels = predict(Model, trainingFeatures);

% Get the known labels
trainingLabels = trainingSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(trainingLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

%% Display and Compute ROC curve

% Compute the ROC curve.

% Label the test sample observations.  
% Display the results for the observations in the test sample.
[label_train,score_train] = predict(Model,trainingFeatures);
[X_train,Y_train,T_train,AUC_train] = perfcurve(trainingLabels,score_train(:,1),'cat'); 

% Plot the ROC curve
figure(1), plot(X_train,Y_train)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by KNN, Training Data Set')

disp(AUC_train)
%% Test classifier's prediction accuracy and produce confusion matrix

predictedLabels1 = predict(Model, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat1 = confusionmat(testLabels, predictedLabels1);

% Convert confusion matrix into percentage form
confMat1 = bsxfun(@rdivide,confMat1,sum(confMat1,2)) 

%% Display and Compute ROC curve

% Compute the ROC curve.

% Label the test sample observations.  
% Display the results for the observations in the test sample.
[label_test,score_test] = predict(Model,testFeatures);
[X_test,Y_test,T_test,AUC_test] = perfcurve(testLabels,score_test(:,1),'cat'); 

% Plot the ROC curve
figure(2), plot(X_test,Y_test)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by KNN, Test Data Set')

disp(AUC_test)
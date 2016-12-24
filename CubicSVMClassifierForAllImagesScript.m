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
dataFolder = './data/train';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);

% Use the smallest overlap set
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

%% Pre-process Images For CNN
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% Feature Extraction 
% Get the network weights for the second convolutional layer
w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Use features from one of the deeper layers
featureLayer = 'fc7';

allFeaturesFolder = './';
allFeaturesFile = 'allFeaturesWithBlur25000.mat'; 
allFeaturesFullMatFile = fullfile(allFeaturesFolder, allFeaturesFile);

% Check that the code is only downloaded once
if ~exist(allFeaturesFullMatFile, 'file')
    disp('Building training features... This will take a really long time...');     
    allFeatures = activations(convnet, imds, featureLayer, ...
      'MiniBatchSize', 32, 'OutputAs', 'columns');
    save(allFeaturesFullMatFile, 'allFeatures');
else
    load allFeaturesWithBlur25000.mat
end

%% Prepare data

rng(1);

% Matrix for model
allLabels = imds.Labels;
totalLabels = table(allLabels);
allFeaturesTable = table(allFeatures');
totalTable = [allFeaturesTable totalLabels];

% Partition the data
holdoutCVP = cvpartition(allLabels,'holdout',0.4);

featuresTrain = allFeaturesTable(holdoutCVP.training,:);
labelsTrain = allLabels(holdoutCVP.training);
featuresTest = table(allFeaturesTable(holdoutCVP.test,:));
labelsTest = allLabels(holdoutCVP.test);

% Set up matrices
trainMatrix = totalTable(holdoutCVP.training,:);
testMatrix = totalTable(holdoutCVP.test,:);

%% Train model
[trainingClassifierCubicSVM, validationAccuracy] = trainedSVMClassifier(trainMatrix);

% Preform cross validation
CVSVMModel1 = crossval(trainingClassifierCubicSVM.ClassificationSVM,'kfold',10);

% Get metrics
classLoss1 = kfoldLoss(CVSVMModel1)
validationAccuracy1 = 1 - classLoss1
[validationPredictions1, validationScores1] = kfoldPredict(CVSVMModel1);

% Display confusion matrix
[Conf_Mat,order] = confusionmat(labelsTrain,validationPredictions1);
disp(Conf_Mat)

% Explain what happens here
SVMModel = fitPosterior(trainingClassifierCubicSVM.ClassificationSVM);
[~,score_svm] = resubPredict(SVMModel);
[X2,Y2,T,train_AUC] = perfcurve(allLabels(holdoutCVP.training),score_svm(:,1),'cat'); 

% Plot the ROC curve
figure(1), plot(X2,Y2)
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for SVM, Training dataset')

% Display the area under the curve.
disp(train_AUC)

%% Test model

[testingClassifierCubicSVM, testValidationAccuracy] = trainClassifier(testMatrix);

% Preform cross validation
CVSVMModel2 = crossval(testingClassifierCubicSVM.ClassificationSVM,'kfold',10);

% Get metrics
classLoss2 = kfoldLoss(CVSVMModel2)
validationAccuracy2 = 1 - classLoss2
[validationPredictions2, validationScores2] = kfoldPredict(CVSVMModel2);

% Display confusion matrix
[Conf_Mat2,order] = confusionmat(labelsTest,validationPredictions2);
disp(Conf_Mat2)

% Explain what happens here
SVMModel2 = fitPosterior(testingClassifierCubicSVM.ClassificationSVM);
[~,score_svm] = resubPredict(SVMModel2);
[X2,Y2,T,test_AUC] = perfcurve(allLabels(holdoutCVP.test),score_svm(:,1),'cat'); 

% Plot the ROC curve
figure(2), plot(X2,Y2)
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for SVM, Testing dataset')

% Display the area under the curve.
disp(test_AUC)
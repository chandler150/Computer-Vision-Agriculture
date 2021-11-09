%% Transfer Learning -1
% parpool(2); %% Use multiple CPUs/GPUs to train the model
% load image data from the folder Images/train
imagepath = fullfile('Images', 'train');
% use the image folder names as the classification labels
imds = imageDatastore(imagepath, 'IncludeSubfolders', true,'LabelSource', 'FolderNames');
% Anonymous Functions create simple functions without having to% create a file for them
% Resize the input images to meet the input image size% requirements
imds.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
% Split inputs into training and validation sets
[trainDS, valDS] = splitEachLabel (imds, 0.7, 0.3,'randomized');
% train the network
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valDS, ...
    'ValidationFrequency',3,...
    'Verbose',false, ...
    'Plots','training-progress');
%'ExecutionEnvironment', 'parallel';
trainedFlowerNet = trainNetwork(trainDS,flowerNet,options);

%% Transfer Learning -1 continued
I = imread("daisy.jpg");
I = imresize(I,[224 224]);
% Classify the image using flowerNet
% I is a 227 X 227 X3 flower image
label = classify(trainedFlowerNet,I)

%% Transfer Learning -2
% parpool(2); %% Use multiple CPUs/GPUs to train the model
% load image data from the folder Images/train
imagepath = fullfile('Images2', 'train');
% use the image folder names as the classification labels
imds = imageDatastore(imagepath,'IncludeSubfolders',true,'LabelSource','FolderNames');
% Anonymous Functions create simple functions without having to
% create a file for them% Resize theinput images to meet the input image size
% requirements
imds.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
% Split inputs into training and validation sets
[trainDS, valDS] = splitEachLabel (imds, 0.7, 0.3,'randomized');
% train the network
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valDS, ...
    'ValidationFrequency',3,...
    'Verbose',false, ...
    'Plots','training-progress');%, ...
    %'ExecutionEnvironment', 'gpu');
trainedFlowerNet = trainNetwork(trainDS,flowerNet2,options);

%% Transfer Learning -1 continued
I = imread("tulip2.jpg");
I = imresize(I,[224 224]);
% Classify the image using flowerNet
% I is a 227 X 227 X3 flower image
label = classify(trainedFlowerNet,I)
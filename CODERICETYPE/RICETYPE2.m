%% CLASSIFYING THE TYPES OF RICE GRAINS USING DEEP CNN
% Clearing the workspace
clc
clearvars
close all
% Load image data
imds = imageDatastore('RICETYPE2', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split data into training, validation, and test sets
[trainImgs, valImgs, testImgs] = splitEachLabel(imds, 0.6, 0.2, 0.2, 'randomized');

% Create CNN architecture
layers = [
    imageInputLayer([250 250 3])
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(5) % 5 classes
    softmaxLayer
    classificationLayer
];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 32, ...
    'ValidationData', valImgs, ...
    'ValidationFrequency', 5, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(trainImgs, layers, options);
% INPUT IMAGE
[filename, pathname] = uigetfile(...    
    {'*.jpg','Supported Files (*.jpg,*.img,*.tiff,)'; ...
    '*.JPG','jpg Files(*.jpg)'},...
    'MultiSelect', 'on');
im=imread(filename);%Read any colorimage 
    figure,imshow(im) 
    subplot(2,1,1),imshow(im);
title('Original Grain Image');
subplot(2,1,2),imhist(im(:,:,1));
title('INPUT IMAGE HISTOGRAM');
I = rgb2gray(im);
figure
imshow(I)
I = imnoise(rgb2gray(im),'salt & pepper',0.02);
subplot(1,2,1),imshow(I);
title('Noise addition');
K = medfilt2(I);
subplot(1,2,2),imshow(K);
title('Noise removal using median filter');
tic
level = graythresh(K);
BW = imbinarize(K,level);
imshowpair(K,BW,'montage')
T = adaptthresh(K, 0.4);
BW = imbinarize(K,T);
figure
imshowpair(K,BW, 'montage')
%pause(0.5)
[B,L] = bwboundaries(BW,'noholes');
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on

for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
end
captionFontSize = 14;
%% Convert RGB2Grayscale
originalImage = imread(filename);
originalImage = rgb2gray(originalImage);
subplot(2, 3, 1);
imshow(originalImage);
title('Original Image', 'FontSize', captionFontSize); 
%% Explore the image/data using histogram:
[pixelCount, grayLevels] = imhist(originalImage);
subplot(2, 3, 2);
bar(pixelCount);
xlim([0 grayLevels(end)]); % Scale x axis manually.
grid on;
thresholdValue = 90; % Choose the threshold value in ORder to mask out the background noise.
% Show the threshold as a vertical red bar on the histogram.
hold on;
maxYValue = ylim;
line([thresholdValue, thresholdValue], maxYValue, 'Color', 'r');
annotationText = sprintf('Thresholded at %d gray levels', thresholdValue);
text(double(thresholdValue + 5), double(0.5 * maxYValue(2)), annotationText, 'FontSize', 10, 'Color', [0 .5 0]);
text(double(thresholdValue - 70), double(0.94 * maxYValue(2)), 'Background', 'FontSize', 10, 'Color', [0 0 .5]);
text(double(thresholdValue + 50), double(0.94 * maxYValue(2)), 'Foreground', 'FontSize', 10, 'Color', [0 0 .5]);

%% Get and show binary image
binaryImage = originalImage > thresholdValue;
binaryImage = imfill(binaryImage, 'holes');
% Display the binary image.
subplot(2, 3, 3);
imshow(binaryImage); 
title('Binary Image', 'FontSize', captionFontSize); 
%% Maskout the background noise
MaskedImage = originalImage;
MaskedImage(~binaryImage) = 0;
subplot(2, 3, 4);
imshow(MaskedImage); 
title('Masked Image', 'FontSize', captionFontSize); 
%% Get the centroid, mean intensity, permieter of the whole seed etc
blobMeasurements = regionprops(binaryImage, originalImage, 'all');
%% Calculate the chalkiness by first exploring the histogram
[pixelCount_1, grayLevels_1] = imhist(MaskedImage);
thresholdValue_1 = 180; %% Choose the threshold that segments the chalkiness
binary_MaskedImage = MaskedImage > thresholdValue_1; 
binary_MaskedImage = imfill(binary_MaskedImage, 'holes');
%% Display the binary Masked  image.
subplot(2, 3, 5);
imshow(binary_MaskedImage); 
title('Binary Masked Image', 'FontSize', captionFontSize); 
%% Get the centroid, mean intensity, permieter of the chlky area
blobMeasurements_subblobs = regionprops(binary_MaskedImage, originalImage, 'all');
msgbox('Initiating Quality checking process')
pause(1.0)
% Plot the borders of all the seeds on the original grayscale image using the coordinates returned by bwboundaries.
subplot(2, 3, 6);
imshow(originalImage);
title('chalkiness', 'FontSize', captionFontSize); 
axis image; % Make sure image is not artificially stretched because of screen's aspect ratio.
hold on;
boundaries = bwboundaries(binaryImage);
numberOfBoundaries = size(boundaries, 1);
for k = 1 : numberOfBoundaries
	thisBoundary = boundaries{k};
	plot(thisBoundary(:,2), thisBoundary(:,1), 'g', 'LineWidth', 2);
end
boundaries_Masked = bwboundaries(binary_MaskedImage);
numberOfBoundaries_Masked = size(boundaries_Masked, 1);
for k = 1 : numberOfBoundaries_Masked
	thisBoundary_Masked = boundaries_Masked{k};
	plot(thisBoundary_Masked(:,2), thisBoundary_Masked(:,1), 'b', 'LineWidth', 2);
end
hold off;
%% Overlay the chalky area on the original image
f = imread(filename);
bw = im2bw(f);
%We emperically agreed on the follwoing values
T       = [0.1 0.5];
sigma   = 1; 
imedge = edge(bw, 'canny', T, sigma);
% Using imfill fill all the regions
% to get closed objects 
imf = imfill(imedge, 'holes');
% Label all the objects which are detected by BWlabel
[g n] = bwlabel(imf);
% using regionprops, get the properties of each object
% calculate the desired properties
% from region props
stats = regionprops(g);
stats = regionprops(bw,'Centroid', 'MajorAxisLength','MinorAxisLength')
averageHeight  = 0;
averageWidth = 0;
for i = 1:numel(stats);    
   averageHeight = averageHeight + stats(i).MajorAxisLength;
   averageWidth = averageWidth + stats(i).MinorAxisLength;   
end;
averageWidth = averageWidth/numel(stats);
averageHeight = averageHeight/numel(stats);
% collect the details for each detected rice grain from regionprops in
tabular = regionprops('table', bw,'Centroid', 'MajorAxisLength','MinorAxisLength');
display('Summary');
display(tabular);
display(sprintf('Number of objects in the image are %d', n));
display(sprintf('Average Width of each food grain is Width = %f', averageWidth));
display(sprintf('Average Height of each food grain is Height = %f', averageHeight));
%% Chalkiness
Sum_Subblobs_Perimeter = sum([blobMeasurements_subblobs(1:end).Perimeter]);
Sum_Blobs_Perimeter = sum([blobMeasurements(1:end).Perimeter]);
Percentage_Chalkiness=(Sum_Subblobs_Perimeter/Sum_Blobs_Perimeter)*100;
 %% Done classification 
     class = classify(net,im);%% Classification
     msgbox(char(class))
% Evaluate the network
YPred = classify(net, testImgs);
YTest = testImgs.Labels;

% Calculate confusion matrix
C = confusionmat(YTest, YPred);

% Plot confusion matrix
figure;
confusionchart(C, categories(imds.Labels));

% Calculate specificity, sensitivity, precision, recall, and F1-score
numClasses = size(C, 1);
specificity = zeros(numClasses, 1);
sensitivity = zeros(numClasses, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = C(i, i);
    FP = sum(C(:, i)) - TP;
    FN = sum(C(i, :)) - TP;
    TN = sum(C(:)) - TP - FP - FN;
    
    specificity(i) = TN / (TN + FP);
    sensitivity(i) = TP / (TP + FN);
    precision(i) = TP / (TP + FP);
    recall(i) = sensitivity(i);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Display metrics
disp('Confusion Matrix:');
disp(C);
disp('Specificity:');
disp(specificity);
disp('Sensitivity:');
disp(sensitivity);
disp('Precision:');
disp(precision);
disp('Recall:');
disp(recall);
disp('F1-Score:');
disp(f1Score);
% Display a message box indicating successful completion
msg = 'CLASSIFICATION OF TYPE OF RICE SUCCESSFULLY COMPLETED USING CNN';
fprintf('%s\n', msg);
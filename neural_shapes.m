%Classifying images of Geometric shapes using neural network
%author: Sourabh Garg

%Importing inputs and targets
x = P_C;
t = shapeslabel;

%Creating Training set(17 images of each geometric shapes)
train_x=[x(:,1:17),x(:,21:37),x(:,41:57)];
train_t=[t(:,1:17),t(:,21:37),t(:,41:57)];
size(train_x);
size(train_t);
%Creating Testing set(3 images of each geometric shapes)
test_x=[x(:,18:20),x(:,38:40),x(:,58:60)];
test_t=[t(:,18:20),t(:,38:40),t(:,58:60)];
size(test_x);
size(test_t);

% Choose a Training Function: 'trainlm','trainbr','trainscg'
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
%one hidden layer with 10 neurons
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
% net.divideFcn = 'dividerand';  % Divide data randomly
% net.divideMode = 'sample';  % Divide up every sample
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;

% Choose a Performance Function(help nnperformance)
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions(help nnplot)
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
               'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,train_x,train_t);
train_y = net(train_x);
e = gsubtract(train_t,train_y);
performance = perform(net,train_t,train_y);
train_t_label = vec2ind(train_t)
train_y_label = vec2ind(train_y)
percentErrors = sum(train_t_label ~= train_y_label)/numel(train_y_label)

% Test the network
test_y=net(test_x);
test_y_label=vec2ind(test_y)
test_t_label=vec2ind(test_t)
percentErrors_test = sum(test_t_label ~= test_y_label)/numel(test_y_label)
% % Recalculate Training, Validation and Test Performance
% trainTargets = t .* tr.trainMask{1};
% valTargets = t .* tr.valMask{1};
% testTargets = t .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,y)
% valPerformance = perform(net,valTargets,y)
% testPerformance = perform(net,testTargets,y)

% View the Network
view(net);

% Plots
% Uncomment these lines to enable various plots.(some modification will be required accordingly)
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)


%Extracting features from images and applying PCA for dimensionality reduction
%author: Sourabh Garg

clear
size_PC=zeros(1,60);
P_C=zeros(320,60);
for i = 1:60
    %Reading image one by one 
    %convert rgb image into grayscale
    I1 = rgb2gray(imread(strcat('shapes/',strcat(int2str(i),'.png'))));
    
    %features extraction using SURF algorithms
    points1 = detectSURFFeatures(I1);
    [features1,valid_points1] = extractFeatures(I1,points1);

    %Dimensionality reduction of extracted features using PCA 
    [pca1,zscore]=pca(features1);

    %Converting matrix into 1-dimension vector of size 320(5 Principal components)
    %Padding the smaller sized vector with '0'
    B = reshape(pca1,[],1);
    B(320)=0;

    %Transformed inputs for classification 
    P_C(:,i)=B(1:320);
    size_PC(i)=size(pca1,2);

end


size_PC
P_C
size(P_C);
size(size_PC)

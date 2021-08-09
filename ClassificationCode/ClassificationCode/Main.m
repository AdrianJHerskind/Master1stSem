clc;
clear all;
%open picture
[P, cmap]=imread('PicWithGT.png');%picture of the field
p2=imread('GroundThruth23G.bmp');%groundtruth labelled by the first interpreter
p3=imread('GroundThruth23.bmp');%groundtruth labelled by the second interpreter
%transform in HSV color space
HSV=(rgb2hsv(P(:,:,1:3)));
[nrows,ncolumns,~]=size(HSV);
%CREATION OF MASK USING HUE CHANNEL
mask=cast(zeros(nrows,ncolumns),'uint8');
for i=1:nrows
    for j=1:ncolumns
        if((HSV(i,j,1)>0.18)&&(HSV(i,j,1)<0.44)&&(HSV(i,j,2)>0.15))
            mask(i,j)=1;
        else
            mask(i,j)=0;
        end
    end
end
%APPLYING MASK
KmeansImage=cast(zeros(nrows,ncolumns,3),'uint8');
for i=1:3
    KmeansImage(:,:,i)=mask(:,:).*P(:,:,i);
end
%Transform immage in UVW color space
X = 0.49*double(KmeansImage(:,:,1)) + 0.310*double(KmeansImage(:,:,2)) + 0.200*double(KmeansImage(:,:,3));
Y= 0.177 *double(KmeansImage(:,:,1)) + 0.813 *double (KmeansImage(:,:,2)) + 0.011*double(KmeansImage(:,:,3));
Z = 0.010*double(KmeansImage(:,:,2)) + 0.990*double(KmeansImage(:,:,3));
U = cast((2/3)*X,'uint8');
V = cast(Y ,'uint8');
W = cast(1/2*(-X+3*Y+Z),'uint8');
UVW=cast(zeros(nrows,ncolumns,3),'uint8');
UVW(:,:,1)=U;
UVW(:,:,2)=V;
UVW(:,:,3)=W;
%Classification using k means, k=3 background, weed and crops
KM=imsegkmeans(UVW(:,:,1:3),3);
%Creation of grid
%parameters to be inputed or calculated
gridlengthmeters=3;
groundres=0.025;
treshperc=0.0001;
selclass=2;
%create grid and saves a matrix with the "percentages of pixels of the
%researched class" PM
[result,PM]=gridmaker(KM,selclass,treshperc,gridlengthmeters,groundres);
%[result]=gridmaker(KM,selclass,treshperc,gridlengthmeters,groundres);

%Remove part of the grid that are not part of the actual immage (WORKS
%ONLY FOR .TIF)
%result=result.*P(:,:,4)./255;
%printing out the immage with the "grid"
B = labeloverlay(P(:,:,1:3),result);
figure();
imshow(B);
title('result');
%Calculating the area that is going to be sprayed and the total area of
%the field
tosprayareainpixel=nnz(result);
tosprayareainm=tosprayareainpixel*groundres*groundres;
[gridhit,optimalgrid,hitsperc,totmiss]=validation(result,gridlengthmeters,groundres,p2,p3);
%works only for TIFF
%totfieldarea=nnz(P(:,:,4))*groundres*groundres;
totfieldarea = nnz(sum(P, 3));

OptimalGridToSpray = nnz(optimalgrid);
totmissinmeters = totmiss * groundres * groundres;
PercOfSprField = (tosprayareainpixel/totfieldarea) *100
OptimalSpraying = (OptimalGridToSpray/totfieldarea) *100

%evaluation on cropped image

%Coordinates For Testing
x1=5721;
y1=999;
x2=5953
y2=1154;
I7=imcrop(P,[x1 y1 x2-x1 y2-y1]);figure;imshow(I7);
I3 = imcrop(KM,[x1 y1 x2-x1 y2-y1]);figure;imshow(I3*75);
I4=imcrop(GT,[x1 y1 x2-x1 y2-y1]);figure;imshow(I4*75);
%  I5=imcrop(result,[x1 y1 x2-x1 y2-y1]);figure;imshow(I5*75);
% I5=imcrop(HSV,[x1 y1 x2-x1 y2-y1]);figure;imshow(I5);
%NumberOfPixelsInCroppedImage= numel(I3)
%Calculating the results for the classifier
TP= nnz(I3==3&I4==1);
FP= nnz(I3==3&I4~=1);
FN = nnz(I3~=3&I4==1);
TN = nnz(I3~=3&I4~=1);
?

% TN = nnz(KM~=3&GT~=1&sum(P,3)>0);
% FP= nnz(IM==3&GT~=1&sum(P,3)>0);
%SumOfResults = TP+FP+FN+TN
TruePositiveRate = TP/(TP+FN)
FalseNegativeRate= FN/(FN+TP)
TrueNegativeRate = TN/(FP+TN)
FalsePositiveRate = FP/(FP+TN)




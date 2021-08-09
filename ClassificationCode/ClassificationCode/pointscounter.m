p2=imread('GTSecondAttemptOpendronemapGiorgioALLGT.bmp');
p2=imbinarize(p2(:,:,1));
p3=imread('GTSecondAttemptOpendronemapKarolina.bmp');
p3=imbinarize(p3(:,:,1));
[w,h]=size(p2);
a=zeros(w,h,3);
a(:,:,1)=p2.*p3;
a(:,:,2)=abs((((p2-a(:,:,1))+1).^2)-1);
a(:,:,3)=abs((((p3-a(:,:,1))+1).^2)-1);
imshow(a);
[labeledImage, numberOfObject] = bwlabel(a(:,:,1),8);
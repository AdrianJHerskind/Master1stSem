function [gridhit,optimalgrid,hitsperc,totmiss]=validation(resultgrid,gridlengthmeters,groundres,p2,p3)
%p2=imread('GroundThruth23G.bmp');
p2=imbinarize(p2(:,:,1));
%p3=imread('GroundThruth23.bmp');
p3=imbinarize(p3(:,:,1));
a=cast(p2.*p3,'uint8');
totmiss=nnz(a-(a.*resultgrid));
hitsperc=(nnz(a)-totmiss)/nnz(a);

[nrows,ncolumns]=size(resultgrid);
pixgridlength=floor(gridlengthmeters/groundres);
heigth=pixgridlength;
width=pixgridlength;
gridhit=cast(zeros(nrows,ncolumns),'uint8');
optimalgrid=cast(zeros(nrows,ncolumns),'uint8');
for ii=1:pixgridlength:nrows
        if(ii+pixgridlength>nrows)
            heigth=nrows-ii;
        end
        
for jj=1:pixgridlength:ncolumns
    if(jj+pixgridlength>ncolumns)
        width=1+ncolumns-jj;
    end
    flag=0;
    for i=1:heigth
        for j=1:width
            if(a(i+ii-1,j+jj-1)==1 && resultgrid(i+ii-1,j+jj-1)==1)%check if pixel ij is 1 and it's detected
                gridhit(ii:(ii+heigth-1),jj:(jj+width-1))=ones(heigth,width);
                flag=1;
                
            end
            if(a(i+ii-1,j+jj-1)==1)
            optimalgrid(ii:(ii+heigth-1),jj:(jj+width-1))=ones(heigth,width);
            end
            if(flag)
                break;
            end
        end
        if(flag)
            break;
        end
    end
    
end
width=pixgridlength;
end

end
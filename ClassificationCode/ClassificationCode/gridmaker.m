%{ First attempt to create the grid
function [ImageGrid]=gridmaker(KM,selclass,treshperc,gridlengthmeters,groundres)
    [nrows,ncolumns]=size(KM);    
    pixgridlength=floor(gridlengthmeters/groundres);
    heigth=pixgridlength;
    width=pixgridlength;
    ImageGrid=cast(zeros(nrows,ncolumns),'uint8');
    for ii=1:pixgridlength:nrows
        if(ii+pixgridlength>nrows)
            heigth=nrows-ii;
        end
        
        for jj=1:pixgridlength:ncolumns
            if(jj+pixgridlength>ncolumns)
                width=1+ncolumns-jj;
            end
            pixtresh=heigth*width*treshperc;
            count=0;
            flag=0;
            for i=1:heigth
                for j=1:width
                    if(KM(i+ii-1,j+jj-1)==selclass)%check if pixel ij is in the one of the classes of weeds
                        count=count+1;
                        if(count>=pixtresh)
                            ImageGrid(ii:(ii+heigth-1),jj:(jj+width-1))=ones(heigth,width);
                            flag=1;
                        end
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
  %}  

function [ImageGrid,PercentageMatrix]=gridmaker(KM,selclass,treshperc,gridlengthmeters,groundres,PM)
[nrows,ncolumns]=size(KM);
pixgridlength=floor(gridlengthmeters/groundres);
h=ceil(nrows/pixgridlength);
w=ceil(ncolumns/pixgridlength);
PercentageMatrix=zeros(h,w);
ImageGrid=cast(zeros(nrows,ncolumns),'uint8');
%check if Percentage Matrix already exist
if exist('PM','var') == 1
    PercentageMatrix=PM;
else
    %creating percentage matrix
    for ii=1:h
        
        rstart=((ii-1)*pixgridlength)+1;
        rend=ii*pixgridlength;
        
        if(ii*pixgridlength>nrows)
            rend=nrows;
        end
        
        for jj=1:w
            
            cstart=((jj-1)*pixgridlength)+1;
            cend=jj*pixgridlength;
            
            if(jj*pixgridlength>ncolumns)
                cend=ncolumns;
            end
            %fills the value of the percentage matrix
            PercentageMatrix(ii,jj)=nnz(KM(rstart:rend,cstart:cend)==selclass)/((1+cend-cstart)*(1+rend-rstart));
        end
    end    
end
%filling grid
for i=1:h
    
    rstart=((i-1)*pixgridlength)+1;
    rend=i*pixgridlength;
    
    if(i*pixgridlength>nrows)
        rend=nrows;
    end
    
    for j=1:w
        
        cstart=((j-1)*pixgridlength)+1;
        cend=j*pixgridlength;
        
        if(j*pixgridlength>ncolumns)
            cend=ncolumns;
        end
        %filling the "grid mask"
        if (PercentageMatrix(i,j)>=treshperc)
            ImageGrid(rstart:rend,cstart:cend)=ones((1+rend-rstart),(1+cend-cstart));
        end
    end
end

end


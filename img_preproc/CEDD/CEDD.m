function [ DescriptorVector ] = CEDD(ImageRGB)

DescriptorVector(1:144) = 0; % CEDD

T0 = 14;
T1 = 0.68;
T2 = 0.98;
T3 = 0.98;
T = -1;

Compact = 0; %false

% ImageRGB = imread(image_source);
width = size(ImageRGB,2);
height = size(ImageRGB,1);

MeanRed = 0; MeanGreen = 0; MeanBlue = 0;
Edges(1:6) = 0;
NeighborhoodArea1 = 0; NeighborhoodArea2 = 0; NeighborhoodArea3 = 0; NeighborhoodArea4 = 0;
Mask1 = 0; Mask2 = 0; Mask3 = 0; Mask4 = 0; Mask5 = 0;

Max = 0;
PixelCount(1:2,1:2) = 0;

Fuzzy10BinResultTable(1:10) = 0;
Fuzzy24BinResultTable(1:24) = 0;

NumberOfBlocks=1600; 

Step_X = floor(width / sqrt(NumberOfBlocks));
Step_Y = floor(height / sqrt(NumberOfBlocks));
    
    if (mod(Step_X,2) ~= 0)
        Step_X = Step_X - 1;
    end
    if (mod(Step_Y,2) ~= 0)
       Step_Y = Step_Y - 1;
    end


%Thanasis

CororRed(1: Step_X*Step_Y) = 0;
CororGreen(1: Step_X*Step_Y) = 0;
CororBlue(1: Step_X*Step_Y) = 0;
CororRedTemp(1: Step_X*Step_Y) = 0;
CororGreenTemp(1: Step_X*Step_Y) = 0;
CororBlueTemp(1: Step_X*Step_Y) = 0;

% for i=1:height
%     for j=1:width
%         ImageGridRed(j,i) = double(ImageRGB(i,j,1));
%         ImageGridGreen(j,i) = double(ImageRGB(i,j,2));
%         ImageGridBlue(j,i) = double(ImageRGB(i,j,3));
%         ImageGrid(j,i) = 0.299 * double(ImageRGB(i,j,1)) + 0.587 * double(ImageRGB(i,j,2)) + 0.114 * double(ImageRGB(i,j,3));
%     end
% end

ImageGridRed = double(ImageRGB(:,:,1))';
ImageGridGreen = double(ImageRGB(:,:,2))';
ImageGridBlue = double(ImageRGB(:,:,3))';
% ImageGridRed = importdata('ImageGridRedC#.txt')';
% ImageGridGreen = importdata('ImageGridGreenC#.txt')';
% ImageGridBlue = importdata('ImageGridBlueC#.txt')';
ImageGrid = 0.299 * ImageGridRed + 0.587 * ImageGridGreen + 0.114 * ImageGridBlue;

% ImageGridRed(1:width,1:height) = double(ImageRGB(1:height,1:width,1));
% ImageGridGreen(1:width,1:height) = double(ImageRGB(1:height,1:width,2));
% ImageGridBlue(1:width,1:height) = double(ImageRGB(1:height,1:width,3));
% ImageGrid(1:width,1:height) = 0.299 * double(ImageRGB(1:height,1:width,1)) + 0.587 * double(ImageRGB(1:height,1:width,2)) + 0.114 * double(ImageRGB(1:height,1:width,3));


  TemoMAX_X = Step_X * sqrt(NumberOfBlocks);
  TemoMAX_Y = Step_Y * sqrt(NumberOfBlocks);

for y= 1:Step_Y:TemoMAX_Y
    for x= 1:Step_X:TemoMAX_X
        MeanRed = 0; MeanGreen = 0; MeanBlue = 0;
        NeighborhoodArea1 = 0; NeighborhoodArea2 = 0; NeighborhoodArea3 = 0; NeighborhoodArea4 = 0;
        Edges(1:6) = -1;
        
        TempSum = 1;
        
        for i= 1:2
            for j= 1:2
                PixelCount(i,j) = 0;
            end
        end
        
        for i= y:1:y+Step_Y-1
            for j= x:1:x+Step_X-1
                CororRed(TempSum) = ImageGridRed(j, i);
                CororGreen(TempSum) = ImageGridGreen(j, i);
                CororBlue(TempSum) = ImageGridBlue(j, i);

                CororRedTemp(TempSum) = ImageGridRed(j, i);
                CororGreenTemp(TempSum) = ImageGridGreen(j, i);
                CororBlueTemp(TempSum) = ImageGridBlue(j, i);
                
                TempSum = TempSum+1;

                % Texture Information

                if (j < (x + Step_X / 2) && i < (y + Step_Y / 2))
                    NeighborhoodArea1 = NeighborhoodArea1 + ImageGrid(j, i);
                end
                if (j >= (x + Step_X / 2) && i < (y + Step_Y / 2)) 
                    NeighborhoodArea2 = NeighborhoodArea2 + ImageGrid(j, i);
                end
                if (j < (x + Step_X / 2) && i >= (y + Step_Y / 2)) 
                    NeighborhoodArea3 = NeighborhoodArea3 + ImageGrid(j, i);
                end
                if (j >= (x + Step_X / 2) && i >= (y + Step_Y / 2)) 
                    NeighborhoodArea4 = NeighborhoodArea4 + ImageGrid(j, i);
                end


            end
        end
        
        
        NeighborhoodArea1 = fix(NeighborhoodArea1 * (4.0 / (Step_X * Step_Y)));
        NeighborhoodArea2 = fix(NeighborhoodArea2 * (4.0 / (Step_X * Step_Y)));
        NeighborhoodArea3 = fix(NeighborhoodArea3 * (4.0 / (Step_X * Step_Y)));
        NeighborhoodArea4 = fix(NeighborhoodArea4 * (4.0 / (Step_X * Step_Y)));
        
        Mask1 = abs(NeighborhoodArea1 * 2 + NeighborhoodArea2 * (-2) + NeighborhoodArea3 * (-2) + NeighborhoodArea4 * 2);
        Mask2 = abs(NeighborhoodArea1 * 1 + NeighborhoodArea2 * 1 + NeighborhoodArea3 * (-1) + NeighborhoodArea4 * (-1));
        Mask3 = abs(NeighborhoodArea1 * 1 + NeighborhoodArea2 * (-1) + NeighborhoodArea3 * 1 + NeighborhoodArea4 * (-1));
        Mask4 = abs(NeighborhoodArea1 * sqrt(2) + NeighborhoodArea2 * 0 + NeighborhoodArea3 * 0 + NeighborhoodArea4 * (-sqrt(2)));
        Mask5 = abs(NeighborhoodArea1 * 0 + NeighborhoodArea2 * sqrt(2) + NeighborhoodArea3 * (-sqrt(2)) + NeighborhoodArea4 * 0);
        
        Max = max(Mask1,max(Mask2,max(Mask3,max(Mask4,Mask5))));
        
        Mask1 = Mask1 ./ Max;
        Mask2 = Mask2 ./ Max;
        Mask3 = Mask3 ./ Max;
        Mask4 = Mask4 ./ Max;
        Mask5 = Mask5 ./ Max;
        
        T = 0;
        
        if(Max < T0)
           Edges(1) = 0;
           T = 1;
        else
           T = 0;
           if(Mask1 > T1)
           T=T+1;
           Edges(T) = 1;
           end
           
           if(Mask2 > T2)
           T=T+1;
           Edges(T) = 2;
           end
           
           if(Mask3 > T2)
           T=T+1;
           Edges(T) = 3;
           end
           
           if(Mask4 > T3)
           T=T+1;
           Edges(T) = 4;
           end
           
           if(Mask5 > T3)
           T=T+1;
           Edges(T) = 5;
           end
            
            
        end
            
       
        for i =1:Step_Y * Step_X
           MeanRed = MeanRed + CororRed(i);
           MeanGreen = MeanGreen + CororGreen(i);
           MeanBlue = MeanBlue + CororBlue(i);
        end
        
%         MeanRed = round(MeanRed / (Step_Y * Step_X))/255.0;
%         MeanGreen = round(MeanGreen / (Step_Y * Step_X))/255.0;
%         MeanBlue = round(MeanBlue / (Step_Y * Step_X))/255.0;
        
        MeanRed = fix(MeanRed / (Step_Y * Step_X))/255.0;
        MeanGreen = fix(MeanGreen / (Step_Y * Step_X))/255.0;
        MeanBlue = fix(MeanBlue / (Step_Y * Step_X))/255.0;

        HSV = rgb2hsv([MeanRed MeanGreen MeanBlue]);
        
        HSV(1) = fix(HSV(1) * 360);
        HSV(2) = fix(HSV(2) * 255);
        HSV(3) = fix(HSV(3) * 255);
        
        
        if(Compact == 0)
            Fuzzy10BinResultTable = Fuzzy10Bin(HSV(1), HSV(2), HSV(3), 2); 
            Fuzzy24BinResultTable = Fuzzy24Bin(HSV(1), HSV(2), HSV(3), Fuzzy10BinResultTable, 2); 
            
            for i=1:T
               for j=1:24 
                   if(Fuzzy24BinResultTable(j) > 0)
                      DescriptorVector(24 * Edges(i) + j) = DescriptorVector(24 * Edges(i) + j) + Fuzzy24BinResultTable(j);
                   end
               end
            end
        else
            Fuzzy10BinResultTable = Fuzzy10Bin(HSV(1), HSV(2), HSV(3), 2);
            for i=1:T+1
               for j=1:10
                   if(Fuzzy10BinResultTable(j) > 0)
                      DescriptorVector(10 * Edges(i) + j) = DescriptorVector(10 * Edges(i) + j) + Fuzzy10BinResultTable(j);
                   end
               end
            end           
        end
    end
end

SUM = sum(DescriptorVector);
DescriptorVector(1:144) = DescriptorVector(1:144)/SUM;

DescriptorVector = CEDDQuant(DescriptorVector);

end


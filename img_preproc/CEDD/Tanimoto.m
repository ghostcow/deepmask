function [ Distance ] = Tanimoto( histogram1,histogram2 )

Temp1=sum(histogram1);
Temp2=sum(histogram2);
TempCount1=0;
TempCount2=0;
TempCount3=0;

if (Temp1 == 0) && (Temp2 == 0)
    % 2 zeros vectors
    Distance = 0;
    return;
end
% TODO: what if only one of Temp1/Temp2 equal to zero ?

for i=1:size(histogram1,2)                                   
    TempCount1 = TempCount1 + (histogram1(i) / Temp1) * (histogram2(i) / Temp2);   
    TempCount2 = TempCount2 + (histogram2(i) / Temp2) * (histogram2(i) / Temp2);
    TempCount3 = TempCount3 + (histogram1(i) / Temp1) * (histogram1(i) / Temp1);
end
Distance = (100 - 100 * (TempCount1 / (TempCount2 + TempCount3 - TempCount1)));  

end
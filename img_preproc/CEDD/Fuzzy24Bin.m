function [ histogram ] = Fuzzy24Bin( hue, saturation, value, Fuzzy10binHist, method )
histogram(1:24) = 0;

ResultsTable(1:3) = 0;

SaturationMembershipValues = [0 0 68 188 68 188 255 255];
ValueMembershipValues = [0 0 68 188 68 188 255 255];

SaturationActivation(1:2) = 0;
ValueActivation(1:2) = 0;
Temp = 0;

%find membership values 

%Saturation
TempSaturation = 1;
for i=1:4:8
   if(saturation >= SaturationMembershipValues(i+1) && saturation <= SaturationMembershipValues(i+2))
      SaturationActivation(TempSaturation) = 1;    
   end
   
   if(saturation >= SaturationMembershipValues(i) && saturation < SaturationMembershipValues(i+1))
      SaturationActivation(TempSaturation) = (saturation - SaturationMembershipValues(i)) / (SaturationMembershipValues(i+1) - SaturationMembershipValues(i));    
   end
   
    if(saturation > SaturationMembershipValues(i+2) && saturation < SaturationMembershipValues(i+3))
      SaturationActivation(TempSaturation) = (saturation - SaturationMembershipValues(i+2)) / (SaturationMembershipValues(i+2) - SaturationMembershipValues(i+3)) + 1;    
   end
   
   TempSaturation = TempSaturation + 1;
end

%Value
TempValue = 1;
for i=1:4:8
   if(value >= ValueMembershipValues(i+1) && value <= ValueMembershipValues(i+2))
      ValueActivation(TempValue) = 1;    
   end
   
   if(value >= ValueMembershipValues(i) && value < ValueMembershipValues(i+1))
      ValueActivation(TempValue) = (value - ValueMembershipValues(i)) / (ValueMembershipValues(i+1) - ValueMembershipValues(i));    
   end
   
    if(value > ValueMembershipValues(i+2) && value < ValueMembershipValues(i+3))
      ValueActivation(TempValue) = (value - ValueMembershipValues(i+2)) / (ValueMembershipValues(i+2) - ValueMembershipValues(i+3)) + 1;    
   end
   
   TempValue = TempValue + 1;
end

for i=4:10
   Temp = Temp +  Fuzzy10binHist(i); 
end

Fuzzy24BinRulesDefinition =[
                          1,1,1
                          0,0,2                     
                          0,1,0
                          1,0,2];

if(Temp > 0)
    RuleActivation = -1;
    
    for i=1:length(Fuzzy24BinRulesDefinition)
       if((SaturationActivation(Fuzzy24BinRulesDefinition(i,1)+1)>0) && (ValueActivation(Fuzzy24BinRulesDefinition(i,2)+1)>0))
           
           RuleActivation = Fuzzy24BinRulesDefinition(i,3);
           Minimum = 0;
           Minimum = min(SaturationActivation(Fuzzy24BinRulesDefinition(i,1)+1),ValueActivation(Fuzzy24BinRulesDefinition(i,2)+1));
           
           ResultsTable(RuleActivation+1) = ResultsTable(RuleActivation+1) + Minimum;           
       end
    end 
    
end



for i=1:3
   histogram(i) = histogram(i) +  Fuzzy10binHist(i);
end

for i=3:9
    histogram((i-2)*3+1) = histogram((i-2)*3+1) + Fuzzy10binHist(i+1)*ResultsTable(1);
    histogram((i-2)*3+2) = histogram((i-2)*3+2) + Fuzzy10binHist(i+1)*ResultsTable(2);
    histogram((i-2)*3+3) = histogram((i-2)*3+3) + Fuzzy10binHist(i+1)*ResultsTable(3);
    
end


end

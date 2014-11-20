function [ histogram ] = Fuzzy10Bin( hue, saturation, value, method )
histogram(1:10) = 0;

HueMembershipValues = [0,0,5, 10,5,10,35,50,35,50,70, 85,70,85,150, 165, 150,165,195, 205,195,205,265, 280,265,280,315, 330, 315,330,360,360];
           
SaturationMembershipValues = [0,0,10, 75,10,75,255,255];

ValueMembershipValues = [0,0,10,75,10,75,180,220,180,220,255,255];

HueActivation(1:8) = 0;
SaturationActivation(1:2) = 0;
ValueActivation(1:3) = 0;


%find membership values 

%Hue
TempHue = 1;
for i=1:4:32
   if(hue >= HueMembershipValues(i+1) && hue <= HueMembershipValues(i+2))
      HueActivation(TempHue) = 1;    
   end
   
   if(hue >= HueMembershipValues(i) && hue < HueMembershipValues(i+1))
      HueActivation(TempHue) = (hue - HueMembershipValues(i)) / (HueMembershipValues(i+1) - HueMembershipValues(i));    
   end
   
    if(hue > HueMembershipValues(i+2) && hue < HueMembershipValues(i+3))
      HueActivation(TempHue) = (hue - HueMembershipValues(i+2)) / (HueMembershipValues(i+2) - HueMembershipValues(i+3)) + 1;    
   end
   
   TempHue = TempHue + 1;
end

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
for i=1:4:12
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

%method einai panta 2: multiparticipate defazzificator
Fuzzy10BinRulesDefinition = [
                          0,0,0,2                          
                          0,1,0,2
                          0,0,2,0
                          0,0,1,1
                          1,0,0,2                          
                          1,1,0,2
                          1,0,2,0
                          1,0,1,1
                          2,0,0,2                          
                          2,1,0,2
                          2,0,2,0
                          2,0,1,1
                          3,0,0,2                         
                          3,1,0,2
                          3,0,2,0
                          3,0,1,1
                          4,0,0,2                          
                          4,1,0,2
                          4,0,2,0
                          4,0,1,1
                          5,0,0,2                          
                          5,1,0,2
                          5,0,2,0
                          5,0,1,1
                          6,0,0,2                          
                          6,1,0,2
                          6,0,2,0
                          6,0,1,1
                          7,0,0,2                          
                          7,1,0,2
                          7,0,2,0
                          7,0,1,1
                          0,1,1,3
                          0,1,2,3                     
                          1,1,1,4
                          1,1,2,4
                          2,1,1,5
                          2,1,2,5
                          3,1,1,6
                          3,1,2,6
                          4,1,1,7
                          4,1,2,7
                          5,1,1,8
                          5,1,2,8
                          6,1,1,9
                          6,1,2,9
                          7,1,1,3
                          7,1,2,3]; 


if(method == 2) 
    RuleActivation = -1;
    
    for i=1:48
       if((HueActivation(Fuzzy10BinRulesDefinition(i,1)+1)>0) &&(SaturationActivation(Fuzzy10BinRulesDefinition(i,2)+1)>0) && (ValueActivation(Fuzzy10BinRulesDefinition(i,3)+1)>0))
           
           RuleActivation = Fuzzy10BinRulesDefinition(i,4);
           Minimum = 0;
           Minimum = min(HueActivation(Fuzzy10BinRulesDefinition(i,1)+1),min(SaturationActivation(Fuzzy10BinRulesDefinition(i,2)+1),ValueActivation(Fuzzy10BinRulesDefinition(i,3)+1)));
           
           histogram(RuleActivation+1) = histogram(RuleActivation+1) + Minimum;           
       end
    end    
end

end


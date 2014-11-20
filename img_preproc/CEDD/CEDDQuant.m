function [ FinalCEDD ] = CEDDQuant( CEDDVector )

QuantTable = [180.19686541079636,23730.024499150866,61457.152912541605,113918.55437576842,179122.46400035513,260980.3325940354,341795.93301552488,554729.98648386425];
QuantTable2 = [209.25176965926232, 22490.5872862417345, 60250.8935141849988, 120705.788057580583, 181128.08709063051, 234132.081356900555, 325660.617733105708, 520702.175858657472];
QuantTable3 = [405.4642173212585, 4877.9763319071481, 10882.170090625908, 18167.239081219657, 27043.385568785292, 38129.413201299016, 52675.221316293857, 79555.402607004813];
QuantTable4 = [405.4642173212585, 4877.9763319071481, 10882.170090625908, 18167.239081219657, 27043.385568785292, 38129.413201299016, 52675.221316293857, 79555.402607004813];
QuantTable5 = [968.88475977695578, 10725.159033657819, 24161.205360376698, 41555.917344385321, 62895.628446402261, 93066.271379694881, 136976.13317822068, 262897.86056221306];
QuantTable6 = [968.88475977695578, 10725.159033657819, 24161.205360376698, 41555.917344385321, 62895.628446402261, 93066.271379694881, 136976.13317822068, 262897.86056221306];



FinalCEDD(1:144) = 0;
ElementsDistance(1:8) = 0;

Maximum = 1;

for i=1:14
    FinalCEDD(i) = 0;
    for j=1:8
        ElementsDistance(j) = abs(CEDDVector(i) - QuantTable(j)/1000000);    
    end
    Maximum = 1;
    for j=1:8
        if(ElementsDistance(j) < Maximum)
           Maximum = ElementsDistance(j);
           FinalCEDD(i) = j-1;
        end
    end
end

for i=1:24
    FinalCEDD(i) = 0;
    for j=1:8
        ElementsDistance(j) = abs(CEDDVector(i) - QuantTable(j)/1000000);    
    end
    Maximum = 1;
    for j=1:8
        if(ElementsDistance(j) < Maximum)
           Maximum = ElementsDistance(j);
           FinalCEDD(i) = j-1;
        end
    end
end

for i=25:48
    FinalCEDD(i) = 0;
    for j=1:8
        ElementsDistance(j) = abs(CEDDVector(i) - QuantTable2(j)/1000000);    
    end
    Maximum = 1;
    for j=1:8
        if(ElementsDistance(j) < Maximum)
           Maximum = ElementsDistance(j);
           FinalCEDD(i) = j-1;
        end
    end
end

for i=49:72
    FinalCEDD(i) = 0;
    for j=1:8
        ElementsDistance(j) = abs(CEDDVector(i) - QuantTable3(j)/1000000);    
    end
    Maximum = 1;
    for j=1:8
        if(ElementsDistance(j) < Maximum)
           Maximum = ElementsDistance(j);
           FinalCEDD(i) = j-1;
        end
    end
end

for i=73:96
    FinalCEDD(i) = 0;
    for j=1:8
        ElementsDistance(j) = abs(CEDDVector(i) - QuantTable4(j)/1000000);    
    end
    Maximum = 1;
    for j=1:8
        if(ElementsDistance(j) < Maximum)
           Maximum = ElementsDistance(j);
           FinalCEDD(i) = j-1;
        end
    end
end

for i=97:120
    FinalCEDD(i) = 0;
    for j=1:8
        ElementsDistance(j) = abs(CEDDVector(i) - QuantTable5(j)/1000000);    
    end
    Maximum = 1;
    for j=1:8
        if(ElementsDistance(j) < Maximum)
           Maximum = ElementsDistance(j);
           FinalCEDD(i) = j-1;
        end
    end
end

for i=121:144
    FinalCEDD(i) = 0;
    for j=1:8
        ElementsDistance(j) = abs(CEDDVector(i) - QuantTable6(j)/1000000);    
    end
    Maximum = 1;
    for j=1:8
        if(ElementsDistance(j) < Maximum)
           Maximum = ElementsDistance(j);
           FinalCEDD(i) = j-1;
        end
    end
end

end


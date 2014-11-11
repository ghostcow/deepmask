function [Labels, DecisionValue]= CLSliblinearC(Samples, Model)

[predic_label,acc,DecisionValue] = predict(ones(size(Samples,2),1), sparse(Samples'), Model.svmmodel);
Labels  = predic_label(:,1);
if (size(predic_label,2)>1)
  DecisionValue = predic_label(:,2);
end

if sign(DecisionValue(1))~=sign(Labels(1))
  DecisionValue = -DecisionValue;
end
% if sign(Model.FirstLabel) ~= sign(DecisionValue(1) * Labels(1))
%     DecisionValue(1)
%     Labels(1)
%     Model.FirstLabel
%     error('SVM labels bug!');
% end

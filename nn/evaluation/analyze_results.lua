require 'optim'
require 'gnuplot'

confusion_ = torch.load('../results_small_model2/confusion_test')
confusion = confusion_.mat
numPersons = confusion:size()[1]

--- confusion matrix
print ('success rate per person')
successRates = torch.Tensor(numPersons)
numTruePredictions = 0
for iPerson = 1,numPersons do
    predictions = confusion[{iPerson}]
    numTruePredictions = numTruePredictions + predictions[iPerson]
    successRates[{iPerson}] = 100*(predictions[iPerson] / predictions:sum())
    print(string.format('person %d - %f %%', iPerson, successRates[{iPerson}]))
end
gnuplot.plot(successRates)
print(string.format('\naverage success rate %f %%', 100 * successRates:mean()))
print(string.format('total success rate %f %%', 100 * numTruePredictions/confusion:sum()))

--- top-5 error
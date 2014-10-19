require 'optim'
require 'gnuplot'

confusion = torch.load('../results_small_model2_061014/confusion_test')
numPersons = confusion.mat:size()[1]

--- confusion matrix
print ('success rate per person')
successRates = torch.Tensor(numPersons)
numTruePredictions = 0
for iPerson = 1,numPersons do
    predictions = confusion.mat[{iPerson}]
    numTruePredictions = numTruePredictions + predictions[iPerson]
    successRates[{iPerson}] = 100*(predictions[iPerson] / predictions:sum())
    print(string.format('person %d - %f %%', iPerson, successRates[{iPerson}]))
end
gnuplot.plot(successRates)
print(string.format('\naverage success rate %f %%', successRates:mean()))
print(string.format('total success rate %f %%', 100 * numTruePredictions/confusion.mat:sum()))

--- top-5 error
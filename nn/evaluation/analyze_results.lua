require 'optim'
require 'gnuplot'

confusion = torch.load('../results_small_model2/confusion_train')
confusion = confusion.mat
numPersons = confusion:size()[1]

print ('success rate per person')
successRates = torch.Tensor(numPersons)
for iPerson = 1,numPersons do
    predictions = confusion[{iPerson}]
    successRates[{iPerson}] = 100*(predictions[iPerson] / predictions:sum())
    print(string.format('person %d - %f %%\n', iPerson, successRates[{iPerson}]))
end

gnuplot.plot(successRates)


require 'svm'

LIBSVM_TRAIN_OPTIONS = '-t 0'

function generateSparseTensor(tensor)
    local temp = {}

    for i = 1, tensor:size(1) do
        if (tensor[i] ~= 0) then
            table.insert(temp, {i, tensor[i]})
        end
    end

    local indices = torch.IntTensor(#temp)
    local values = torch.FloatTensor(#temp)

    for i = 1, #temp do
        indices[i] = temp[i][1]
        values[i] = temp[i][2]
    end

    return {indices, values}
end

function tensorToSvmFormat(data_set, labels)
    local svm_data = {}

    for i = 1, data_set:size(1) do
        local new_svm_entry = {}

        new_svm_entry[1] = labels[i]
        new_svm_entry[2] = generateSparseTensor(data_set[i])

        svm_data[i] = new_svm_entry
    end

    return svm_data
end

function chiSquaredDist(id1, id2)
    return torch.cdiv(torch.pow(id1 - id2,2),id1 + id2 + 0.0001)
end

----------------------------------------------------------------------
-- Receives a data_set to train on in following format:
--    data_set  =   tensor of N X 2 X 4096 dimensions
--    labels    =   tensor of N of {1,-1}
----------------------------------------------------------------------
function trainChiSquared(data_set, labels)
    chiSquaredDists = torch.FloatTensor(data_set:size(1), data_set:size(3))
    for i = 1, data_set:size(1) do
        chiSquaredDists[i] = chiSquaredDist(data_set[i][1], data_set[i][2])
    end

    local train_data = tensorToSvmFormat(chiSquaredDists, labels)
    return libsvm.train(train_data, LIBSVM_TRAIN_OPTIONS)
end


----------------------------------------------------------------------
-- Receives a data_set to test on in following format:
--    data_set  =    tensor of N X 2 X 4096 dimensions
--    labels    =   tensor of N of {1,-1}
----------------------------------------------------------------------
function predictChiSquared(classifier, data_set, labels)
    chiSquaredDists = torch.FloatTensor(data_set:size(1), data_set:size(3))
    for i = 1, data_set:size(1) do
        chiSquaredDists[i] = chiSquaredDist(data_set[i][1], data_set[i][2])
    end

    local predict_data = tensorToSvmFormat(chiSquaredDists, labels)
    return libsvm.predict(predict_data, classifier)
end
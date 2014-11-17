function params = setParamsDefaults_impl(params, keyValuePairs)

for iField = 1:length(keyValuePairs)
    fieldName = keyValuePairs{iField}{1};
    if ~isfield(params, fieldName)
        params.(fieldName) = keyValuePairs{iField}{2};
    end
end

end
require 'math'

function isnan(x)
    -- nan is the only number not equal to itself
    if (x ~= x) then
        return 1
    else
        return 0
    end
end

function isinf(x)
    -- nan is the only number not equal to itself
    if (x == math.huge) then
        return 1
    else
        return 0
    end
end

function isValid(tensor)
    local z1 = tensor:clone():abs()
    local z2 = tensor:clone():abs()

    z1:apply(isnan) -- check for nan's
    z2:apply(isinf) -- check for inf
    return ((z1:sum() + z2:sum()) == 0)
end

function MyAssert(condition, message, useAssert)
    if useAssert then
        -- real assert
        assert(condition, message)
    else
        -- fake assert just print warning message
        if not condition then print(message) end
    end
end
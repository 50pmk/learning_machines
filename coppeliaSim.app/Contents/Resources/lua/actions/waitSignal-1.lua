function action_waitSignal_accept(cmd)
    local function checkField(fieldName, fieldType, mathType)
        assert(cmd[fieldName] ~= nil, 'missing ' .. fieldName .. ' field')
        if fieldType then
            assert(type(cmd[fieldName]) == fieldType, fieldName .. ' field must be a ' .. fieldType)
            if mathType then
                assert(math.type(cmd[fieldName]) == mathType, fieldName .. ' field must be a ' .. mathType)
            end
        end
    end

    checkField('signalName', 'string')

    checkField('signalType', 'string')
    require 'tablex'
    local v = {'int', 'float', 'string'}
    assert(table.find(v, cmd.signalType), 'invalid signal type. must be one of: ' .. table.join(v))

    if cmd.signalValue ~= nil then
        if cmd.signalType == 'int' then
            checkField('signalValue', 'number', 'integer')
        elseif cmd.signalType == 'float' then
            checkField('signalValue', 'number')
        elseif cmd.signalType == 'string' then
            checkField('signalValue', 'string')
        end
    end

    return true
end

function action_waitSignal_tick(cmd)
    local value = nil
    if cmd.signalType == 'int' then
        value = sim.getInt32Signal(cmd.signalName)
    elseif cmd.signalType == 'float' then
        value = sim.getFloatSignal(cmd.signalName)
    elseif cmd.signalType == 'string' then
        value = sim.getStringSignal(cmd.signalName)
    end
    if value ~= nil and (cmd.signalValue == value or cmd.signalValue == nil) then
        return {transition = 'succeed'}
    end
end

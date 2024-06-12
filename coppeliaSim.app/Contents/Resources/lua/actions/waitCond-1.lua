function action_waitCond_accept(cmd)
    assert(cmd.condition, 'missing condition field')
    assert(type(cmd.condition) == 'function', 'condition field must be a function')
    return true
end

function action_waitCond_tick(cmd)
    if cmd.condition() then
        return {transition = 'succeed'}
    end
end

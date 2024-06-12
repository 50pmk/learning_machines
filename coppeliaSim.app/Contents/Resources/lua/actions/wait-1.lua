function action_wait_accept(cmd)
    assert(cmd.duration, 'missing duration field')
    assert(type(cmd.duration) == 'number', 'duration field must be a number')
    return true
end

function action_wait_execute(cmd)
    cmd.startTime = sim.getSystemTime()
    return true
end

function action_wait_tick(cmd)
    local elapsed = sim.getSystemTime() - cmd.startTime
    if elapsed >= cmd.duration then
        return {transition = 'succeed'}
    end
end

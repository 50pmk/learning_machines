function action_join_tick(cmd)
    action_join_joined = action_join_joined or {}
    action_join_joined[cmd.id] = action_join_joined[cmd.id] or {}
    action_join_joined[cmd.id][cmd._server.name] = true
    local numJoined = 0
    for k, v in pairs(action_join_joined[cmd.id]) do
        numJoined = numJoined + 1
    end
    if numJoined == cmd.n then
        return {transition = 'succeed'}
    end
end

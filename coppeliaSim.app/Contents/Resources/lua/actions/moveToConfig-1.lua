--[[ USAGE EXAMPLE
function sysCall_init()
    sim = require('sim')
    require'actionServer-1'
    require'actions/moveToConfig-1'

    actionServer = ActionServer()
    
    -- Following is required:
    -------------------------
    local params = {}
    params.joints = {}
    for i = 1, 6, 1 do
        params.joints[i] = sim.getObject('./UR5_joint*', {index = i - 1})
    end
    params.targetPos = {1.4981571262069, 0.23091510688649, 1.1006232214541, 1.8100532979545, 1.4981571702606, -3.1415916760289}
    -------------------------
    
    -- Following is optional:
    -------------------------
    params.maxVel = {math.pi, math.pi, math.pi, math.pi, math.pi, math.pi}
    params.maxAccel = {0.2 * math.pi, 0.2 * math.pi, 0.2 * math.pi, 0.2 * math.pi, 0.2 * math.pi, 0.2 * math.pi}
    params.maxJerk = {0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi}
    params.flags = sim.ruckig_phasesync  -- Ruckig flags

    params.vel = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    params.accel = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    params.targetVel = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    -------------------------

    id = actionServer:send({cmd = 'moveToConfig', params = params})
end

function sysCall_actuation()
    actionServer:tick()
end
--]]

sim = require('sim')

function action_moveToConfig_accept(cmd)
    cmd.params = cmd.params or {}

    if cmd.params.targetPos == nil or type(cmd.params.targetPos) ~= 'table' then
        error("missing or invalid 'params.targetPos' field")
    end

    if cmd.params.joints == nil or type(cmd.params.joints) ~= 'table' or #cmd.params.joints ~= #cmd.params.targetPos then
        error("missing or invalid 'params.joints' field")
    end
    
    cmd.params.flags = cmd.params.flags or -1
    
    cmd.params.maxVel = cmd.params.maxVel or table.rep(0.5 * math.pi, #cmd.params.joints)
    cmd.params.maxAccel = cmd.params.maxAccel or table.rep(0.1 * math.pi, #cmd.params.joints)
    cmd.params.maxJerk = cmd.params.maxJerk or table.rep(0.2 * math.pi, #cmd.params.joints)

    cmd.params.vel = cmd.params.vel or table.rep(0.0, #cmd.params.joints)
    cmd.params.accel = cmd.params.accel or table.rep(0.0, #cmd.params.joints)
    cmd.params.targetVel = cmd.params.targetVel or table.rep(0.0, #cmd.params.joints)
    return true
end

function action_moveToConfig_execute(cmd)
    local pos = {}
    for i = 1, #cmd.params.joints, 1 do
        pos[i] = sim.getJointPosition(cmd.params.joints[i])
    end
    cmd.params.ruckigObject = sim.ruckigPos(#cmd.params.joints, 0.0001, -1, table.add(pos, cmd.params.vel, cmd.params.accel),
                    table.add(cmd.params.maxVel, cmd.params.maxAccel, cmd.params.maxJerk), table.rep(1, #cmd.params.joints),
                    table.add(cmd.params.targetPos, cmd.params.targetVel))
    return true
end

function action_moveToConfig_cancel(cmd)
    action_moveToConfig_cleanup(cmd)
    return true
end

function action_moveToConfig_cleanup(cmd)
    if cmd.params.ruckigObject then
        sim.ruckigRemove(cmd.params.ruckigObject)
        cmd.params.ruckigObject = nil
    end
end

function action_moveToConfig_tick(cmd)
    local result, newPosVelAccel = sim.ruckigStep(cmd.params.ruckigObject, sim.getSimulationTimeStep())
    local retVal = {}
    if result >= 0 then
        retVal.feedback = {}
        retVal.feedback.pos = table.slice(newPosVelAccel, 1, #cmd.params.joints)
        retVal.feedback.vel = table.slice(newPosVelAccel, #cmd.params.joints + 1, 2* #cmd.params.joints)
        retVal.feedback.accel = table.slice(newPosVelAccel, 2 * #cmd.params.joints + 1, 3 * #cmd.params.joints)
        if cmd.callback then
            cmd.callback(retVal.feedback.pos, cmd)
        else
            for i = 1, #cmd.params.joints, 1 do
                if sim.isDynamicallyEnabled(cmd.params.joints[i]) then
                    sim.setJointTargetPosition(cmd.params.joints[i], retVal.feedback.pos[i])
                else    
                    sim.setJointPosition(cmd.params.joints[i], retVal.feedback.pos[i])
                end
            end
        end
        if result == 1 then
            retVal.transition = 'succeed'
            retVal.result = retVal.feedback
            action_moveToConfig_cleanup(cmd)
        end
    else
        error('sim.ruckigStep returned error code ' .. result)
    end
    return retVal
end

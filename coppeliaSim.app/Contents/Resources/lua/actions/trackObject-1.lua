--[[ USAGE EXAMPLE
function sysCall_init()
    sim = require('sim')
    require'actionServer-1'
    require'actions/trackObject-1'

    actionServer = ActionServer()
    
    -- Following is required:
    -------------------------
    local params = {}
    params.ik = {tip = sim.getObject('./UR5_tip'), target = sim.getObject('./UR5_target')}
    -------------------------

    -- Following is optional:
    -------------------------
    params.ik.base = sim.getObject('.')
    params.ik.method = simIK.method_damped_least_squares
    params.ik.damping = 0.02
    params.ik.iterations = 20
    params.ik.constraints = simIK.constraint_pose
    params.ik.joints = {} -- to indicate ik mode joints. Others would be passive if joint field present
    for i = 1, 6, 1 do
        params.ik.joints[i] = sim.getObject('./UR5_joint*', {index = i - 1})
    end
    params.ik.precision = {0.001, 0.5 * math.pi / 180}
    params.ik.allowError = true
    params.ik.breakFlags = 0

    params.maxVel = {math.pi, math.pi, math.pi, math.pi, math.pi, math.pi}
    params.maxAccel = {0.2 * math.pi, 0.2 * math.pi, 0.2 * math.pi, 0.2 * math.pi, 0.2 * math.pi, 0.2 * math.pi}
    params.maxJerk = {0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi}
    params.gradual = true
    params.flags = sim.ruckig_phasesync  -- Ruckig flags
    -------------------------

    id = actionServer:send({cmd = 'trackObject', params = params})

end

function sysCall_actuation()
    actionServer:tick()
end
--]]

sim = require('sim')
simIK = require('simIK')

function action_trackObject_accept(cmd)
    cmd.params = cmd.params or {}
    
    cmd.params.ik = cmd.params.ik or {}
    
    if type(cmd.params.ik) ~= 'table' then
        error("invalid 'params.ik' field")
    end
    if cmd.params.ik.tip == nil or (type(cmd.params.ik.tip) ~= 'number' or not sim.isHandle(cmd.params.ik.tip) ) then
        error("missing or invalid 'params.ik.tip' field")
    end
    if cmd.params.ik.target == nil and (type(cmd.params.ik.target) ~= 'number' or not sim.isHandle(cmd.params.ik.target) ) then
        error("missing or invalid 'params.ik.target' field")
    end
    if cmd.params.ik.base and cmd.params.ik.base ~= -1 and (type(cmd.params.ik.base) ~= 'number' or not sim.isHandle(cmd.params.ik.base) ) then
        error("invalid 'params.ik.base' field")
    end
    if cmd.params.ik.joints and (type(cmd.params.ik.joints) ~= 'table' or #cmd.params.ik.joints == 0 ) then
        error("invalid 'params.ik.joints' field")
    end
    
    cmd.params.ik.base = cmd.params.ik.base or -1
    cmd.params.ik.method = cmd.params.ik.method or simIK.method_damped_least_squares
    cmd.params.ik.damping = cmd.params.ik.damping or 0.02
    cmd.params.ik.iterations = cmd.params.ik.iterations or 20
    cmd.params.ik.constraints = cmd.params.ik.constraints or simIK.constraint_pose
    cmd.params.ik.precision = cmd.params.ik.precision or {0.001, 0.5 * math.pi / 180}
    cmd.params.ik.allowError = cmd.params.ik.allowError
    cmd.params.ik.breakFlags = cmd.params.ik.breakFlags or 0

    if cmd.params.gradual == nil then
        cmd.params.gradual = true
    end
    cmd.params.flags = cmd.params.flags or -1

    if cmd.params.ik.joints == nil then
        cmd.params.ik.joints = {}
        local obj = sim.getObjectParent(cmd.params.ik.tip)
        while obj ~= cmd.params.ik.base do
            if sim.getObjectType(obj) == sim.object_joint_type and sim.getJointType(obj) ~= sim.joint_spherical_subtype then
                table.insert(cmd.params.ik.joints, 1, obj)
            end
            obj = sim.getObjectParent(obj)
        end
    end

    if #cmd.params.ik.joints == 0 then
        error("did not find any non-spherical joints")
    end
    
    cmd.params.maxVel = cmd.params.maxVel or map(function(h) return (0.5 * math.pi) end, cmd.params.ik.joints)
    cmd.params.maxAccel = cmd.params.maxAccel or map(function(h) return (0.1 * math.pi) end, cmd.params.ik.joints)
    cmd.params.maxJerk = cmd.params.maxJerk or map(function(h) return (0.2 * math.pi) end, cmd.params.ik.joints)
    
    cmd.params.vel = cmd.params.vel or table.rep(0.0, #cmd.params.ik.joints)
    cmd.params.accel = cmd.params.accel or table.rep(0.0, #cmd.params.ik.joints)
    cmd.params.targetVel = cmd.params.targetVel or table.rep(0.0, #cmd.params.ik.joints)

    return true
end

function action_trackObject_execute(cmd)
    cmd.params.ik.ikEnv = simIK.createEnvironment()
    cmd.params.ik.ikGroup = simIK.createGroup(cmd.params.ik.ikEnv)
    simIK.setGroupCalculation(cmd.params.ik.ikEnv, cmd.params.ik.ikGroup, cmd.params.ik.method, cmd.params.ik.damping, cmd.params.ik.iterations)
    cmd.params.ik.ikElement, cmd.params.ik.simToIkMap, cmd.params.ik.ikToSimMap = simIK.addElementFromScene(cmd.params.ik.ikEnv, cmd.params.ik.ikGroup, cmd.params.ik.base, cmd.params.ik.tip, cmd.params.ik.target, cmd.params.ik.constraints)
    simIK.setElementPrecision(cmd.params.ik.ikEnv, cmd.params.ik.ikGroup, cmd.params.ik.ikElement, cmd.params.ik.precision)
    for k, v in pairs(cmd.params.ik.simToIkMap) do
        if sim.getObjectType(k) == sim.object_joint_type then
            local found = false
            for i = 1, #cmd.params.ik.joints do
                if cmd.params.ik.joints[i] == k then
                    found = true
                    break
                end
            end
            if not found then
                simIK.setJointMode(cmd.params.ik.ikEnv, v, simIK.jointmode_passive)
            end
        end
    end

    return true
end

function action_trackObject_cancel(cmd)
    action_trackObject_cleanup(cmd)
    return true
end

function action_trackObject_cleanup(cmd)
    simIK.eraseEnvironment(cmd.params.ik.ikEnv)
    cmd.params.ik.ikEnv = nil
    cmd.params.ik.ikGroup = nil
end

function action_trackObject_tick(cmd)
    local function apply(conf, cmd)
        if cmd.callback then
            cmd.callback(conf, cmd)
        else
            for i = 1, #cmd.params.ik.joints do
                if sim.isDynamicallyEnabled(cmd.params.ik.joints[i]) then
                    sim.setJointTargetPosition(cmd.params.ik.joints[i], conf[i])
                else    
                    sim.setJointPosition(cmd.params.ik.joints[i], conf[i])
                end
            end
        end
    end
    
    local retVal = {}
    for i = 1, #cmd.params.ik.joints do
        simIK.setJointPosition(cmd.params.ik.ikEnv, cmd.params.ik.simToIkMap[cmd.params.ik.joints[i]], sim.getJointPosition(cmd.params.ik.joints[i]))
    end
    simIK.setObjectPose(cmd.params.ik.ikEnv, cmd.params.ik.simToIkMap[cmd.params.ik.target], sim.getObjectPose(cmd.params.ik.target))
    local r, f = simIK.handleGroup(cmd.params.ik.ikEnv, cmd.params.ik.ikGroup, {syncWorlds = false, allowError = cmd.params.ik.allowError})
    if f & cmd.params.ik.breakFlags ~= 0 then
        error('simIK.handleGroup returned flags ' .. f)
    end
    local newJointVals = {}
    for i = 1, #cmd.params.ik.joints do
        newJointVals[i] = simIK.getJointPosition(cmd.params.ik.ikEnv, cmd.params.ik.simToIkMap[cmd.params.ik.joints[i]])
    end
    if cmd.params.gradual then
        local pos = {}
        for i = 1, #cmd.params.ik.joints, 1 do
            pos[i] = sim.getJointPosition(cmd.params.ik.joints[i])
        end
        local ruckigObject = sim.ruckigPos(#cmd.params.ik.joints, 0.0001, -1, table.add(pos, cmd.params.vel, cmd.params.accel),
                        table.add(cmd.params.maxVel, cmd.params.maxAccel, cmd.params.maxJerk), table.rep(1, #cmd.params.ik.joints),
                        table.add(newJointVals, cmd.params.targetVel))
    
        local result, newPosVelAccel = sim.ruckigStep(ruckigObject, sim.getSimulationTimeStep())
        sim.ruckigRemove(ruckigObject)
        if result >= 0 then
            retVal.feedback = {}
            retVal.feedback.pos = table.slice(newPosVelAccel, 1, #cmd.params.ik.joints)
            cmd.params.vel = table.slice(newPosVelAccel, #cmd.params.ik.joints + 1, 2* #cmd.params.ik.joints)
            cmd.params.accel = table.slice(newPosVelAccel, 2 * #cmd.params.ik.joints + 1, 3 * #cmd.params.ik.joints)
            retVal.feedback.vel = cmd.params.vel
            retVal.feedback.accel = cmd.params.accel
            apply(retVal.feedback.pos, cmd)
        else
            error('sim.ruckigStep returned error code ' .. result)
        end
    else
        apply(newJointVals, cmd)
    end

    return retVal
end

--[[ USAGE EXAMPLE
function sysCall_init()
    sim = require('sim')
    require'actionServer-1'
    require'actions/moveToPose-1'

    actionServer = ActionServer()
    
    -- Following is required:
    -------------------------
    local params = {}
    -- One of the following 2 is optional (i.e. just object movement or full arm movement via IK)
    --params.object = sim.getObject('./UR5_tip')
    params.ik = {tip = sim.getObject('./UR5_tip'), target = sim.getObject('./UR5_target')}
    params.targetPose = {0.0, 0.2, 0.50096988332966, 1.9832943768143e-05, -0.70724195339448, -7.2268179229349e-06, 0.70697158281871}
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

    params.maxVel={0.345, 0.345, 0.345, 4.5}
    params.maxAccel={0.13, 0.13, 0.13, 1.24}
    params.maxJerk={0.1, 0.1, 0.1, 0.2}
    params.flags = sim.ruckig_phasesync  -- Ruckig flags
    
    params.vel = table.rep(0.0, 4)
    params.accel = table.rep(0.0, 4)
    params.targetVel = table.rep(0.0, 4)
    -------------------------

    id = actionServer:send({cmd = 'moveToPose', params = params})
end

function sysCall_actuation()
    actionServer:tick()
end
--]]

sim = require('sim')
simIK = require('simIK')

function action_moveToPose_accept(cmd)
    cmd.params = cmd.params or {}
    
    if cmd.params.object then
        if type(cmd.params.object) ~= 'number' or not sim.isHandle(cmd.params.object) then
            error("invalid 'params.object' field")
        end
    else
        if cmd.params.ik == nil then
            error("either 'params.object' or 'params.ik' field is required, none found")
        else
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
        end
    end
    
    if cmd.params.targetPose == nil or type(cmd.params.targetPose) ~= 'table' or #cmd.params.targetPose ~= 7 then
        error("missing or invalid 'params.targetPose' field")
    end

    if cmd.params.object == nil then
        cmd.params.ik.base = cmd.params.ik.base or -1
        cmd.params.ik.method = cmd.params.ik.method or simIK.method_damped_least_squares
        cmd.params.ik.damping = cmd.params.ik.damping or 0.02
        cmd.params.ik.iterations = cmd.params.ik.iterations or 20
        cmd.params.ik.constraints = cmd.params.ik.constraints or simIK.constraint_pose
        cmd.params.ik.precision = cmd.params.ik.precision or {0.001, 0.5 * math.pi / 180}
        cmd.params.ik.allowError = cmd.params.ik.allowError
    end
    cmd.params.maxVel = cmd.params.maxVel or {0.2, 0.2, 0.2, 1.0 * math.pi}
    cmd.params.maxAccel = cmd.params.maxAccel or {0.1, 0.1, 0.1, 0.5 * math.pi}
    cmd.params.maxJerk = cmd.params.maxJerk or {0.1, 0.1, 0.1, 0.5 * math.pi}
    cmd.params.flags = cmd.params.flags or -1

    cmd.params.vel = cmd.params.vel or table.rep(0.0, 4)
    cmd.params.accel = cmd.params.accel or table.rep(0.0, 4)
    cmd.params.targetVel = cmd.params.targetVel or table.rep(0.0, 4)
    return true
end

function action_moveToPose_execute(cmd)
    if cmd.params.object then
        cmd.params.startMatrix = sim.getObjectMatrix(cmd.params.object)
    else
        -- We use IK to move an arm
        cmd.params.ik.ikEnv = simIK.createEnvironment()
        cmd.params.ik.ikGroup = simIK.createGroup(cmd.params.ik.ikEnv)
        simIK.setGroupCalculation(cmd.params.ik.ikEnv, cmd.params.ik.ikGroup, cmd.params.ik.method, cmd.params.ik.damping, cmd.params.ik.iterations)
        local ikElement, simToIkMap, ikToSimMap = simIK.addElementFromScene(cmd.params.ik.ikEnv, cmd.params.ik.ikGroup, cmd.params.ik.base, cmd.params.ik.tip, cmd.params.ik.target, cmd.params.ik.constraints)
        simIK.setElementPrecision(cmd.params.ik.ikEnv, cmd.params.ik.ikGroup, ikElement, cmd.params.ik.precision)
        if cmd.params.ik.joints and #cmd.params.ik.joints > 0 then
            for k, v in pairs(simToIkMap) do
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
        end
        cmd.params.startMatrix = sim.getObjectMatrix(cmd.params.ik.tip)
    end

    cmd.params.targetMatrix = sim.buildMatrixQ(table.slice(cmd.params.targetPose, 1, 3), {cmd.params.targetPose[4], cmd.params.targetPose[5], cmd.params.targetPose[6], cmd.params.targetPose[7]})
    local axis
    axis, cmd.params.angle = sim.getRotationAxis(cmd.params.startMatrix, cmd.params.targetMatrix)
    local dx = {cmd.params.targetMatrix[4] - cmd.params.startMatrix[4], cmd.params.targetMatrix[8] - cmd.params.startMatrix[8], cmd.params.targetMatrix[12] - cmd.params.startMatrix[12], cmd.params.angle }
    cmd.params.ruckigObject = sim.ruckigPos(4, 0.0001, cmd.params.flags, table.add({0, 0, 0, 0}, cmd.params.vel, cmd.params.accel), table.add(cmd.params.maxVel, cmd.params.maxAccel, cmd.params.maxJerk), {1, 1, 1, 1}, table.add(dx, cmd.params.targetVel))

    return true
end

function action_moveToPose_cancel(cmd)
    action_moveToPose_cleanup(cmd)
    return true
end

function action_moveToPose_cleanup(cmd)
    if cmd.params.ruckigObject then
        sim.ruckigRemove(cmd.params.ruckigObject)
        cmd.params.ruckigObject = nil
    end
    if cmd.params.object == nil then
        simIK.eraseEnvironment(cmd.params.ik.ikEnv)
        cmd.params.ik.ikEnv = nil
        cmd.params.ik.ikGroup = nil
    end
    
    cmd.params.startMatrix = nil
    cmd.params.targetMatrix = nil
    cmd.params.angle = nil
end

function action_moveToPose_tick(cmd)
    local retVal = {}
    local result, newPosVelAccel = sim.ruckigStep(cmd.params.ruckigObject, sim.getSimulationTimeStep())
    if result >= 0 then
        local t = 0
        if math.abs(cmd.params.angle) > math.pi * 0.00001 then
            t = newPosVelAccel[4] / cmd.params.angle
        end
        local outMatrix = sim.interpolateMatrices(cmd.params.startMatrix, cmd.params.targetMatrix, t)
        outMatrix[4] = cmd.params.startMatrix[4] + newPosVelAccel[1]
        outMatrix[8] = cmd.params.startMatrix[8] + newPosVelAccel[2]
        outMatrix[12] = cmd.params.startMatrix[12] + newPosVelAccel[3]

        retVal.feedback = {}
        retVal.feedback.pose = sim.matrixToPose(outMatrix)
        retVal.feedback.vel = table.slice(newPosVelAccel, 5, 8)
        retVal.feedback.accel = table.slice(newPosVelAccel, 9, 12)
        
        if cmd.callback then
            cmd.callback(retVal.feedback.pose, cmd)
        else
            if cmd.params.object then
                sim.setObjectPose(cmd.params.object, retVal.feedback.pose)
            else
                sim.setObjectPose(cmd.params.ik.target, retVal.feedback.pose)
                simIK.handleGroup(cmd.params.ik.ikEnv, cmd.params.ik.ikGroup, {syncWorlds = true, allowError = cmd.params.ik.allowError})
            end
        end
        
        if result == 1 then
            retVal.transition = 'succeed'
            retVal.result = retVal.feedback
            action_moveToPose_cleanup(cmd)
        end
    else
        error('sim.ruckigStep returned error code ' .. result)
    end

    return retVal
end

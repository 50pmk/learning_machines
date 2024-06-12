ActionServer = {}

function ActionServer:send(cmd)
    if type(cmd) == 'string' then
        cmd = {cmd = cmd}
    end

    self.lastCmdId = self.lastCmdId + 1
    local queuedCmd = {
        _id = self.lastCmdId,
        _server = self,
    }
    for k, v in pairs(cmd) do
        queuedCmd[k] = v
    end

    local info = self:_callActionMethod(queuedCmd, 'info', {})
    local api = info.api or 1

    assert(api == self.api, 'mismatching actionServer api (requires api = ' .. api .. ')')

    local accepted = self:_callActionMethod(queuedCmd, 'accept', true)
    if accepted then
        table.insert(self.queue, queuedCmd)
        self:_setStatus(queuedCmd._id, 'accepted')
        return queuedCmd._id
    else
        self:_setStatus(queuedCmd._id, 'rejected')
    end
end

function ActionServer:cancel(cmdId)
    require 'tablex'
    local index = table.find(self.queue, cmdId, function(cmd)
        return cmd._id == cmdId
    end)
    if index then
        table.remove(self.queue, index)
    elseif self.currentCmd._id == cmdId then
        local cancelOk = self:_callActionMethod(self.currentCmd, 'cancel', true)
        if not cancelOk then return end
        self.currentCmd = nil
    end
    self:_setStatus(cmdId, 'canceled')
end

function ActionServer:getStatus(cmdId)
    return self.status[cmdId]
end

function ActionServer:getFeedback(cmdId)
    return self.feedback[cmdId]
end

function ActionServer:getResult(cmdId)
    return self.result[cmdId]
end

function ActionServer:tick()
    -- if not doing anything, pick next command:
    while self.currentCmd == nil do
        if #self.queue == 0 then return end
        self.currentCmd = table.remove(self.queue, 1)
        local executeOk = self:_callActionMethod(self.currentCmd, 'execute', true)
        if executeOk then
            self:_setStatus(self.currentCmd._id, 'executing')
        else
            self:_setStatus(self.currentCmd._id, 'aborted')
            self.currentCmd = nil
        end
    end

    -- continue running current action:
    local r = self:_callActionMethod(self.currentCmd, 'tick')
    if r == nil then
        -- r == nil --> continue execution in next tick
        return
    end
    if type(r) ~= 'table' then
        r = {transition = 'succeed', result = r}
    end
    if r.transition == 'abort' then
        self:_setStatus(self.currentCmd._id, 'aborted')
        self.currentCmd = nil
    elseif r.transition == 'succeed' then
        self.result[self.currentCmd._id] = r.result
        self:_setStatus(self.currentCmd._id, 'succeeded')
        self.currentCmd = nil
    elseif r.feedback ~= nil then
        self.feedback[self.currentCmd._id] = r.feedback
    end
end

function ActionServer:_isModuleLoaded(n)
    for k, v in pairs(package.loaded) do
        if string.find('/' .. k, '[./]' .. n .. '%-?%d?$') then
            return true
        end
    end
end

function ActionServer:_callActionMethod(cmd, method, defaultRetIfNoSuchMethod)
    local funcName = 'action_' .. cmd.cmd .. '_' .. method
    if _G[funcName] == nil then
        if defaultRetIfNoSuchMethod == nil then
            if self:_isModuleLoaded(cmd.cmd) then
                error('function "' .. funcName .. '" not found')
            else
                error('module "' .. cmd.cmd .. '" not loaded')
            end
        else
            return defaultRetIfNoSuchMethod
        end
    end
    return _G[funcName](cmd)
end

function ActionServer:addListener(listenerType, listener, cmdId)
    cmdId = cmdId or 'global'
    self.listeners[listenerType] = self.listeners[listenerType] or {}
    self.listeners[listenerType][cmdId] = self.listeners[listenerType][cmdId] or {}
    table.insert(self.listeners[listenerType][cmdId], listener)
end

function ActionServer:_notifyListener(listenerType, cmdId)
    self.listeners[listenerType] = self.listeners[listenerType] or {}
    self.listeners[listenerType]['global'] = self.listeners[listenerType]['global'] or {}
    self.listeners[listenerType][cmdId] = self.listeners[listenerType][cmdId] or {}
    for _, cmdId1 in ipairs{cmdId, 'global'} do
        for i, listener in ipairs(self.listeners[listenerType][cmdId1]) do
            listener(listenerType, cmdId)
        end
    end
end

function ActionServer:_setStatus(cmdId, status)
    if self.status[cmdId] ~= status then
        self.status[cmdId] = status
        self:_notifyListener('statusChanged', cmdId)
    end
end

function ActionServer:__index(k)
    return ActionServer[k]
end

setmetatable(ActionServer, {__call = function(meta, name)
    name = name or 'default'
    ActionServer._byName = ActionServer._byName or {}
    assert(ActionServer._byName[name] == nil, 'action server name must be unique')
    local self = setmetatable({
        api = 1,
        name = name,
        lastCmdId = 0,
        queue = {},
        currentCmd = nil,
        toCancel = {},
        status = {},
        result = {},
        feedback = {},
        listeners = {},
    }, meta)
    ActionServer._byName[name] = self
    return self
end})

function ActionServer:byName(name)
    assert(ActionServer._byName[name] ~= nil, 'invalid action server name')
    return ActionServer._byName[name]
end

function ActionServer:defineFunctionalInterface()
    local publicFunctions = {
        'send',
        'cancel',
        'getStatus',
        'getFeedback',
        'getResult',
        'tick',
    }
    for _, funcName in ipairs(publicFunctions) do
        _G['actionServer_' .. funcName] = function(name, ...)
            local as = ActionServer:byName(name)
            return as[funcName](as, ...)
        end
    end
end

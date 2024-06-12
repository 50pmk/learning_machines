local codeEditorInfos=[[
string handle=simSkel.createObject()
simSkel.destroyObject(string handle)
simSkel.setData(string handle,int a,int b)
int currentSize=simSkel.compute(string handle)
int[] output=simSkel.getOutput(string handle)
]]

registerCodeEditorInfos("simSkel",codeEditorInfos)

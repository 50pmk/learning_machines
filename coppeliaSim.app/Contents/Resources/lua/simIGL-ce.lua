local codeEditorInfos=[[
map result=simIGL.meshBoolean(map a,map b,int op)
grid b,grid fi=simIGL.randomPointsOnMesh(int n,map m,bool convertToWorldCoords=false)
map m=simIGL.sweptVolume(map m,string transformFunc,int timeSteps,int gridSize,float isoLevel=0)
float[] distances=simIGL.exactGeodesic(map m,int[] vs,int[] fs,int[] vt,int[] ft)
grid e,grid ue,grid emap,grid uec,grid uee=simIGL.uniqueEdgeMap(grid f)
int[] r,int[] s=simIGL.closestFacet(map m,grid points,grid emap,grid uec,grid uee,int[] indices={})
map m=simIGL.upsample(map m,int n=1)
float[3] c,float vol=simIGL.centroid(map m)
grid bc=simIGL.barycenter(grid v,grid f)
int result,grid tv,grid tt,grid tf=simIGL.tetrahedralize(map m,string switches="")
map m=simIGL.convexHull(float[] points)
map m,int[] j=simIGL.intersectWithHalfSpace(map m,float[3] pt,float[3] n)
float[] vol=simIGL.volume(map m)
map mesh=simIGL.getMesh(int h,map options={})
int handleResult=simIGL.meshBooleanShape(int[] handles,int op)
int handleResult=simIGL.convexHullShape(int[] handles)
simIGL.boolean_op.union
simIGL.boolean_op.intersection
simIGL.boolean_op.difference
simIGL.boolean_op.symmetric_difference
simIGL.boolean_op.resolve
]]

registerCodeEditorInfos("simIGL",codeEditorInfos)

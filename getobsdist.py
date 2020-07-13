import pandas as pd 
import numpy as np
df=pd.read_csv("/home/armaan/Downloads/Data_Set/Rover_Displacement.csv")
obs=[0,2000,3700,5300,7000,7500,11700,13700,13710]
heights=[[0,0],[150,150],[0,150],[150,0],[0,150],[150,0],[200,200],[150,150],[150,150]]
dist_frames=[]
last_index=[0]
heights_df=np.zeros((4494,2))
heights_df=pd.DataFrame({'Height Left':heights_df[:,0],'Height Right':heights_df[:,1]})
for i in obs:
	dist_frames.append(pd.DataFrame(i-df["Linear Displacement1 (mm)"]))
#print(dist_frames[0])

res_df=df.copy()
print(res_df)
for i in dist_frames:
	for index,row in i.iterrows():
		if row["Linear Displacement1 (mm)"]<0:
			print(row["Linear Displacement1 (mm)"],index)
			last_index.append(index)
			break
#res_df['Linear Displacement1 (mm)'].iloc[0:280]=dist_frames[0]['Linear Displacement1 (mm)'].iloc[0:280]*0
#adding all object distances to res_df
for i in range(len(last_index)-1):
	if i==0:
		res_df['Linear Displacement1 (mm)'].iloc[last_index[i]:last_index[i+1]]=(dist_frames[i]['Linear Displacement1 (mm)'].iloc[last_index[i]:last_index[i+1]]*0)
		#print(dist_frames[i]['Linear Displacement1 (mm)'].iloc[i:i+1])
	else:
		res_df['Linear Displacement1 (mm)'].iloc[last_index[i]:last_index[i+1]]=dist_frames[i]['Linear Displacement1 (mm)'].iloc[last_index[i]:last_index[i+1]]
res_df['Linear Displacement1 (mm)'].iloc[last_index[len(last_index)-1]:]=0
for i in range(len(last_index)-1):

	for index,row in res_df.iloc[last_index[i]:last_index[i+1]].iterrows():
			if row["Linear Displacement1 (mm)"]<=100:
				heights_df["Height Left"].iloc[index]=heights[i][0]
				heights_df["Height Right"].iloc[index]=heights[i][1]
				print(row["Linear Displacement1 (mm)"],index)
				#last_index.append(index)
heights_df.iloc[last_index[len(last_index)-1]:]=0
res_df=res_df.join(heights_df)
print(res_df.iloc[2944])
res_df.to_csv("/home/armaan/Downloads/Data_Set/Object_info.csv",index=False)

"""
2.0_simplify_obj.py

이 스크립트는 3D 모델 데이터셋의 각 폴더에 있는 model.obj 파일을 정규화(normalize)하여 model_normalized.obj로 저장합니다.

[동작 설명]
- 각 서브폴더(예: cylinder_0, cylinder_1, ...)에 있는 model.obj 파일을 읽어 중심을 원점으로 이동시키고, 대각선 길이가 1이 되도록 스케일을 맞춥니다.
- 정규화된 결과를 model_normalized.obj로 저장합니다.
- 데이터가 많을 경우, 여러 프로세스로 나누어 병렬 처리할 수 있습니다.

[입력]
- target_dir: 3D 모델 obj 파일들이 들어있는 폴더 (예: /workspace/data/cylinder_data/shapes)
- 각 서브폴더에 model.obj 파일이 있어야 함

[출력]
- 각 서브폴더에 model_normalized.obj 파일이 생성됨

[실행 인자]
- --share_id (int, 기본값 0): 전체 작업을 나눌 때 이 프로세스가 담당할 분할 번호 (0부터 시작)
- --share_total (int, 기본값 1): 전체 분할 개수 (병렬 처리 시 사용)
- --target_dir (str, 기본값 '/workspace/data/cylinder_data/shapes'): obj 파일들이 들어있는 폴더 경로

[예시 실행]
python 2.0_simplify_obj.py --share_id 0 --share_total 1 --target_dir /workspace/data/cylinder_data/shapes
# 또는 기본값 사용 시
python 2.0_simplify_obj.py

[병렬 처리 예시]
# 4개로 나누어 4개의 프로세스에서 동시에 실행
python 2.0_simplify_obj.py --share_id 0 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.0_simplify_obj.py --share_id 1 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.0_simplify_obj.py --share_id 2 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.0_simplify_obj.py --share_id 3 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
"""
import os
import numpy as np
import argparse

## target dir ./data/data_name/
## dir required ./data/data_name/shapes
#normalize obj and create new dirs
#python bin_voxelization/0_simplify_obj.py 0 1 /local-scratch2/fenggeny/SECAD-Net-main/data/mingrui_data/shapes
parser = argparse.ArgumentParser()
parser.add_argument("--share_id", type=int, default=0, help="id of the share [0]")
parser.add_argument("--share_total", type=int, default=1, help="total num of shares [1]")
parser.add_argument("--target_dir", type=str, default="/workspace/data/cylinder_data/shapes", help="target directory [default: /workspace/data/cylinder_data/shapes]")
FLAGS = parser.parse_args()

target_dir = FLAGS.target_dir
if not os.path.exists(target_dir):
	print("ERROR: this dir does not exist: "+target_dir)
	exit()

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

share_id = FLAGS.share_id
share_total = FLAGS.share_total

start = int(share_id*len(obj_names)/share_total)
end = int((share_id+1)*len(obj_names)/share_total)
obj_names = obj_names[start:end]


def load_obj(dire):
	fin = open(dire,'r')
	lines = fin.readlines()
	fin.close()
	
	vertices = []
	triangles = []
	
	for i in range(len(lines)):
		line = lines[i].split()
		if len(line)==0:
			continue
		if line[0] == 'v':
			x = float(line[1])
			y = float(line[2])
			z = float(line[3])
			vertices.append([x,y,z])
		if line[0] == 'f':
			x = int(line[1].split("/")[0])
			y = int(line[2].split("/")[0])
			z = int(line[3].split("/")[0])
			triangles.append([x-1,y-1,z-1])
	
	vertices = np.array(vertices, np.float32)
	triangles = np.array(triangles, np.int32)
	
	#normalize diagonal=1
	x_max = np.max(vertices[:,0])
	y_max = np.max(vertices[:,1])
	z_max = np.max(vertices[:,2])
	x_min = np.min(vertices[:,0])
	y_min = np.min(vertices[:,1])
	z_min = np.min(vertices[:,2])
	x_mid = (x_max+x_min)/2
	y_mid = (y_max+y_min)/2
	z_mid = (z_max+z_min)/2
	x_scale = x_max - x_min
	y_scale = y_max - y_min
	z_scale = z_max - z_min
	scale = np.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
	
	vertices[:,0] = (vertices[:,0]-x_mid)/scale
	vertices[:,1] = (vertices[:,1]-y_mid)/scale
	vertices[:,2] = (vertices[:,2]-z_mid)/scale
	
	return vertices, triangles

def write_obj(dire, vertices, triangles):
	fout = open(dire, 'w')
	for ii in range(len(vertices)):
		fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("f "+str(triangles[ii,0]+1)+" "+str(triangles[ii,1]+1)+" "+str(triangles[ii,2]+1)+"\n")
	fout.close()


for i in range(len(obj_names)):
	this_subdir_name = os.path.join(target_dir, obj_names[i])
	if not os.path.isdir(this_subdir_name):
		os.makedirs(this_subdir_name)
	# sub_names = os.listdir(this_subdir_name)
	# if len(sub_names)==0:
	#     command = "rm -r "+this_subdir_name
	#     os.system(command)
	#     continue

	this_name = os.path.join(this_subdir_name, "model.obj")
	out_name = os.path.join(this_subdir_name, "model_normalized.obj")

	print(i,this_name)
	v,t = load_obj(this_name)
	#os.remove(this_name)
	write_obj(out_name, v,t)
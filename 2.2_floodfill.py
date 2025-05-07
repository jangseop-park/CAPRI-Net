"""
2.2_floodfill.py

이 스크립트는 2.1_voxelize.py로 생성된 model_normalized.binvox 파일에 대해 flood fill 알고리즘을 적용하여 내부를 채운 model_filled.binvox 파일을 생성합니다.

[동작 설명]
- 각 서브폴더(예: cylinder_0, cylinder_1, ...)에 있는 model_normalized.binvox 파일을 읽어 flood fill을 적용합니다.
- 내부가 채워진 결과를 model_filled.binvox로 저장합니다.
- 데이터가 많을 경우, 여러 프로세스로 나누어 병렬 처리할 수 있습니다.

[입력]
- target_dir: binvox 파일들이 들어있는 폴더 (예: /workspace/data/cylinder_data/shapes)
- 각 서브폴더에 model_normalized.binvox 파일이 있어야 함

[출력]
- 각 서브폴더에 model_filled.binvox 파일이 생성됨 (flood fill 결과)

[실행 인자]
- --share_id (int, 기본값 0): 전체 작업을 나눌 때 이 프로세스가 담당할 분할 번호 (0부터 시작)
- --share_total (int, 기본값 1): 전체 분할 개수 (병렬 처리 시 사용)
- --target_dir (str, 기본값 '/workspace/data/cylinder_data/shapes'): binvox 파일들이 들어있는 폴더 경로

[예시 실행]
python 2.2_floodfill.py --share_id 0 --share_total 1 --target_dir /workspace/data/cylinder_data/shapes
# 또는 기본값 사용 시
python 2.2_floodfill.py

[병렬 처리 예시]
# 4개로 나누어 4개의 프로세스에서 동시에 실행
python 2.2_floodfill.py --share_id 0 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.2_floodfill.py --share_id 1 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.2_floodfill.py --share_id 2 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.2_floodfill.py --share_id 3 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
"""
import sys
import os
sys.path.append('/workspace/bin_voxelization')
import numpy as np
import cv2
import h5py
import binvox_rw_customized
import mcubes
import cutils
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--share_id", type=int, default=0, help="id of the share [0]")
parser.add_argument("--share_total", type=int, default=1, help="total num of shares [1]")
parser.add_argument("--target_dir", type=str, default="/workspace/data/cylinder_data/shapes", help="target directory [default: /workspace/data/cylinder_data/shapes]")
FLAGS = parser.parse_args()

target_dir = FLAGS.target_dir
if not target_dir.endswith('/'):
    target_dir += '/'
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

def write_ply_triangle(name, vertices, triangles):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()


queue = np.zeros([1024*1024*32,3], np.int32)
state_ctr = np.zeros([1024*1024*32,2], np.int32)

start_time = time.time()
for i in range(len(obj_names)):
    this_name = target_dir + obj_names[i] + "/model_normalized.binvox"
    # this_name = target_dir + obj_names[i] + "/model.binvox"
    out_name = target_dir + obj_names[i] + "/model_filled.binvox"
    print(i,this_name)
    if os.path.exists(out_name):
        continue
    voxel_model_file = open(this_name, 'rb')
    vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

    batch_voxels = vox_model.data.astype(np.uint8) + 1
    cutils.floodfill(batch_voxels,queue)
    cutils.get_state_ctr(batch_voxels,state_ctr)

    with open(out_name, 'wb') as fout:
        binvox_rw_customized.write(vox_model, fout, state_ctr)
    print('left time: ', ( (time.time() - start_time) * (len(obj_names) - i)/(i+1) ))
    
    voxel_model_file = open(out_name, 'rb')
    vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file)
    batch_voxels = vox_model.data.astype(np.uint8)
    vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
    write_ply_triangle("vox.ply", vertices, triangles)

print("[완료] 모든 shape에 대해 flood fill이 정상적으로 처리되었습니다.")


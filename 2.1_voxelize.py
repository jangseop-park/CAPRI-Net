"""
2.1_voxelize.py

이 스크립트는 2.0_simplify_obj.py로 정규화된 model_normalized.obj 파일을 binvox를 이용해 voxel 데이터로 변환합니다.

[동작 설명]
- 각 서브폴더(예: cylinder_0, cylinder_1, ...)에 있는 model_normalized.obj 파일을 읽어 binvox 명령어로 voxelization을 수행합니다.
- 데이터가 많을 경우, 여러 프로세스로 나누어 병렬 처리할 수 있습니다.

[입력]
- target_dir: 정규화된 obj 파일들이 들어있는 폴더 (예: /workspace/data/cylinder_data/shapes)
- 각 서브폴더에 model_normalized.obj 파일이 있어야 함

[출력]
- 각 서브폴더에 model_normalized.obj.binvox 파일이 생성됨 (binvox 결과)

[실행 인자]
- --share_id (int, 기본값 0): 전체 작업을 나눌 때 이 프로세스가 담당할 분할 번호 (0부터 시작)
- --share_total (int, 기본값 1): 전체 분할 개수 (병렬 처리 시 사용)
- --target_dir (str, 기본값 '/workspace/data/cylinder_data/shapes'): obj 파일들이 들어있는 폴더 경로

[예시 실행]
python 2.1_voxelize.py --share_id 0 --share_total 1 --target_dir /workspace/data/cylinder_data/shapes
# 또는 기본값 사용 시
python 2.1_voxelize.py

[병렬 처리 예시]
# 4개로 나누어 4개의 프로세스에서 동시에 실행
python 2.1_voxelize.py --share_id 0 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.1_voxelize.py --share_id 1 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.1_voxelize.py --share_id 2 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
python 2.1_voxelize.py --share_id 3 --share_total 4 --target_dir /workspace/data/cylinder_data/shapes
"""
import numpy as np
import cv2
import os
import h5py
import mcubes
import argparse

#require ./data/data_name/shapes
#python bin_voxelization/1_voxelize.py 0 1 /local-scratch2/fenggeny/SECAD-Net-main/data/mingrui_data/shapes

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

BINVOX_PATH = "/workspace/bin_voxelization/binvox"

for i in range(len(obj_names)):
    this_name = os.path.join(target_dir,obj_names[i]+"/model_normalized.obj")
    print(i,this_name)

    command = f"{BINVOX_PATH} -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -d 1024 -e {this_name}"
    os.system(command)


"""
2.5_voxel2pc.py

이 스크립트는 2.4_imnet_sampling.py로 생성된 ae_voxel_points_samples.hdf5 또는 voxel_64.npz 파일에서 포인트 클라우드 데이터를 추출하여 저장합니다.

[동작 설명]
- data_flag 값에 따라 동작이 달라집니다.
    0: 각 shape 폴더의 voxel_64.npz 파일에서 포인트 클라우드를 추출하여
       → dataset_dir/shapes/각_샘플명/voxel2pc.npz로 저장
    1: ae_voxel_points_samples.hdf5에서 배치로 포인트 클라우드를 추출하여
       → dataset_dir/voxel2pc.hdf5로 저장

[입력]
- dataset_dir: 데이터셋 폴더 경로 (예: /workspace/data/cylinder_data)
- data_flag: 동작 모드 (0: 단일 npz, 1: 배치 hdf5, 기본값 1)

[출력 및 저장 구조]
- data_flag=0: 각 shape 폴더에 voxel2pc.npz (예: dataset_dir/shapes/cylinder_0/voxel2pc.npz)
    ├─ points: (8192, 4)  # [x, y, z, value] (int8)
    └─ names: 샘플명 (str)

- data_flag=1: dataset_dir/voxel2pc.hdf5
    ├─ points: (N, 8192, 4)  # [x, y, z, value] (int8)
    └─ voxels: (N, 64, 64, 64, 1)  # 원본 복셀 (int8)
    ※ N은 샘플 개수, 순서는 names.npz와 동일

	
	
[points/voxels 데이터셋 값의 의미]
- points: 각 샘플(모델)마다 8192개의 포인트가 저장되며, 각 포인트는 [x, y, z, value] 형태입니다.
    - x, y, z: 64³ 복셀 그리드 상의 좌표 (0~63)
    - value: 해당 좌표의 복셀 값 (0 또는 1, int8)
        - 1: 내부(occupied, 채워진 복셀)
        - 0: 외부(empty, 비어있는 복셀)
- voxels: 각 샘플의 전체 64³ 복셀 그리드가 저장되며, 각 복셀 값은 0(비어있음) 또는 1(채워짐)입니다.

[64³ 전체 복셀과 8192 샘플의 차이]
- 64³ = 262,144: 64×64×64 복셀 그리드의 전체 복셀 개수입니다. 모든 복셀을 다 포인트로 쓰면 너무 많고, 대부분은 비어있거나 중복 정보가 많습니다.
- 8192: 전체 복셀 중에서 "대표적인 위치"만 골라서 8192개만 샘플링합니다. 이렇게 하면 데이터 용량도 줄고, 학습/후처리에도 효율적입니다.

[16×16×16×2, 16×16×16×8의 의미]
- 16×16×16 = 4096: 64³ 그리드를 16³의 블록(작은 큐브)으로 나누는 개념입니다. 즉, 64/16=4, 각 축을 4등분해서 총 4096개의 작은 영역(블록)으로 나눕니다.
- ×2, ×8: 각 블록에서 여러 개의 포인트를 샘플링하기 위해 곱해주는 계수입니다.
    - ×2: 각 블록에서 2개씩 샘플링 → 총 8192개
    - ×8: 각 블록에서 8개씩 샘플링 → 총 32,768개 (하지만 실제 코드에서는 8192로 제한)
- 실제로는 코드마다 샘플링 방식이 조금씩 다를 수 있습니다. (get_points_from_vox 함수 참고)

[8192 숫자의 유래]
- 8192 = 16 x 16 x 16 x 2 = 4096 x 2 = 8192
- 64³ 복셀 그리드에서 경계 및 내부/외부를 고르게 샘플링하기 위해 16³(4096) 기준에 2를 곱해서 8192개 포인트를 샘플링하도록 설계됨
- 샘플링 방식은 get_points_from_vox 함수 참고

[예시 실행]
python 2.5_voxel2pc.py --data_flag 1 --dataset_dir /workspace/data/cylinder_data
# 또는 기본값 사용 시
python 2.5_voxel2pc.py
"""
import numpy as np
import h5py, sys, os
from scipy.spatial import cKDTree as KDTree
import time
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_flag", type=int, default=1, help="동작 모드 (0: 단일 npz, 1: 배치 hdf5, 기본값 1)")
parser.add_argument("--dataset_dir", type=str, default="/workspace/data/cylinder_data", help="데이터셋 폴더 경로 [기본값: /workspace/data/cylinder_data]")
args = parser.parse_args()
data_flag = args.data_flag
dataset_dir = args.dataset_dir

batch_size_64 = 16*16*16*8

def get_points_from_vox(Pvoxel_model_64):
	
	# --- P 64 ---
	dim_voxel = 64
	batch_size = batch_size_64
	voxel_model_temp = Pvoxel_model_64
	
	sample_points = np.zeros([batch_size,3],np.uint8)
	sample_values = np.zeros([batch_size,1],np.uint8)
	batch_size_counter = 0
	voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
	nei = 2
	temp_range = list(range(nei,dim_voxel-nei,4))+list(range(nei+1,dim_voxel-nei,4))+list(range(nei+2,dim_voxel-nei,4))+list(range(nei+3,dim_voxel-nei,4))
	
	for j in temp_range:
		if (batch_size_counter>=batch_size): break
		for i in temp_range:
			if (batch_size_counter>=batch_size): break
			for k in temp_range:
				if (batch_size_counter>=batch_size): break
				if (np.max(voxel_model_temp[i-nei:i+nei+1,j-nei:j+nei+1,k-nei:k+nei+1])!= \
					np.min(voxel_model_temp[i-nei:i+nei+1,j-nei:j+nei+1,k-nei:k+nei+1])):
					#si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
					sample_points[batch_size_counter,0] = i#si+i*multiplier
					sample_points[batch_size_counter,1] = j#sj+j*multiplier
					sample_points[batch_size_counter,2] = k#sk+k*multiplier
					sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
					voxel_model_temp_flag[i,j,k] = 1
					batch_size_counter +=1
	if (batch_size_counter>=batch_size):
		print("64-- batch_size exceeded!")
		exceed_64_flag = 1
	else:
		exceed_64_flag = 0
		#fill other slots with random points
		while (batch_size_counter<batch_size):
			while True:
				i = random.randint(0,dim_voxel-1)
				j = random.randint(0,dim_voxel-1)
				k = random.randint(0,dim_voxel-1)
				if voxel_model_temp_flag[i,j,k] != 1: break
			#si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
			sample_points[batch_size_counter,0] = i#si+i*multiplier
			sample_points[batch_size_counter,1] = j#sj+j*multiplier
			sample_points[batch_size_counter,2] = k#sk+k*multiplier
			sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
			voxel_model_temp_flag[i,j,k] = 1
			batch_size_counter +=1
	
	# Psample_points_64 = sample_points
	# Psample_values_64 = sample_values
	points_value = np.concatenate((sample_points, sample_values), 1)
	return points_value

def voxel2pc_batch(dataset_dir):

	data_dict_name = os.path.join(dataset_dir, 'ae_voxel_points_samples.hdf5')
	data_dict = h5py.File(data_dict_name, 'r')
	voxels_all = data_dict['voxels'][:]
	data_dict.close()

	shape_number = voxels_all.shape[0]
	sampled_points = np.zeros((shape_number, batch_size_64, 4), np.int8)

	for index in range(shape_number):
		start_time = time.time()
		print(f'processing, {index}')

		voxels = voxels_all[index]
		voxels = voxels.reshape(64, 64, 64)

		points_value = get_points_from_vox(voxels)
		sampled_points[index, :, :] = points_value
		print('left time, ', (time.time() - start_time) * (shape_number - index) / 3600)

	hdf5_path = os.path.join(dataset_dir, 'voxel2pc.hdf5')
	hdf5_file = h5py.File(hdf5_path, 'w')
	hdf5_file.create_dataset("points", [shape_number, batch_size_64, 4], np.int8, compression=9)
	hdf5_file.create_dataset("voxels", [shape_number, 64, 64, 64, 1], np.int8, compression=9)
	hdf5_file["points"][:] = sampled_points
	hdf5_file["voxels"][:] = voxels_all
	hdf5_file.close()
	

def voxel2pc_single(dataset_dir):
	npz = np.load(dataset_dir + '/names.npz',  allow_pickle=True)
	names = npz["names"]

	# init kdtree
	shape_number = len(names)
	for index in range(shape_number):
		start_time = time.time()
		mesh_fn = names[index]
		print('processing, ', mesh_fn)
		pc_path = os.path.join(dataset_dir, 'shapes', mesh_fn, 'voxel_64.npz')

		voxels = np.load(pc_path)['voxels']
		voxels = voxels.reshape(64, 64, 64)
		#partialpc = partialpc*0.8 # scale the mesh

		points_value = get_points_from_vox(voxels)
		out_file = os.path.join(dataset_dir, 'shapes', mesh_fn, 'voxel2pc.npz')
		np.savez(out_file, points = points_value, names = mesh_fn)

		print('left time, ', (time.time() - start_time) * (len(names) - index) / 3600)
		
if __name__ == '__main__':
	if data_flag == 0:
		#voxel file npz
		voxel2pc_single(dataset_dir)
	else:
		#batch file hdf5
		voxel2pc_batch(dataset_dir)

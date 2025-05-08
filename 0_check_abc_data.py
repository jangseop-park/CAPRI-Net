"""
이 스크립트는 3D shape 학습 파이프라인에서 사용되는 주요 데이터 파일 구조를 확인하고, train/test 데이터의 필요성도 설명합니다.

[주요 데이터 파일 구조]
| 파일명              | 주요 키/shape                              | 용도/특징                        |
|---------------------|------------------------------------------|---------------------------------|
| ae_train.hdf5       | voxels (N,64,64,64,1), points (N,M,4)    | 복셀+포인트 기반 학습용            |
| ae_test.hdf5        | voxels (N,64,64,64,1), points (N,M,4)    | 복셀+포인트 기반 테스트용           |
| train_names.npz     | names (N,) 또는 train_names (N,)          | 학습 샘플명 리스트                |
| test_names.npz      | names (N,) 또는 test_names (N,)           | 테스트 샘플명 리스트              |
| points2mesh.hdf5    | points (N,M,6), values (N,M,1)           | 표면 포인트 기반 학습용            |
| voxel2mesh.hdf5     | voxels (N,64,64,64,1), points (N,8192,4) | 복셀→포인트 변환 기반 학습/테스트   |

- ae_train.hdf5: 복셀(occupancy grid)과 포인트별 occupancy 정보 모두 포함
- ae_test.hdf5: 복셀(occupancy grid)과 포인트별 occupancy 정보 모두 포함 (테스트용)
- train_names.npz: 학습에 사용할 샘플명 리스트
- test_names.npz: 테스트에 사용할 샘플명 리스트
- points2mesh.hdf5: 표면 포인트와 점유 정보만 포함
- voxel2mesh.hdf5: 복셀에서 샘플링한 포인트 클라우드와 occupancy 정보 포함

[train/test 데이터가 필요한 이유]
- train 데이터: 네트워크가 실제로 학습(가중치 업데이트)에 사용하는 데이터셋
- test 데이터: 학습에 사용되지 않은 새로운 데이터로, 모델의 일반화 성능(오버피팅 방지, 실제 성능 평가)에 필수

---
"""

import h5py
import numpy as np
import os

# 확인할 파일 경로를 지정하세요
root = '/workspace/data/abc_data'  # 예시 경로, 필요시 수정

file_list = [
    ('ae_train.hdf5', 'hdf5'),
    ('ae_test.hdf5', 'hdf5'),
    ('points2mesh.hdf5', 'hdf5'),
    ('train_names.npz', 'npz'),
    ('test_names.npz', 'npz'),
    ('voxel2mesh.hdf5', 'hdf5'),
]
#asdf


for fname, ftype in file_list:
    fpath = os.path.join(root, fname)
    print(f'\n===== {fname} =====')
    if not os.path.exists(fpath):
        print('파일이 존재하지 않습니다.')
        continue
    if ftype == 'hdf5':
        with h5py.File(fpath, 'r') as f:
            print('키 목록:', list(f.keys()))
            for k in f.keys():
                try:
                    print(f'  - {k}: shape={f[k].shape}, dtype={f[k].dtype}')
                except Exception as e:
                    print(f'  - {k}: shape/타입 확인 불가 ({e})')
    elif ftype == 'npz':
        npz = np.load(fpath, allow_pickle=True)
        print('키 목록:', list(npz.keys()))
        for k in npz.keys():
            arr = npz[k]
            print(f'  - {k}: shape={arr.shape}, dtype={arr.dtype}')


"""
출력 결과

===== ae_train.hdf5 =====
키 목록: ['points_16', 'points_32', 'points_64', 'voxels']
  - points_16: shape=(5000, 4608, 4), dtype=float32
  - points_32: shape=(5000, 9216, 4), dtype=float32
  - points_64: shape=(5000, 26624, 4), dtype=float32
  - voxels: shape=(5000, 64, 64, 64), dtype=float32

===== ae_test.hdf5 =====
키 목록: ['points_16', 'points_32', 'points_64', 'voxels']
  - points_16: shape=(1000, 4608, 4), dtype=uint8
  - points_32: shape=(1000, 9216, 4), dtype=uint8
  - points_64: shape=(1000, 28672, 4), dtype=uint8
  - voxels: shape=(1000, 64, 64, 64, 1), dtype=uint8

===== points2mesh.hdf5 =====
키 목록: ['points', 'voxels']
  - points: shape=(1000, 8192, 6), dtype=float32
  - voxels: shape=(1000, 64, 64, 64), dtype=float32

===== train_names.npz =====
키 목록: ['train_names', 'test_names', 'names']
  - train_names: shape=(5000,), dtype=<U45
  - test_names: shape=(1078,), dtype=<U45
  - names: shape=(6078,), dtype=<U45

===== test_names.npz =====
키 목록: ['test_names']
  - test_names: shape=(1000,), dtype=<U45

===== voxel2mesh.hdf5 =====
키 목록: ['points', 'voxels']
  - points: shape=(1000, 32768, 4), dtype=float32
  - voxels: shape=(1000, 64, 64, 64), dtype=float32
"""
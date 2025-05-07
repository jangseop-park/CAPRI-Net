"""
1.1.check_data.py

이 스크립트는 HDF5 파일로 저장된 voxel 및 point cloud 데이터를 불러와 통계 정보를 출력하고, PyVista로 3D voxel grid와 SDF point cloud 이미지를 저장합니다.

[주요 기능]
- HDF5 파일에서 voxel/point cloud 데이터 로드
- 데이터 shape, min/max, unique 값 등 통계 출력
- 임의 샘플의 voxel/point cloud 시각화 및 이미지 저장

[입력 예시]
- data_source: 데이터셋 폴더 경로 (예: '/workspace/data/abc_data')
- voxel2pc.hdf5 또는 voxel2mesh.hdf5 파일

[출력 예시]
- figures/voxel_grid_sample.png
- figures/sdf_point_cloud_sample.png
- 콘솔에 데이터 통계 및 샘플 정보
"""

# =============================
# 사용자 설정 파라미터 (여기만 수정)
# =============================
data_source = '/workspace/data/abc_data'  # 데이터셋 폴더 경로
hdf5_filename = 'voxel2pc.hdf5'           # 사용할 HDF5 파일명 (예: voxel2pc.hdf5, voxel2mesh.hdf5 등)

# =============================

import os
import numpy as np
import torch
import h5py
import pyvista as pv

# 🔹 X 서버 없이 PyVista 실행
pv.start_xvfb()

# -------------------------------------------------------
# 1️⃣ 데이터 로드
# -------------------------------------------------------
filename_voxels = os.path.join(data_source, hdf5_filename)
data_dict = h5py.File(filename_voxels, 'r')

# Voxel Data (B, 1, 64, 64, 64)
data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()
data_voxels = data_voxels.squeeze(-1).unsqueeze(1)  # (B, 1, 64, 64, 64)
print(f"[data_voxels] Original shape: {data_voxels.shape}")
print(f"[data_voxels] Min: {data_voxels.min()}, Max: {data_voxels.max()}, Unique values: {torch.unique(data_voxels).shape[0]}")

# Point Data (B, N, 4) -> [:, :, :3]는 3D 좌표, [:, :, 3]은 SDF 값
data_points = torch.from_numpy(data_dict['points'][:]).float()  # (B, N, 4)
print(f"[data_points] Original shape: {data_points.shape}")

# 좌표 정규화
data_points[:, :, :3] = (data_points[:, :, :3] + 0.5)/64-0.5
print(f"[data_points] Transformed shape: {data_points.shape}")
print(f"[data_points] x Min: {data_points[:, :, 0].min()}, x Max: {data_points[:, :, 0].max()}")
print(f"[data_points] y Min: {data_points[:, :, 1].min()}, y Max: {data_points[:, :, 1].max()}")
print(f"[data_points] z Min: {data_points[:, :, 2].min()}, z Max: {data_points[:, :, 2].max()}")

# 랜덤 샘플링하여 일부 데이터 출력
sample_idx = torch.randint(0, data_voxels.shape[0], (1,)).item()
print(f"\n### Sample Index: {sample_idx} ###")
print(f"Voxel Data Sample (flattened): {data_voxels[sample_idx].view(-1)}")
print(f"Point Cloud Sample: {data_points[sample_idx, :5]}")  # 상위 5개 포인트 출력

# -------------------------------------------------------
# 2️⃣ 3D Voxel Grid 시각화 (이미지 저장)
# -------------------------------------------------------
rel_fig_path = os.path.relpath(data_source, "/workspace")
figures_dir = os.path.join("/workspace/figures", rel_fig_path)
os.makedirs(figures_dir, exist_ok=True)

voxel_img_path = os.path.join(figures_dir, "voxel_grid_sample.png")
sdf_img_path = os.path.join(figures_dir, "sdf_point_cloud_sample.png")

voxels = data_voxels[sample_idx].squeeze().numpy()  # (64, 64, 64)
plotter_voxel = pv.Plotter(off_screen=True)
grid = pv.ImageData()
grid.dimensions = np.array(voxels.shape) + 1  # (64, 64, 64) → (65, 65, 65)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = voxels.flatten(order="F")
plotter_voxel.add_mesh(grid.threshold(0.5), opacity=0.5, show_edges=True, cmap="coolwarm")
plotter_voxel.add_axes()
plotter_voxel.add_title("Voxel Grid")
plotter_voxel.screenshot(voxel_img_path)
print(f"Voxel Grid image saved at: {voxel_img_path}")

# -------------------------------------------------------
# 3️⃣ SDF Point Cloud 시각화 (이미지 저장)
# -------------------------------------------------------
points = data_points[sample_idx].numpy()  # (N, 4)
plotter_points = pv.Plotter(off_screen=True)
xyz = points[:, :3]
sdf_values = points[:, 3]
point_cloud = pv.PolyData(xyz)
point_cloud["SDF"] = sdf_values
plotter_points.add_mesh(point_cloud, scalars="SDF", point_size=5.0, render_points_as_spheres=True, cmap="coolwarm")
plotter_points.add_axes()
plotter_points.add_title("SDF Point Cloud")
plotter_points.screenshot(sdf_img_path)
print(f"SDF Point Cloud image saved at: {sdf_img_path}") 
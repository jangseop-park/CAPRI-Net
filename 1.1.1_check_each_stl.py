"""
1.1.1.check_each_stl.py

이 스크립트는 여러 STL 파일을 불러와 PyVista로 3D 시각화 이미지를 저장합니다.

[주요 기능]
- 지정한 STL 파일들을 3D로 시각화하여 PNG 이미지로 저장
- STL 파일 경로 구조를 /workspace/figures 하위에 그대로 반영해 저장 (data/ 포함)

[입력 예시]
- stl_files: STL 파일 경로 리스트 (예: '/workspace/data/su_data/shapes/0/Polyline.STL' 등)

[출력 예시]
- /workspace/figures/data/su_data/shapes/0/Polyline.STL.png
- /workspace/figures/data/su_data/shapes/1/Spline.STL.png
- ...
- 콘솔에 저장 경로 출력
"""
import pyvista as pv
import os

# 🔹 X 서버 없이 PyVista 실행
pv.start_xvfb()

# STL 파일 경로 리스트 (필요에 따라 수정)
stl_files = [
    "/workspace/data/su_data/shapes/0/Polyline.STL",
    "/workspace/data/su_data/shapes/1/Spline.STL",
    "/workspace/data/su_data/shapes/2/Frame_body.STL",
    "/workspace/data/su_data/shapes/3/Frame_Timming_gear.STL",
    "/workspace/data/su_data/shapes/4/output_mesh_60x30x20.stl"
]

# figures 폴더 생성
os.makedirs("/workspace/figures", exist_ok=True)

# STL 파일을 불러와 시각화 및 이미지 저장
for stl_file in stl_files:
    mesh = pv.read(stl_file)
    if mesh.n_points == 0:
        print(f"❌ {stl_file} 파일이 비어 있음!")
        continue
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="lightblue", opacity=1.0, show_edges=True)
    plotter.add_axes()
    plotter.add_title(f"STL Visualization: {os.path.basename(stl_file)}")
    plotter.camera_position = 'xy'
    plotter.view_isometric()
    plotter.reset_camera()
    rel_path = os.path.relpath(stl_file, "/workspace")
    img_path = os.path.join("/workspace/figures", rel_path + ".png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plotter.screenshot(img_path)
    print(f"✅ STL image saved at: {img_path}") 
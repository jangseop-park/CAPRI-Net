"""
1.1.2.check_all_stl.py

이 스크립트는 /workspace/data/cylinder_data/shapes 하위의 cylinder_0 ~ cylinder_299 폴더에서 optimized_design.stl 파일을 불러와 PyVista로 3D 시각화 이미지를 저장합니다.

[주요 기능]
- 각 폴더의 optimized_design.stl 파일을 3D로 시각화하여 PNG 이미지로 저장
- STL 파일 경로 구조를 /workspace/figures 하위에 그대로 반영해 저장 (data/ 포함)

[입력 예시]
- /workspace/data/cylinder_data/shapes/cylinder_0/optimized_design.stl
- /workspace/data/cylinder_data/shapes/cylinder_1/optimized_design.stl
- ...
- /workspace/data/cylinder_data/shapes/cylinder_299/optimized_design.stl

[출력 예시]
- /workspace/figures/data/cylinder_data/shapes/cylinder_0/optimized_design.stl.png
- /workspace/figures/data/cylinder_data/shapes/cylinder_1/optimized_design.stl.png
- ...
- /workspace/figures/data/cylinder_data/shapes/cylinder_299/optimized_design.stl.png
- 콘솔에 저장 경로 출력
"""
import os
import pyvista as pv

# 🔹 X 서버 없이 PyVista 실행
pv.start_xvfb()

shapes_root = "/workspace/data/cylinder_data/shapes"
folders = [d for d in os.listdir(shapes_root) if os.path.isdir(os.path.join(shapes_root, d))]

for folder in folders:
    stl_path = os.path.join(shapes_root, folder, "optimized_design.stl")
    if not os.path.exists(stl_path):
        print(f"❌ STL 파일 없음: {stl_path}")
        continue
    mesh = pv.read(stl_path)
    if mesh.n_points == 0:
        print(f"❌ {stl_path} 파일이 비어 있음!")
        continue
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="lightblue", opacity=1.0, show_edges=True)
    plotter.add_axes()
    plotter.add_title(f"STL Visualization: {os.path.basename(stl_path)}")
    plotter.camera_position = 'xy'
    plotter.view_isometric()
    plotter.reset_camera()
    rel_path = os.path.relpath(stl_path, "/workspace")
    img_path = os.path.join("/workspace/figures", rel_path + ".png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plotter.screenshot(img_path)
    print(f"✅ STL image saved at: {img_path}") 
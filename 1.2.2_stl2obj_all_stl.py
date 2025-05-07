"""
1.1.2.check_all_stl.py

이 스크립트는 /workspace/data/cylinder_data/shapes 하위의 cylinder_0 ~ cylinder_299 폴더에서 optimized_design.stl 파일을 찾아 model.obj로 변환합니다.

[주요 기능]
- cylinder_0 ~ cylinder_299 폴더를 순회하며 optimized_design.stl 파일을 trimesh로 불러와 model.obj로 저장
- 변환 성공 시 경로를 출력, 실패 시 에러 메시지 출력

[입력 예시]
- /workspace/data/cylinder_data/shapes/cylinder_0/optimized_design.stl
- /workspace/data/cylinder_data/shapes/cylinder_1/optimized_design.stl
- ...
- /workspace/data/cylinder_data/shapes/cylinder_299/optimized_design.stl

[출력 예시]
- /workspace/data/cylinder_data/shapes/cylinder_0/model.obj
- /workspace/data/cylinder_data/shapes/cylinder_1/model.obj
- ...
- /workspace/data/cylinder_data/shapes/cylinder_299/model.obj
- 콘솔에 변환 결과 출력
"""
import os
import trimesh

shapes_root = "/workspace/data/cylinder_data/shapes"

folders = [d for d in os.listdir(shapes_root) if os.path.isdir(os.path.join(shapes_root, d))]

for folder in folders:
    stl_path = os.path.join(shapes_root, folder, "optimized_design.stl")
    obj_path = os.path.join(shapes_root, folder, "model.obj")
    if not os.path.exists(stl_path):
        print(f"❌ STL 파일 없음: {stl_path}")
        continue
    try:
        mesh = trimesh.load_mesh(stl_path)
        mesh.export(obj_path)
        print(f"✅ 변환 완료: {obj_path}")
    except Exception as e:
        print(f"❌ 변환 실패: {stl_path} → {e}") 
"""
1.2.stl2obj.py

이 스크립트는 여러 STL 파일을 OBJ 파일로 변환합니다.

[주요 기능]
- 지정한 STL 파일들을 trimesh로 불러와 OBJ 파일로 저장
- 변환 완료 시 경로를 출력

[입력 예시]
- input_stl: 변환할 STL 파일 경로
- output_obj: 저장할 OBJ 파일 경로

[출력 예시]
- 각 폴더별 model.obj 파일 생성
- 콘솔에 변환 완료 메시지 출력
"""
import trimesh
import os

# 변환할 STL/OBJ 파일 목록
stl_files = [
    "/workspace/data/su_data/shapes/0/Polyline.STL",
    "/workspace/data/su_data/shapes/1/Spline.STL",
    "/workspace/data/su_data/shapes/2/Frame_body.STL",
    "/workspace/data/su_data/shapes/3/Frame_Timming_gear.STL",
    "/workspace/data/su_data/shapes/4/output_mesh_60x30x20.stl",
]

for input_stl in stl_files:
    output_obj = os.path.join(os.path.dirname(input_stl), "model.obj")
    mesh = trimesh.load_mesh(input_stl)
    mesh.export(output_obj)
    print(f"파일 변환 완료: {output_obj}") 
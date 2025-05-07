"""
2.3_make_name_list.py

이 스크립트는 데이터셋 폴더 내의 파일 이름 리스트를 다양한 방식으로 npz 또는 txt 파일로 저장합니다.

[동작 설명]
- in_path 폴더 내의 파일 이름을 읽어 names.npz, names.txt 등으로 저장하거나, 분할(split) 및 변환 작업을 수행합니다.
- data_flag 값에 따라 동작이 달라집니다.
    0: names.npz 파일을 train/test로 분할하여
       → train_names.npz, test_names.npz를 in_path와 같은 상위 폴더에 저장
    1: names.txt를 names.npz로 변환하여
       → names.npz를 in_path에 저장
    2: 폴더 내 파일명을 names.npz로 저장
       → names.npz, test_names.npz를 in_path의 상위 폴더에 저장
    3: shapenet 형식의 train.txt/test.txt를 names.npz로 저장
       → names.npz를 in_path에 저장
    4: shape index 생성
       → fine-tuning_index_all.npz를 in_path의 상위 폴더에 저장
    기타: names.txt로 저장
       → names.txt를 in_path의 상위 폴더에 저장

[입력]
- in_path: 파일 이름 리스트를 만들 폴더 경로 (예: /workspace/data/cylinder_data/shapes)
- data_flag: 동작 모드 (0~4, 기본값 2)

[출력 및 저장 구조]
- data_flag 값에 따라 아래와 같이 다양한 파일이 생성되며, 각 파일의 저장 구조는 다음과 같습니다.

| data_flag | 생성 파일명 (경로)                  | 파일 포맷/구조 설명                                                                 |
|-----------|-------------------------------------|--------------------------------------------------------------------------------|
| 0         | train_names.npz, test_names.npz     | npz: 각각 'train_names', 'test_names' 키로 리스트 저장                            |
|           |   (in_path의 상위 폴더)             |                                                                                  |
| 1         | names.npz (in_path)                 | npz: 'names' 키로 리스트 저장                                                     |
| 2         | names.npz, test_names.npz           | npz: 'names', 'test_names' 키로 전체 리스트 저장 (in_path의 상위 폴더)              |
|           |   (in_path의 상위 폴더)             |                                                                                  |
| 3         | names.npz (in_path)                 | npz: 'train_names', 'test_names', 'names' 키로 각각 리스트 저장                   |
| 4         | fine-tuning_index_all.npz           | npz: 'indexes' 키로 인덱스 리스트 저장 (in_path의 상위 폴더)                        |
|           |   (in_path의 상위 폴더)             |                                                                                  |
| 기타      | names.txt (in_path의 상위 폴더)     | txt: 한 줄에 하나씩 파일명 저장                                                      |

- npz 파일은 numpy의 savez로 저장되며, 각 키에 해당하는 리스트(문자열)가 배열로 저장됩니다.
- names.txt는 각 파일명을 한 줄에 하나씩 저장한 텍스트 파일입니다.

[예시 실행]
python 2.3_make_name_list.py --in_path /workspace/data/cylinder_data/shapes --data_flag 2
# 또는 기본값 사용 시
python 2.3_make_name_list.py
"""
import trimesh
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import traceback
import argparse
import sys

####
#input one directory and all files in it
#output npz file
def Read_directory_to_npz(in_path):
	def model_names(model_path):
		""" Return model names"""
		model_names = [name for name in os.listdir(model_path)]
		return model_names

	models = model_names(in_path)
	models.sort()
	print(models)
	out_path = os.path.join(os.path.dirname(in_path), 'names.npz')
	np.savez(out_path, names = models, test_names=models)
	print('finished {}'.format(in_path))

	print('save to {}'.format(out_path))

def Read_directory_to_txt(in_path):
	def model_names(model_path):
		""" Return model names"""
		model_names = [name for name in os.listdir(model_path)]
		return model_names

	models = model_names(in_path)
	models.sort()
	#print(models)
	out_path = os.path.join(os.path.dirname(in_path), 'names.txt')
	#np.savez(out_path, names = models, test_names=models)
	f = open(out_path, 'w')
	for name in models:
		f.write(name+'\n')
	f.close()
	print('finished {}'.format(in_path))

	print('save to {}'.format(out_path))

def generate_shape_index(in_path):
	def model_names(model_path):
		""" Return model names"""
		model_names = [name for name in os.listdir(model_path)]
		return model_names

	models = model_names(in_path)
	models.sort()
	#print(models)
	all_name_path = os.path.join(os.path.dirname(in_path), 'names.npz')
	all_test_names = np.load(all_name_path)['test_names']
	indexs = []
	for i in range(len(all_test_names)):
		#print('all_test_names[i], ', all_test_names[i])
		#print('model, ', models[i])
		if all_test_names[i]+'.off' in models:
			indexs.append(i)
	print(len(indexs))
	indexs = np.array(indexs)
	out_path = os.path.join(os.path.dirname(in_path), 'fine-tuning_index_all.npz')

	np.savez(out_path, indexes = indexs)
	print('finished {}'.format(in_path))

	print('save to {}'.format(out_path))

def simple_read_names_txt(in_path):
	test_file = os.path.join(in_path, 'names.txt')
	with open(test_file) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	test_content = [x.strip()[:-4] for x in content]
	out_file = os.path.join(in_path, 'names.npz')

	np.savez(out_file, names = test_content)
	print('save to {}'.format(out_file))

def split_npz(in_path, split_num):
	file_path = os.path.join(in_path, 'names.npz')

	# models = np.load(in_path)['names']

	# models_train = models[:split_num]
	# models_test = models[split_num:]

	models_train = np.load(file_path)['train_names']
	models_test = np.load(file_path)['test_names']

	train_file = os.path.join(in_path, 'train_names.npz')
	test_file = os.path.join(in_path, 'test_names.npz')

	np.savez(train_file, train_names = models_train)
	np.savez(test_file, test_names = models_test)
	#np.savez(in_path, train_names = models_train, test_names = models_test, names=models)
	print('finished {}'.format(in_path))

def read_names_shapenet(in_path):

	#cate = os.path.basename(in_path)#03001627

	#train_file = os.path.join(in_path, '%s_vox256_train.txt'%(cate))
	train_file = os.path.join(in_path, 'train.txt')
	with open(train_file) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	train_content = [(x.strip()) for x in content]

	test_file = os.path.join(in_path, 'test.txt')
	with open(test_file) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	test_content = [(x.strip()) for x in content]
	#print(test_content)
	out_path = os.path.join(in_path, 'names.npz')
	np.savez(out_path, train_names = train_content, test_names = test_content, names = train_content + test_content)
	print('finished {}'.format(in_path))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--in_path", type=str, default="/workspace/data/cylinder_data/shapes", help="파일 이름 리스트를 만들 폴더 경로 [기본값: /workspace/data/cylinder_data/shapes]")
	parser.add_argument("--data_flag", type=int, default=2, help="동작 모드 (0~4, 기본값 2)")
	args = parser.parse_args()
	in_path = args.in_path
	data_flag = args.data_flag
	#0 split npz, 1 transfer txt to npz, 2 read directory into npz, 3 read shapenet names
	if data_flag == 0:
		split_npz(in_path, split_num=5000)
	elif data_flag == 1:
		simple_read_names_txt(in_path)
	elif data_flag == 2:
		#read names into npz
		Read_directory_to_npz(in_path)
	elif data_flag == 3:
		read_names_shapenet(in_path)
	elif data_flag == 4:
		generate_shape_index(in_path)
	else:
		Read_directory_to_txt(in_path)
	
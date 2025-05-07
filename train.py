"""
train.py

이 스크립트는 3D shape 데이터셋을 학습하는 메인 트레이닝 파이프라인입니다.

[전체 동작 개요]
- 실험 폴더(experiment_directory) 및 specs.json 설정을 불러와 네트워크를 학습합니다.
- 데이터셋(hdf5/npz 등)에서 voxel/point cloud 데이터를 불러와 인코더, 디코더, 제너레이터 네트워크를 학습합니다.
- 러닝레이트 스케줄, 옵티마이저, 체크포인트 저장, best 모델 저장 등 학습에 필요한 모든 과정을 자동으로 처리합니다.
- 이어서 학습(continue) 및 다양한 실험 옵션(phase, surface, shapenet 등)을 지원합니다.

[학습 Phase 설명]
- Phase 0: 기본적인 shape 인코딩/디코딩 학습
  - Encoder: 입력 shape를 latent code로 인코딩
  - Decoder: latent code를 primitive 파라미터로 디코딩
  - 실행: python train.py --phase 0

- Phase 1: primitive 조합 학습
  - Generator: primitive들을 조합하여 최종 shape 생성
  - Phase 0에서 학습된 Encoder/Decoder 사용
  - 실행: python train.py --phase 1 --continue best_stage0_64.pth

- Phase 2: 전체 네트워크 fine-tuning
  - Encoder, Decoder, Generator 전체를 함께 fine-tuning
  - Phase 1까지 학습된 모델을 기반으로 학습
  - 실행: python train.py --phase 2 --continue best_stage1_64.pth

[Phase별 생성되는 파일]
- Phase 0:
  - best_stage0_64.pth: Phase 0의 best 모델
  - latest.pth: 최신 모델
  - {epoch}.pth: 특정 에폭의 모델

- Phase 1:
  - best_stage1_64.pth: Phase 1의 best 모델
  - latest.pth: Phase 1의 최신 모델
  - {epoch}.pth: Phase 1의 특정 에폭 모델

- Phase 2:
  - best_stage2_64.pth: Phase 2의 best 모델
  - latest.pth: Phase 2의 최신 모델
  - {epoch}.pth: Phase 2의 특정 에폭 모델

[학습 완료 확인 방법]
1. 로그에서 "best loss updated" 메시지가 더 이상 출력되지 않음
2. loss 값이 수렴하는지 확인
3. best_stage{phase}_64.pth 파일이 더 이상 업데이트되지 않음

[주요 옵션 설명]
1. --experiment (-e): 실험 디렉토리 지정 (기본값: "abc_voxel")
2. --continue: 이전 학습 체크포인트에서 이어서 학습할 경우 체크포인트 파일 경로
3. --leaky: soft min max 옵션 활성화
4. --grid_sample: 데이터셋 옵션 (기본값: 64)
5. --surface: 포인트 클라우드 옵션 (기본값: False)
6. --shapenet_flag: ShapeNet 데이터셋 옵션 (기본값: False)
7. --gpu (-g): GPU ID (기본값: 0)
8. --phase (-p): 학습 단계 (기본값: 0)

[데이터셋 옵션별 특징]
1. 기본 모드 (surface=False, shapenet_flag=False):
   - 일반적인 3D shape 학습
   - 복셀과 포인트 기반 occupancy 정보 사용

2. Surface 모드 (surface=True):
   - 표면 포인트 클라우드 기반 학습
   - 포인트 분포만으로 global feature 학습

3. ShapeNet 모드 (shapenet_flag=True):
   - ShapeNet 데이터셋 특화 학습
   - 복셀과 포인트 기반 occupancy 정보 사용

[학습 루프 구조]
- DataLoader로 배치 반복
- 각 배치에서 랜덤 포인트 배치 선택 및 분할
- 인코더(Encoder)로 shape code 추출 → 디코더(Decoder)로 primitive 생성 → 제너레이터(Generator)로 예측값 생성
- 손실(loss) 계산 및 역전파, 옵티마이저 스텝
- 주기적으로 체크포인트/베스트 모델 저장, 로그 및 남은 시간 출력
"""
import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time

import utils
import utils.workspace as ws
from networks.losses import loss

class LearningRateSchedule:
	def get_learning_rate(self, epoch):
		pass


class ConstantLearningRateSchedule(LearningRateSchedule):
	def __init__(self, value):
		self.value = value

	def get_learning_rate(self, epoch):
		return self.value


class StepLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, interval, factor):
		self.initial = initial
		self.interval = interval
		self.factor = factor

	def get_learning_rate(self, epoch):

		return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, warmed_up, length):
		self.initial = initial
		self.warmed_up = warmed_up
		self.length = length

	def get_learning_rate(self, epoch):
		if epoch > self.length:
			return self.warmed_up
		return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

	schedule_specs = specs["LearningRateSchedule"]

	schedules = []

	for schedule_specs in schedule_specs:

		if schedule_specs["Type"] == "Step":
			schedules.append(
				StepLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Interval"],
					schedule_specs["Factor"],
				)
			)
		elif schedule_specs["Type"] == "Warmup":
			schedules.append(
				WarmupLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Final"],
					schedule_specs["Length"],
				)
			)
		elif schedule_specs["Type"] == "Constant":
			schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

		else:
			raise Exception(
				'no known learning rate schedule of type "{}"'.format(
					schedule_specs["Type"]
				)
			)

	return schedules

def get_spec_with_default(specs, key, default):
	try:
		return specs[key]
	except KeyError:
		return default

def init_seeds(seed=0):
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It's safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It's safe to call this function if CUDA is not available; in that case, it is silently ignored.
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main_function(experiment_directory, continue_from, phase, grid_sample, leaky, sample_surface, shapenet_flag):

	init_seeds()

	logging.debug("running " + experiment_directory)

	specs = ws.load_experiment_specifications(experiment_directory)
	print(specs["Description"])
	logging.info("Experiment description: \n" + specs["Description"])

	data_source = specs["DataSource"]

	arch = __import__("networks." + specs["NetworkArch"], fromlist=["Encoder", "Decoder", "Generator"])

	checkpoints = list(
		range(
			specs["SnapshotFrequency"],
			specs["NumEpochs"] + 1,
			specs["SnapshotFrequency"],
		)
	)
	
	for checkpoint in specs["AdditionalSnapshots"]:
		checkpoints.append(checkpoint)
	checkpoints.sort()
	print(checkpoints)
	lr_schedules = get_learning_rate_schedules(specs)

	def save_latest(epoch):

		ws.save_model_parameters(experiment_directory, "latest.pth", encoder, decoder, generator, optimizer_all, epoch)
		
	def save_checkpoints(epoch):

		ws.save_model_parameters(experiment_directory, str(epoch) + ".pth", encoder, decoder, generator, optimizer_all, epoch)
	
	def save_checkpoints_best(epoch):

		ws.save_model_parameters(experiment_directory, "best_stage%d_%d.pth"%(phase, grid_sample), encoder, decoder, generator, optimizer_all, epoch)
		
	def signal_handler(sig, frame):
		logging.info("Stopping early...")
		sys.exit(0)

	def adjust_learning_rate(lr_schedules, optimizer, epoch):

		for i, param_group in enumerate(optimizer.param_groups):
			param_group["lr"] = lr_schedules[0].get_learning_rate(epoch)

	signal.signal(signal.SIGINT, signal_handler)

	#set batch size based on GPU memory size
	if grid_sample == 32:
		scene_per_batch = 24
	elif grid_sample==64:
		scene_per_batch = 24
	else:
		scene_per_batch = 24

	encoder = arch.Encoder().cuda()
	decoder = arch.Decoder().cuda()
	generator = arch.Generator().cuda()
	logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))
	
	num_epochs = specs["NumEpochs"]

	if sample_surface:
		occ_dataset = utils.dataloader.SurfaceSamples(
			data_source, test_flag=False
		)
	else:
		occ_dataset = utils.dataloader.GTSamples(
			data_source, grid_sample=grid_sample, test_flag=False, shapenet_flag=shapenet_flag
		)

	data_loader = data_utils.DataLoader(
		occ_dataset,
		batch_size=scene_per_batch,
		shuffle=True,
		num_workers=4
	)

	num_scenes = len(occ_dataset)

	logging.info("There are {} shapes".format(num_scenes))

	logging.debug(decoder)
	logging.debug(encoder)
	logging.debug(generator)

	optimizer_all = torch.optim.Adam(
		[
			{
				"params": decoder.parameters(),
				"lr": lr_schedules[0].get_learning_rate(0),
				"betas": (0.5, 0.999),
			},
			{
				"params": encoder.parameters(),
				"lr": lr_schedules[0].get_learning_rate(0),
				"betas": (0.5, 0.999),
			},
			{
				"params": generator.parameters(),
				"lr": lr_schedules[0].get_learning_rate(0),
				"betas": (0.5, 0.999),
			},
		]
	)
	start_epoch = 0
	if continue_from is not None:

		logging.info('continuing from "{}"'.format(continue_from))

		model_epoch = ws.load_model_parameters(
			experiment_directory, continue_from, 
			encoder,
			decoder,
			generator,
			optimizer_all
		)

		start_epoch = model_epoch + 1
		logging.debug("loaded")

	logging.info("starting from epoch {}".format(start_epoch))
	logging.info(f"Training, expriment {experiment_directory}, batch size {scene_per_batch}, phase {phase}, \
		grid_sample {grid_sample}, leaky {leaky} , shapenet_flag {shapenet_flag}")
	decoder.train()
	encoder.train()
	generator.train()

	start_time = time.time()
	
	last_epoch_time = 0

	point_batch_size = 16*16*16*2

	load_point_batch_size = occ_dataset.data_points.shape[1]
	point_batch_num = int(load_point_batch_size/point_batch_size)
	print('point batch num, ', point_batch_num)

	best_loss = 999

	for epoch in range(start_epoch, start_epoch + num_epochs):
		
		adjust_learning_rate(lr_schedules, optimizer_all, epoch - start_epoch)

		avarage_left_loss = 0
		avarage_right_loss = 0
		avarage_total_loss = 0
		avarage_num = 0
		iters = 0
		for voxels, occ_data, shape_names, indices in data_loader:

			voxels = voxels.cuda()
			occ_data = occ_data.cuda()
			iters += 1

			which_batch = torch.randint(point_batch_num+1, (1,))
			if which_batch == point_batch_num:
				xyz = occ_data[:,-point_batch_size:, :3]
				occ_gt = occ_data[:,-point_batch_size:, 3]
			else:
				xyz = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, :3]
				occ_gt = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, 3]
			
			optimizer_all.zero_grad()

			shape_code = encoder(voxels)
			primitives = decoder(shape_code)

			G_left, G_right, net_out, net_out_convexes = generator(xyz, primitives, phase, leaky)
			value_loss_left, value_loss_right, total_loss = loss(phase, G_left, G_right, occ_gt, 
				generator.concave_layer_weights, generator.convex_layer_weights)
					
			total_loss.backward()
			optimizer_all.step()

			avarage_left_loss += value_loss_left.detach().item()
			avarage_right_loss += value_loss_right.detach().item()
			avarage_total_loss += total_loss.detach().item()
			avarage_num += 1

		if (epoch+1) % 1 == 0:
			seconds_elapsed = time.time() - start_time
			ava_epoch_time = (seconds_elapsed - last_epoch_time)/1
			left_time = ava_epoch_time*(num_epochs+ start_epoch- epoch)/3600
			last_epoch_time = seconds_elapsed
			left_loss = avarage_left_loss/avarage_num
			right_loss = avarage_right_loss/avarage_num
			t_loss = avarage_total_loss/avarage_num
			logging.debug("epoch = {}/{} err_left = {:.6f}, err_right = {:.6f}, total_loss={:.6f}, 1 epoch time ={:.6f}, left time={:.6f}".format(epoch, 
				num_epochs+start_epoch, left_loss, right_loss, t_loss, ava_epoch_time, left_time))

			if t_loss < best_loss:
				print('best loss updated, ', t_loss)
				save_checkpoints_best(epoch)
				best_loss = t_loss

		if (epoch-start_epoch+1) in checkpoints:
			save_checkpoints(epoch)


if __name__ == "__main__":

	import argparse

	arg_parser = argparse.ArgumentParser(description="Train a Network")

	# ========== surface 옵션 관련 ==========
	arg_parser.add_argument("--experiment", "-e", dest="experiment_directory", default="abc_voxel", help="The experiment directory. This directory should include experiment specifications in 'specs.json', and logging will be done in this directory as well.")
	arg_parser.add_argument("--continue", dest="continue_from", default=None, help="continue from checkpoint")
	arg_parser.add_argument("--leaky", dest="leaky", action="store_true", help="soft min max option")
	arg_parser.add_argument("--grid_sample", dest="grid_sample", default=64, help="dataset option")
	arg_parser.add_argument("--surface", dest="surface", default=False, help="point cloud option")
	arg_parser.add_argument("--shapenet_flag", dest="shapenet_flag", default=False, help="dataset option")
	arg_parser.add_argument("--gpu", "-g", dest="gpu", default=0, help="gpu id")
	arg_parser.add_argument("--phase", "-p", dest="phase", default=0, help="phase stage")
	utils.add_common_args(arg_parser)

	args = arg_parser.parse_args()

	utils.configure_logging(args)
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
	print('gpu: ,', int(args.gpu))
	main_function(args.experiment_directory, args.continue_from, int(args.phase), \
		int(args.grid_sample), args.leaky, args.surface, args.shapenet_flag)

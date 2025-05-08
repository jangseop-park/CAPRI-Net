#!/bin/bash
container_name="$1"
jupyter_port="$2"
tensorboard_port="$3"

docker run \
--gpus all \
-itd \
--shm-size=200g \
--ulimit memlock=-1 \
--ulimit stack=37108864 \
--volume /home/lab1_jwj/research/CAPRI-Net:/workspace \
--volume /mnt/nas1/1_projects/joint_research/CAPRI-Net/data:/workspace/data \
-p "$jupyter_port":8888 \
-p "$tensorboard_port":6006 \
--name "$container_name" d2csg_image:latest
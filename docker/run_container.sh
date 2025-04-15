#!/bin/bash

export IPD_DATASET_ROOT_FOLDER="/media/joao/061A31701A315E3D1/ipd-dataset/bpc_baseline/datasets"

docker rm -f foundationpose 2>/dev/null || true

xhost +local:docker &&\
docker run \
    --gpus all \
    -it \
    --name foundationpose \
    -v "$IPD_DATASET_ROOT_FOLDER:/ipd:ro" \
    -v "/home/joao/Downloads/algorithm_output:/algorithm_output:rw" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=${DISPLAY} \
    -p 5678:5678 \
    foundationpose_custom:latest


#!/bin/bash

docker rm -f foundationpose 2>/dev/null || true

xhost +local:docker &&\
docker run \
    --gpus all \
    -it \
    --name foundationpose \
    -v "/home/joao/Downloads/ipd-train-pbr-sample/ipd:/ipd:ro" \
    -v "/home/joao/Downloads/algorithm_output:/algorithm_output:rw" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=${DISPLAY} \
    -p 5678:5678 \
    foundationpose_custom:latest


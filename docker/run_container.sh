#!/bin/bash

docker rm -f foundationpose 2>/dev/null || true
DIR=$(pwd)/../

xhost +local:docker &&\
docker run \
    --gpus all \
    -it \
    --name foundationpose \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=${DISPLAY} \
    foundationpose_custom:latest


#!/bin/bash

docker rm -f foundationpose
DIR=$(pwd)/../

xhost +local:docker &&\
docker run \
    --gpus all \
    -it \
    --name foundationpose \
    -v $DIR:$DIR \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=${DISPLAY} \
    foundationpose_custom:latest \
    bash -c "cd $DIR && bash"

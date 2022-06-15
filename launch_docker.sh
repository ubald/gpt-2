#!/usr/bin/env bash

docker run -u $(id -u):$(id -g) --runtime=nvidia -v ${PWD}:/gpt-2 -p 8000:8000 -it gpt-2
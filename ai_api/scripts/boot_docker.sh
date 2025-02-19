#!/bin/bash

IMAGE_NAME=ai_api
IMAGE_TAG=latest
CONTAINER_NAME=ai_api_app

docker run --rm \
  -p 8000:8000 \
  --name ${CONTAINER_NAME} \
  ${IMAGE_NAME}:${IMAGE_TAG}

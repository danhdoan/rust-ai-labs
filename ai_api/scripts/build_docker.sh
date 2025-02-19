#!/bin/bash

IMAGE_NAME=ai_api
IMAGE_TAG=latest
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Build the docker image
IMG_ID=cen/ubuntu-tf-gpu
nvidia-docker build -f docker/ubuntu-16.04_tf-1.8.0-gpu.dockerfile -t $IMG_ID .

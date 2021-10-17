IMG_ID=cen/ubuntu-tf-gpu
NV_GPU=1 nvidia-docker run \
-v /home/maruan/.ssh/authorized_keys_docker/cen:/root/.ssh/authorized_keys \
-v /media/storage/data:/data \
-v /media/storage/logs/cen:/logs \
-d -p 3300:22 --name cen --rm \
${IMG_ID}:latest

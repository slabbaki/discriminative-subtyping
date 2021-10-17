# Ubuntu 16.04
# SSH server + rsync
# TensorFlow 1.8.0 GPU

FROM nvidia/cuda:latest
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Environment
ENV CFLAGS="-O2"
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV DATA_PATH=/data

# Linux packages
RUN apt-get update --fix-missing
RUN apt-get install -y \
    software-properties-common \
    make cmake build-essential autoconf libtool openssh-server rsync \
    ca-certificates git grep sed dpkg curl wget bzip2 unzip \
    libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev \
    libbz2-dev libsqlite3-dev \
    mpich htop vim tmux

# CUDA stuff
RUN apt-get install -y --no-install-recommends \
    cuda-command-line-tools-9-0 \
    cuda-cublas-9-0 \
    cuda-cufft-9-0 \
    cuda-curand-9-0 \
    cuda-cusolver-9-0 \
    cuda-cusparse-9-0 \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng12-dev \
    libzmq3-dev

# Pyenv, Python 3, and Python packages
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    pyenv install 3.6.5 && pyenv global 3.6.5 && pyenv rehash
RUN pip install -U \
    pip \
    ipython \
    numpy scipy \
    tensorflow-gpu==1.8.0 \
    keras==2.1.6 \
    cloudpickle \
    scikit-image \
    requests \
    click

# Setup tmux
RUN git clone https://github.com/gpakosz/.tmux.git /root/.tmux && \
    ln -s -f /root/.tmux/.tmux.conf /root/.tmux.conf && \
    cp /root/.tmux/.tmux.conf.local /root/

# Setup SSHD
RUN echo "AuthorizedKeysFile /root/.ssh/authorized_keys" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    mkdir /var/run/sshd && mkdir /root/.ssh && chown -R root:root /root/.ssh

# Setup bashrc
RUN echo "export PYENV_ROOT=/root/.pyenv" >> /root/.bashrc && \
    echo "export PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH" >> /root/.bashrc && \
    echo "export DATA_PATH=/data" >> /root/.bashrc
RUN mkdir /code

CMD ["/usr/sbin/sshd", "-D", "-f", "/etc/ssh/sshd_config"]

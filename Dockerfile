# Based on Deepo

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
ENV GIT_CLONE="git clone --depth 10"
RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# ==================================================================
# tools
# ------------------------------------------------------------------

RUN DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        tmux \
        rsync \
        zsh \
        && \
    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install

# ==================================================================
# python
# ------------------------------------------------------------------

RUN DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.7 \
        python3.7-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.7 ~/get-pip.py && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm

# ==================================================================
# jupyter
# ------------------------------------------------------------------

RUN $PIP_INSTALL \
        jupyter

# ==================================================================
# pytorch
# ------------------------------------------------------------------

RUN $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        torch==1.7.1+cu101 \
        torchvision==0.8.2+cu101 \
        -f https://download.pytorch.org/whl/torch_stable.html

# ==================================================================
# TAkS
# ------------------------------------------------------------------

RUN $PIP_INSTALL \
        plotly \
        kaleido

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# user
# ------------------------------------------------------------------

RUN $GIT_CLONE git://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
RUN echo "export SHELL=$(which zsh)" >> ~/.zshrc
RUN echo "export PATH=\"$USER_HOME/.local/bin:\$PATH\"" >> ~/.zshrc
RUN git clone https://github.com/powerline/fonts.git --depth=1
RUN sh fonts/install.sh
RUN rm -rf fonts
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="agnoster"/g' ~/.zshrc
RUN echo "cd /workspace" >> ~/.zshrc
RUN echo "export PYTHONPATH=\"/opt/ASAP/bin":"\$PYTHONPATH\"" >> ~/.zshrc
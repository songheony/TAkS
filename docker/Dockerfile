# Based on Deepo

FROM rapidsai/rapidsai:cuda10.1-base-ubuntu18.04-py3.7
ENV LANG C.UTF-8
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="/opt/conda/envs/rapids/bin/python -m pip --no-cache-dir install --upgrade"
ENV GIT_CLONE="git clone --depth 10"
RUN apt-get update

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
        zsh

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

 RUN $PIP_INSTALL \
        tensorflow-cpu

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
        kaleido \
        tensorboard \
        gsutil

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
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.zshrc
RUN echo "conda activate rapids" >> ~/.zshrc
RUN echo "cd /workspace" >> ~/.zshrc
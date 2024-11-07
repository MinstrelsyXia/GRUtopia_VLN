# FROM nvcr.io/nvidia/isaac-sim:2023.1.1
FROM nvcr.io/nvidia/isaac-sim:4.0.0

RUN export https_proxy=https://xiaxinyuan:OE6gf5X1v0JkSjKDOoUsVZhCdBbf0mdwfWO2kvWSlKj9L0Jwcfb9ff7snMkk@blsc-proxy.pjlab.org.cn:13128

RUN export http_proxy=https://xiaxinyuan:OE6gf5X1v0JkSjKDOoUsVZhCdBbf0mdwfWO2kvWSlKj9L0Jwcfb9ff7snMkk@blsc-proxy.pjlab.org.cn:13128


RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
RUN echo "deb https://repo.huaweicloud.com/ubuntu jammy main restricted \
deb https://repo.huaweicloud.com/ubuntu jammy-updates main restricted \
deb https://repo.huaweicloud.com/ubuntu jammy universe \
deb https://repo.huaweicloud.com/ubuntu jammy-updates universe \
deb https://repo.huaweicloud.com/ubuntu jammy multiverse \
deb https://repo.huaweicloud.com/ubuntu jammy-updates multiverse \
deb https://repo.huaweicloud.com/ubuntu jammy-backports main restricted universe multiverse \
deb https://repo.huaweicloud.com/ubuntu jammy-security main restricted \
deb https://repo.huaweicloud.com/ubuntu jammy-security universe \
deb https://repo.huaweicloud.com/ubuntu jammy-security multiverse" >> /etc/apt/sources.list
#清华源 
# RUN echo "deb Index of /ubuntu/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror focal main restricted universe multiverse \
# deb Index of /ubuntu/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror focal-updates main restricted universe multiverse \
# deb Index of /ubuntu/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror focal-backports main restricted universe multiverse \
# deb Index of /ubuntu/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror focal-security main restricted universe multiverse" >> /etc/apt/sources.list
RUN apt update
RUN apt-get install -y --allow-downgrades perl-base=5.34.0-3ubuntu1 netbase
RUN apt install -y nginx vim git

COPY ./webui/fe/html /usr/share/nginx/html
RUN chmod -R 755 /usr/share/nginx/html

COPY ./webui/fe/isaac/index.html /isaac-sim/extscache/omni.services.streamclient.webrtc-1.3.8/web/index.html
COPY . /isaac-sim/GRUtopia
WORKDIR /isaac-sim/GRUtopia

RUN mv ./webui/fe/nginx/default /etc/nginx/sites-available/default


# for isaac-sim:2023.1.1 webrtc error (0x800B1000)
RUN sed  -i "s/\"omni.kit.livestream.native\"/#\"omni.kit.livestream.native\"/g" /isaac-sim/apps/omni.isaac.sim.python.kit
RUN sed  -i "s/\"omni.kit.streamsdk.plugins\"/#\"omni.kit.streamsdk.plugins\"/g" /isaac-sim/apps/omni.isaac.sim.python.kit
RUN bash -c "cd ../ && \
    ./python.sh -m venv .venv && source .venv/bin/activate && \
    chmod +x ./GRUtopia/requirements/docker_install_req.sh && \
    cp ./GRUtopia/requirements/docker_install_req.sh . && \
    bash ./docker_install_req.sh"

RUN bash -c "cd ../ && \
    chmod +x ./GRUtopia/webui_start.sh && \
    cp ./GRUtopia/webui_start.sh . && \
    sed 's/^\$python_exe/#\$python_exe/g' ./python.sh > python.env.init && \
    echo 'source /isaac-sim/.venv/bin/activate' >> /root/.bashrc && \
    echo ' . /isaac-sim/python.env.init' >> /root/.bashrc && \
    echo 'set +e' >> /root/.bashrc && \
    echo 'export MDL_SYSTEM_PATH=/isaac-sim/materials/' >> /root/.bashrc"

WORKDIR /isaac-sim

ENTRYPOINT ["/bin/bash"]
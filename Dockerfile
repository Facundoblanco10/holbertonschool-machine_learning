FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y python3.5 python3-pip python3-tk git && \
    pip3 install --upgrade pip==20.3.4 && \
    pip3 install numpy==1.15 && \
    pip3 install scipy==1.3 && \
    pip3 install pycodestyle==2.5 && \
    pip3 install tensorflow==1.12 && \
    pip3 install matplotlib==3.0 && \
    pip3 install Pillow

#!/bin/bash

UBUNTU_CODENAME=jammy
ROOTFS=/root/sandbox-rootfs

sudo mkdir -p $ROOTFS
sudo debootstrap \
  --variant=minbase \
  --arch=arm64 \
  --include=ca-certificates,wget,bzip2,xz-utils,bash,coreutils,findutils,grep,sed,gawk,libc6,libstdc++6,libgcc-s1,libgomp1,openssl,curl \
  $UBUNTU_CODENAME \
  $ROOTFS \
  http://ports.ubuntu.com/ubuntu-ports/


  # https://mirrors.tuna.tsinghua.edu.cn/ubuntu/

  # http://archive.ubuntu.com/ubuntu
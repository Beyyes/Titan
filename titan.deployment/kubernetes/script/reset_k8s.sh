#!/usr/bin/env bash

sudo systemctl stop kubelet
sudo systemctl stop docker
sudo rm -rf /var/lib/cni/
sudo rm -rf /var/lib/kubelet/*
sudo rm -rf /etc/cni/
sudo rm -rf /etc/kubernetes
sudo rm -rf /var/lib/etcd
sudo systemctl start kubelet
sudo systemctl start docker
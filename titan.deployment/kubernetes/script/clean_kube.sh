#!/usr/bin/env bash

sudo apt-get remove kubelet -y
sudo apt-get remove kubeadm -y
sudo apt-get remove kubectl -y
sudo rm -rf ~/.kube
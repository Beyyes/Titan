#!/bin/bash

printInfo(){
    echo "[info] $1"
}

# install docker
if docker -v > /dev/null 2>&1; then
    printInfo "Docker has been installed"
else
    printInfo "install docker..."
    ./install_docker.sh
    printInfo "done"
fi

swapoff --all
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
if curl --help > /dev/null 2>&1; then
    printInfo "curl has been installed"
else
    apt-get update && apt-get install -y apt-transport-https curl
fi

## add apt source for kubernetes

kube_source="http://apt.kubernetes.io/"
if ! grep -q "^deb .*$kube_source" /etc/apt/sources.list /etc/apt/sources.list.d/*; then
    printInfo "add kubernetes source to apt-get..."
    ./add_kube_source.sh
    apt-get update
    printInfo "add kubernetes source done."
else
    printInfo "source has already been added."
fi

# install kubeadm kubectl kubelet
if kubelet --version > /dev/null 2>&1; then
    printInfo "kubelet has been installed"
else
    printInfo "install kubelet..."
    apt-get install -y kubelet=1.10.1-00
fi

if kubectl help > /dev/null 2>&1; then
    printInfo "kubectl has been installed"
else
    printInfo "install kubectl..."
    apt-get install -y kubectl=1.10.1-00
fi

if kubeadm version > /dev/null 2>&1; then
    printInfo "kubeadm has been installed"
else
    printInfo "install kubeadm..."
    apt-get install -y kubeadm=1.10.1-00
fi



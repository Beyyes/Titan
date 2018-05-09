#!/bin/bash

printInfo(){
    echo "[info] $1"
}

./prepare_env.sh

# initialize master for kubernetes
printInfo "init kubernetes master..."
kubeadm init

# set config file of kubectl for dladmin
mkdir -p /home/dladmin/.kube
cp -i /etc/kubernetes/admin.conf /home/dladmin/.kube/config
sudo chown dladmin:dladmin /home/dladmin/.kube/config

# set config file for root
export KUBECONFIG=/etc/kubernetes/admin.conf

# install CNI
printInfo "install CNI(weave-net)"
sysctl net.bridge.bridge-nf-call-iptables=1
export kubever=$(kubectl version | base64 | tr -d '\n')
kubectl apply -f "https://cloud.weave.works/k8s/net?k8s-version=$kubever"

# taint master
kubectl taint nodes --all node-role.kubernetes.io/master-



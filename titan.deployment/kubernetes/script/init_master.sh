#!/bin/bash

printInfo(){
    echo "[info] $1"
}

username="dladmin"

./prepare_env.sh

# initialize master for kubernetes
printInfo "init kubernetes master..."
kubeadm init

# set config file of kubectl for dladmin
if [ -d "/home/$username/.kube" ]; then
    rm -rf "/home/$username/.kube"
fi
mkdir -p "/home/$username/.kube"
cp -i /etc/kubernetes/admin.conf "/home/$username/.kube/config"
sudo chown "$username:$username" "/home/$username/.kube/config"

# set config file for root
export KUBECONFIG=/etc/kubernetes/admin.conf

# install CNI
printInfo "install CNI(weave-net)"
sysctl net.bridge.bridge-nf-call-iptables=1
export kubever=$(kubectl version | base64 | tr -d '\n')
kubectl apply -f "https://cloud.weave.works/k8s/net?k8s-version=$kubever"

# taint master
kubectl taint nodes --all node-role.kubernetes.io/master-



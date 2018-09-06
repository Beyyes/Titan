#!/usr/bin/env bash

if [ ! -f /usr/local/bin/helm ]; then
    echo "install helm"
    curl -LO https://storage.googleapis.com/kubernetes-helm/helm-v2.8.2-linux-amd64.tar.gz
    tar -xzvf helm-v2.8.2-linux-amd64.tar.gz
    cp linux-amd64/helm /usr/local/bin
    rm helm-v2.8.2-linux-amd64.tar.gz
    rm -rf linux-amd64
    echo "install helm done"
fi

if [ ! -n "$(kubectl get sa -n kube-system | awk '{print $1}' | grep ^tiller$)" ]; then
    echo "create service account tiller"
    kubectl -n kube-system create sa tiller
fi

if [ ! -n "$(kubectl get clusterrolebinding | awk '{print $1}' | grep ^tiller$)" ]; then
    echo "create cluster role binding tiller"
    kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller
fi

if [ ! -n "$(kubectl get ns | awk '{print $1}' | grep ^seldon$)" ]; then
    echo "create namespace seldon"
    kubectl create namespace seldon
fi


echo "delete all resouce in seldon namespaces"
kubectl delete svc --all -n seldon
kubectl delete deploy --all -n seldon
kubectl delete job --all -n seldon
kubectl delete ds --all -n seldon
kubectl delete po --all -n seldon

while [ -n "$(kubectl get po -n seldon)" ];
do
    echo "sleep 5 secs waitting for seldon pod terminate"
    sleep 5
done

if [ -n "$(kubectl get po -n kube-system | grep ^tiller-deploy)" ]; then
    echo "helm reset..."
    helm delete seldon-core-analytics --purge
    helm delete seldon-core --purge
    helm delete seldon-core-crd --purge
    helm reset --force
    echo "helm reset done"

    sleep 5
fi

echo "helm init tiller"
sudo helm init --service-account tiller

while [ ! "$(kubectl get po -n kube-system | grep ^tiller-deploy | awk '{print $3}')" == "Running" ];
do
    echo "sleep 5 secs waitting for tiller pod running"
    sleep 5
done

while [ -z "$(kubectl get crd | grep ^seldondeployments)" ];
do
    echo "install seldon-core-crd"
    sudo helm install seldon-core-crd --version 0.1.6 --name seldon-core-crd --repo https://storage.googleapis.com/seldon-charts
    echo "sleep 5 secs waitting for tiller pod ready"
    sleep 5
done

echo "install seldon-core"

sudo helm install seldon-core --version 0.1.6 --name seldon-core --repo https://storage.googleapis.com/seldon-charts --set cluster_manager.rbac=true --namespace seldon

echo "install seldon core done"

sleep 5

# echo "install seldon-core-analytics"
# deploy seldon portal
sudo kubectl create clusterrolebinding seldon-binding --clusterrole=cluster-admin --serviceaccount=seldon:default
echo "clusterrolebinding created\n"

sudo helm install seldon-core-analytics --version 0.2 --name seldon-core-analytics --set grafana_prom_admin_password=password --set persistence.enabled=false --repo https://storage.googleapis.com/seldon-charts --namespace seldon
echo "seldon-core-analytics installed\n"

# deploy seldon application
sudo kubectl apply -f sklearn_iris_deployment.json -n seldon
echo "apply sklearn_iris_deployment\n"

# get token
sudo apt-get install jq
#SERVER=$(kubectl get  svc/seldon-apiserver -n seldon -o jsonpath='{.spec.clusterIP}')
#TOKEN=`curl -s -H "Accept: application/json" oauth-key:oauth-secret@$SERVER:8080/oauth/token -d grant_type=client_credentials | jq -r '.access_token'`

# send request
#curl -s -H "Content-Type:application/json" -H "Accept: application/json" -H "Authorization: Bearer $TOKEN" $SERVER:8080/api/v0.1/predictions -d '{"meta":{},"data":{"names":["sepal lengt (cm)","sepal width (cm)", "petal length (cm)","petal width (cm)"],"ndarray":[[5.1,3.5,1.4,0.2]]}}'
#!/usr/bin/env bash

kubectl create -f dashboard-rbac.yaml
kubectl create -f dashboard-controller.yaml
kubectl create -f dashboard-service.yaml
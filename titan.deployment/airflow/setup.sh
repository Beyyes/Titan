#!/usr/bin/env bash

cd ../../titan.workflow
sudo apt-get install python-setuptools -y
curl -LO https://files.pythonhosted.org/packages/ae/e8/2340d46ecadb1692a1e455f13f75e596d4eab3d11a57446f08259dee8f02/pip-10.0.1.tar.gz
tar -xzvf pip-10.0.1.tar.gz
cd pip-10.0.1
sudo python setup.py install
pip install setuptools --user --upgrade
sudo apt-get -y install python-software-properties
sudo apt-get -y install software-properties-common
sudo apt-get -y install gcc make build-essential libssl-dev libffi-dev python-dev

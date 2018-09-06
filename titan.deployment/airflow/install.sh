#!/usr/bin/env bash

# prepare python environment
# exe dir : titan.workflow
sudo apt-get install python-setuptools -y
curl -LO https://files.pythonhosted.org/packages/ae/e8/2340d46ecadb1692a1e455f13f75e596d4eab3d11a57446f08259dee8f02/pip-10.0.1.tar.gz
tar -xzvf pip-10.0.1.tar.gz
cd pip-10.0.1
sudo python setup.py install
sudo pip install docutils
sudo pip install setuptools --user --upgrade


sudo apt-get -y install python-software-properties && \
sudo apt-get -y install software-properties-common && \
sudo apt-get -y install gcc make build-essential libssl-dev libffi-dev python-dev


# exe dir : titan.workflow
cd titan.workflow
sudo python setup.py install
# need this command to create airflow home
airflow

cd ~/airflow
# is the pip execution position right?
pip install kubernetes
sudo apt-get update
sudo apt-get install mysql-server
service mysql restart
sudo apt-get install libmysqlclient-dev
pip install mysqlclient

# sql_alchemy_conn = mysql://root:Pass_word@localhost/airflow
# executor = LocalExecutor

sudo mkdir -p /usr/lib/systemd/system
sudo cp airflow-webserver.service /usr/lib/systemd/system
sudo cp airflow-scheduler.service /usr/lib/systemd/system
sudo systemctl start airflow-webserver
sudo systemctl start airflow-scheduler

#cd ~/airflow
#airflow initdb
#airflow webserver -p 18880
#airflow scheduler
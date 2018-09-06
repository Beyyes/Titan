#!/usr/bin/env bash

sudo service mysql restart
sudo apt-get install libmysqlclient-dev -y
sudo pip install mysqlclient
cd ~/airflow
sudo mkdir -p /usr/lib/systemd/system
sudo cp airflow/airflow-webserver.service /usr/lib/systemd/system
sudo cp airflow/airflow-scheduler.service /usr/lib/systemd/system
sudo systemctl start airflow-webserver
sudo systemctl start airflow-scheduler

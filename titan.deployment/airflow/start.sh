#!/usr/bin/env bash

sudo service mysql restart
sudo apt-get install libmysqlclient-dev -y
sudo pip install mysqlclient
sudo mkdir -p /usr/lib/systemd/system
sudo cp airflow-webserver.service /usr/lib/systemd/system
sudo cp airflow-scheduler.service /usr/lib/systemd/system
sudo systemctl start airflow-webserver
sudo systemctl start airflow-scheduler
cd ~/airflow
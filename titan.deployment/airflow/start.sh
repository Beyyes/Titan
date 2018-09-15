#!/usr/bin/env bash

sudo chmod u+w /etc/mysql/mysql.conf.d/mysqld.cnf
sudo echo 'explicit_defaults_for_timestamp = true' >> /etc/mysql/mysql.conf.d/mysqld.cnf
sudo chmod u-w /etc/mysql/mysql.conf.d/mysqld.cnf

sudo service mysql restart
sudo apt-get install libmysqlclient-dev -y
sudo pip install mysqlclient

mysql -u root -p
show databases
create database airflow

sudo mkdir -p /usr/lib/systemd/system
sudo cp airflow-webserver.service /usr/lib/systemd/system
sudo cp airflow-scheduler.service /usr/lib/systemd/system
sudo systemctl start airflow-webserver
sudo systemctl start airflow-scheduler
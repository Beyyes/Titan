#!/bin/bash

apt-get install mysql-server mysql-client libmysqlclient-dev
pip install mysqlclient airflow[mysql,crypto,password]
airflow initdb
airflow webserver -p 18880
airflow scheduler
#!/bin/bash


airflow initdb
airflow webserver -p 10100
airflow scheduler
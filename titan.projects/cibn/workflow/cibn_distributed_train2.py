# uncompyle6 version 3.1.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.12 (default, Dec  4 2017, 14:50:18) 
# [GCC 5.4.0 20160609]
# Embedded file name: /home/qiozou/airflow/dags/example_cibn.py
# Compiled at: 2018-04-04 08:55:35
from __future__ import print_function
import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.titan_operator import TitanBootstrapOperator, TitanEndOperator, TitanTrainOperator,\
TitanBuildModelOperator, TitanDownloadModelOperator, TitanDeployModelOperator, TitanHelper
import sys, os, subprocess as sp, uuid
from time import sleep
import logging
import jinja2
import shutil
import glob
import requests as req
import json
import workflow_util as wu
import os

project_name = os.path.splitext(os.path.basename(__file__))[0] 

#HDFS Parameter
hdfs_host = '10.190.190.185'
hdfs_port = '9000'

#Offline Job Parameter
jar_hdfs_path = '/cibn/jar/'
jar_local_path = '/home/dladmin/cibn/offline'
spark_offline_cmd = 'HADOOP_USER_NAME=qiozou /usr/local/spark/bin/spark-submit --class "DataPreprocess" --master yarn --deploy-mode cluster {} {} {} --driver-memory 3g --executor-memory 3g --executor-cores 4 --num-executors 4'
offline_input = 'hdfs://' + hdfs_host + ':' + hdfs_port + '/offline/input/' + project_name + '/Large/'
offline_output = 'hdfs://' + hdfs_host + ':' + hdfs_port + '/offline/output/' + project_name + '/Large/'
offline_dir = '/offline/output/' + project_name + '/Large/'
log = logging.getLogger(__name__)

args = {'owner': 'airflow', 
   'start_date': airflow.utils.dates.days_ago(2), 
   'provide_context': True}

dag = DAG(project_name, schedule_interval='@once', default_args=args)

def on_spark(**kwargs):
    download_offline_binary()
    jar_files = os.path.join(jar_local_path, 'data-preprocess_2.11-1.0.jar')
    cmd = spark_offline_cmd.format(jar_files, offline_input, offline_output) 
    sp.check_call(cmd, shell=True)
    shutil.rmtree(jar_local_path)

def download_offline_binary(host = hdfs_host):
    
    shutil.rmtree(jar_local_path)
    jar_files = os.path.join(jar_hdfs_path, '*')
    mkdir(jar_local_path)
    wu.copy_hdfs_to_local(jar_files, jar_local_path)
    return jar_local_path 

train_code_dir = "hdfs://10.190.190.185:9000/path/code/cibn/code"

def get_train_conf(**kwargs):
    train_file_path = 'hdfs://10.190.190.185:9000/path/test/train_data.ffm'
    train_cmd = 'pip install -r requirements.txt && python train_dist.py --batch_size=500 --train_file_path=%s --train_output_dir={train_output_dir} --show_step=10 --num_epochs=1' % train_file_path 

    train_conf = {'train_task_num': 6,
        'train_memory_mb': 2048,
        'train_job_name': 'tensorflow-cibn-train-dist',
        'train_cpu_num': 2,
        'train_code_dir': train_code_dir,
        'train_cmd': train_cmd
    }
    return train_conf

def get_eval_conf(**kwargs):
    train_output_dir = TitanHelper.get_xcom_value(kwargs['ti'], key='train_output_dir', task_ids='op_train_model')
    eval_file_path = "hdfs://10.190.190.185:9000/path/test/train_data_val.ffm" 
    train_cmd = 'pip install -r requirements.txt && python eval.py --batch_size=500 --eval_file_path=%s --train_output_dir=%s' % (eval_file_path, train_output_dir)

    train_conf = {'train_task_num': 1,
        'train_memory_mb': 2048,
        'train_cpu_num': 2,
        'train_job_name': 'tensorflow-cibn-eval-dist',
        'train_code_dir': train_code_dir,
        'train_dist': False,
        'train_cmd': train_cmd
    }
    return train_conf

def get_download_conf(**kwargs):
    train_output_dir = TitanHelper.get_xcom_value(kwargs['ti'], key='train_output_dir', task_ids='op_train_model')
    download_conf = {
        'train_output_dir': train_output_dir 
    }
    return download_conf

docker_repo = 'deeplearningrepo'
docker_username = 'deeplearningrepo'
docker_password = 'asdlkj!12345'

model_name = 'CibnPredictor' 
model_version = '0.7'
model_memory = 200

deployment_id = 'cibn'
replicas = 1
client_id = 'e2f82c79-6984-4cc1-b517-ff66d5f06a80'
client_secret = 'oauth_secret'

def get_build_conf(**kwargs):
    download_working_dir = TitanHelper.get_xcom_value(kwargs['ti'], key='working_dir', task_ids='op_download_train_output_model')
    build_conf = {
        'download_working_dir': download_working_dir,
        'train_code_dir': train_code_dir,
        'model_name': model_name, 
        'model_version': model_version,
        'docker_repo': docker_repo,
        'docker_username': docker_username,
        'docker_password': docker_password 
    }
    return build_conf

def get_deploy_conf(**kwargs):
    deploy_conf = {
        'deployment_id': deployment_id,
        'predictor_id': deployment_id,
        'client_id': client_id,
        'client_secret': client_secret,
        'replicas': replicas,
        'model_name': model_name,
        'model_version': model_version,
        'model_memory': model_memory,
        'docker_repo': docker_repo 
    }
    return deploy_conf

titan_username = "admin"
titan_password = "password"
 
op_bootstrap = TitanBootstrapOperator(dag=dag)
#op_spark = PythonOperator(task_id='op_spark', dag=dag, python_callable=on_spark)
op_train_model = TitanTrainOperator(task_id='op_train_model',titan_username=titan_username, titan_password=titan_password, train_conf=get_train_conf, dag=dag)

op_eval_model = TitanTrainOperator(task_id='op_eval_model', titan_username=titan_username, titan_password=titan_password, train_conf=get_eval_conf, dag=dag)

op_download_train_output_model = TitanDownloadModelOperator(task_id='op_download_train_output_model', download_conf=get_download_conf, dag=dag)

op_build_serving_model = TitanBuildModelOperator(task_id='op_build_serving_model', build_conf=get_build_conf, dag=dag)

op_deploy_serving_model = TitanDeployModelOperator(task_id='op_deploy_serving_model', deploy_conf=get_deploy_conf, dag=dag)

op_end = TitanEndOperator(dag=dag)

#op_bootstrap >> op_spark
#op_spark >> op_train_model

op_bootstrap >> op_train_model
op_train_model >> op_eval_model
op_eval_model >> op_download_train_output_model
op_download_train_output_model >> op_build_serving_model
op_build_serving_model >> op_deploy_serving_model
op_deploy_serving_model >> op_end

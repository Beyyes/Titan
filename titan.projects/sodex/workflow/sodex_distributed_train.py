# uncompyle6 version 3.1.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.12 (default, Dec  4 2017, 14:50:18) 
# [GCC 5.4.0 20160609]
# Embedded file name: /home/neilbao/airflow/dags/example_cibn.py
# Compiled at: 2018-04-04 08:55:35


import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import logging
import workflow_util as wu
import os

#project_name = 'sodex_train3'
project_name = os.path.splitext(os.path.basename(__file__))[0]

portal_user_name = "adminadmin"
portal_user_password = "adminadmin"

#Yarn train job parameter
train_code_dir = "/path/code/sodex/code"
train_job_name = "tensorflow-hdfs-dist-sodex"
train_task_num = 2
train_cpu_num = 4
train_gpu_num = 0
train_memory_mb = 2048
train_cmd = "pip install -r requirements.txt && python darnn_dist.py --total_episodes=5 --train_output_dir=%s"
 
# Docker Image - transparent to user
#docker_repo = 'deeplearningrepo'
#docker_username = 'deeplearningrepo'
#docker_password = 'asdlkj!12345'

log = logging.getLogger(__name__)

args = {'owner': 'airflow', 
   'start_date': airflow.utils.dates.days_ago(2), 
   'provide_context': True}

dag = DAG(project_name, schedule_interval='@once', default_args=args)

def on_bootstrap(**kwargs):
    tracking_id = wu.get_uuid() 
    wu.put_xcom_value(kwargs['ti'], tracking_id=tracking_id)

def on_end(**kwargs):
    app_tracking_dir = wu.get_app_tracking_dir(**kwargs)
    log.info("rm tmp dir %s" % app_tracking_dir)
    #wu.rm_tmp_dir(app_tracking_dir)

def on_train_model_yarn(**kwargs):
    """
    1. create job config - json file
    2. submit job config
    3. query job status 
    4. fail the job if status == 'FAILED'
    """
    train_job_guid = wu.append_uuid(train_job_name)
    train_output_dir = wu.get_train_output_dir(portal_user_name, train_job_guid)
    
    wu.put_xcom_value(kwargs['ti'], train_output_dir=train_output_dir)

    train_cmd2 = train_cmd % (train_output_dir)
    job_config = wu.create_dist_tensorflow_job_config(train_job_guid, train_code_dir, train_task_num, train_memory_mb, train_cpu_num, train_gpu_num, train_cmd2)
    log.info("job config %s" % job_config)
    wu.submit_job(portal_user_name, portal_user_password, job_config)
    wu.poll_job(train_job_guid)
 
op_bootstrap = PythonOperator(task_id='op_bootstrap', dag=dag, python_callable=on_bootstrap)
op_train_model = PythonOperator(task_id='op_train_model', dag=dag, python_callable=on_train_model_yarn)
op_end = PythonOperator(task_id='op_end', dag=dag, python_callable=on_end)

op_bootstrap >> op_train_model
op_train_model >> op_end

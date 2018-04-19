# uncompyle6 version 3.1.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.12 (default, Dec  4 2017, 14:50:18) 
# [GCC 5.4.0 20160609]
# Embedded file name: /home/neilbao/airflow/dags/example_cibn.py
# Compiled at: 2018-04-04 08:55:35
from __future__ import print_function
import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import sys, os, subprocess as sp, uuid
from time import sleep
import logging
import jinja2
import shutil
import glob

debug_skip_build_serving_model = False
debug_skip_deploy_serving_model = False

#Project Name Parameter
project_name = 'CibnPredictor'

#HDFS Parameter
hdfs_host = '10.190.148.73'
hdfs_port = '9000'

#Offline Job Parameter
jar_hdfs_path = '/cibn/jar/'
jar_local_path = '~/cibn/offline'
spark_offline_cmd = 'HADOOP_USER_NAME=qiozou /usr/local/spark/bin/spark-submit --class "DataPreprocess" --master yarn --deploy-mode cluster {} {} {} --driver-memory 3g --executor-memory 3g --executor-cores 4 --num-executors 4'
offline_input = 'hdfs://' + hdfs_host + ':' + hdfs_port + '/offline/input/' + project_name + '/Large/'
offline_output = 'hdfs://' + hdfs_host + ':' + hdfs_port + '/offline/output/' + project_name + '/Large/'
offline_dir = '/offline/output/' + project_name + '/Large/'

#Model Parameter
model_hdfs_path = '/models/' + project_name + '/0.1'
model_version = '0.2'
model_name = project_name 
model_mount = '/model'
model_train_script = 'train.py'
model_train_cmd = "docker run -v {}:/model -it deeplearningrepo/tensorflow:1.4.1 python /model/{}"
#M
model_memory = 200

# Docker Image - transparent to user
docker_repo = 'deeplearningrepo'
docker_username = 'deeplearningrepo'
docker_password = 'asdlkj!12345'

hdfs_copy_cmd = 'hdfs dfs -copyToLocal hdfs://{}:{}/{} {}'

docker_image_build_cmd = 'docker build --force-rm=true -t {} {}'
docker_image_push_cmd = 'docker push {}'
docker_login_cmd = 'docker login -u {} -p {}'
docker_image_check_cmd = 'docker images -q {}'


cibn_model_deploy_template_path = '/home/neilbao/airflow/templates/model_deploy.json.template'
cibn_model_wrap_cmd = 'docker run -v {}:/model deeplearningrepo/core-python-wrapper:0.7 /model {} {} {} --force'
cibn_model_deploy_cmd = 'kubectl apply -f {} -n seldon'

log = logging.getLogger(__name__)

args = {'owner': 'airflow', 
   'start_date': airflow.utils.dates.days_ago(2), 
   'provide_context': True}

dag = DAG(project_name, schedule_interval='@once', default_args=args)

def get_app_tracking_dir(**kwargs):
    ti = kwargs['ti']
    application_id = ti.xcom_pull(key='application_id', task_ids='op_bootstrap')
    tracking_id = ti.xcom_pull(key='tracking_id', task_ids='op_bootstrap')
    tmp_path = ('/tmp/{}/{}').format(application_id, tracking_id)
    return tmp_path

def get_train_data_hdfs_dir(**kwargs):
    ti = kwargs['ti']
    application_id = ti.xcom_pull(key='application_id', task_ids='op_bootstrap')
    tracking_id = ti.xcom_pull(key='tracking_id', task_ids='op_bootstrap')
    tmp_path = ('/path/{}/{}/train').format(application_id, tracking_id)
    return tmp_path


def mkdir(path):
    if os.path.exists(path): return
    sp.check_call(('mkdir -p {}').format(path), shell=True)

def copy_hdfs_to_local(hdfs_path, local_dst, host=hdfs_host, port=hdfs_port):
    cmd = hdfs_copy_cmd.format(host, port, hdfs_path, local_dst)
    sp.check_call(cmd, shell=True)
    return local_dst

def on_bootstrap(**kwargs):
    kwargs['ti'].xcom_push(key='application_id', value='cibn')
    tracking_id = str(uuid.uuid4())
    kwargs['ti'].xcom_push(key='tracking_id', value=tracking_id)


def on_spark(**kwargs):
	mkdir(jar_local_path)
    ti = kwargs['ti']
    application_id = ti.xcom_pull(key='application_id', task_ids='op_bootstrap')
    tracking_id = ti.xcom_pull(key='tracking_id', task_ids='op_bootstrap')
    download_offline_binary()
    jar_files = os.path.join(jar_local_path, 'data-preprocess_2.11-1.0.jar')
    cmd = spark_offline_cmd.format(jar_files, offline_input, offline_output) 
    sp.check_call(cmd, shell=True)
    shutil.rmtree(jar_local_path)
    

def download_offline_binary(host = hdfs_host):
    
    shutil.rmtree(jar_local_path)
    jar_files = os.path.join(jar_hdfs_path, '*')
    mkdir(jar_local_path)
    copy_hdfs_to_local(jar_files, jar_local_path, host)
    return jar_local_path 

def download_model_profile(**kwargs):

    app_tracking_dir = get_app_tracking_dir(**kwargs)
    mkdir(app_tracking_dir)
    model_files = os.path.join(model_hdfs_path, '*')
    copy_hdfs_to_local(model_files, app_tracking_dir)
    return app_tracking_dir

def download_train_data(**kwargs):

    app_tracking_dir = get_app_tracking_dir(**kwargs)
    mkdir(app_tracking_dir)
    #train_data_dir = get_train_data_hdfs_dir(**kwargs)
    #hard code for test now 
    train_data_dir = offline_dir 
    train_data_files = os.path.join(train_data_dir, '*')
    
    local_train_data_tmp_dir = os.path.join(app_tracking_dir, 'tmp')
    mkdir(local_train_data_tmp_dir)
    copy_hdfs_to_local(train_data_files, local_train_data_tmp_dir)
    
    traindata_dir = os.path.join(local_train_data_tmp_dir, 'train_data')
    valdata_dir = os.path.join(local_train_data_tmp_dir, 'val_data') 

    traindata_files = glob.glob(traindata_dir + '/*.csv')
    valdata_files = glob.glob(valdata_dir + '/*.csv')

    
    local_train_data_dir = os.path.join(app_tracking_dir, 'data')
    mkdir(local_train_data_dir)

    new_train_data_file = os.path.join(local_train_data_dir, 'train_data.ffm')
    shutil.copy(traindata_files[0], new_train_data_file)
    
    new_val_data_file = os.path.join(local_train_data_dir, 'train_data_val.ffm')
    shutil.copy(traindata_files[0], new_val_data_file)
    
    return local_train_data_dir


def on_train_model(**kwargs):
    
    train_data_dir = download_train_data(**kwargs)
    app_tracking_dir = download_model_profile(**kwargs)

    cmd = model_train_cmd.format(app_tracking_dir, model_train_script) 
    sp.check_call(cmd, shell=True)
    
    shutil.rmtree(train_data_dir)
 
def on_build_serving_model(**kwargs):

    if debug_skip_build_serving_model:return

    model_image = ('{}/{}:{}').format(docker_repo, model_name.lower(), model_version)
    #app_tracking_dir = download_model_profile(**kwargs)
    app_tracking_dir = get_app_tracking_dir(**kwargs)

    cmd = cibn_model_wrap_cmd.format(app_tracking_dir, model_name, model_version, docker_repo)
    sp.check_call(cmd, shell=True)

    build_dir = os.path.join(app_tracking_dir, 'build')
    cmd = docker_image_build_cmd.format(model_image, '.')
    sp.check_call(cmd, cwd=build_dir, shell=True)

    cmd = docker_login_cmd.format(docker_username, docker_password)
    sp.check_call(cmd, shell=True)

    cmd = docker_image_push_cmd.format(model_image)
    sp.check_call(cmd, cwd=build_dir, shell=True)


def on_deploy_serving_model(**kwargs):
    app_tracking_dir = get_app_tracking_dir(**kwargs)
    mkdir(app_tracking_dir)
    deploy_file_path = os.path.join(app_tracking_dir, 'deploy.json')
    model_image = ('{}/{}:{}').format(docker_repo, model_name.lower(), model_version)

    with open(cibn_model_deploy_template_path, 'r') as f:
        template_data = f.read()
        content = jinja2.Template(template_data).render(model_name=model_name.lower(), model_version=model_version.lower(), model_image=model_image, model_memory=model_memory)
    
    with open(deploy_file_path, 'w') as f:
        f.write(content)

    log.info(deploy_file_path)
    cmd = cibn_model_deploy_cmd.format(deploy_file_path)

    if debug_skip_deploy_serving_model: return

    sp.check_call(cmd, shell=True)

def foo(**kwargs):
    log.info('foo')


op_bootstrap = PythonOperator(task_id='op_bootstrap', dag=dag, python_callable=on_bootstrap)
op_spark = PythonOperator(task_id='op_spark', dag=dag, python_callable=on_spark)
op_train_model = PythonOperator(task_id='op_train_model', dag=dag, python_callable=on_train_model)
op_build_serving_model = PythonOperator(task_id='op_build_serving_model', dag=dag, python_callable=on_build_serving_model)
op_deploy_serving_model = PythonOperator(task_id='op_deploy_serving_model', dag=dag, python_callable=on_deploy_serving_model)
op_bootstrap >> op_spark
op_spark >> op_train_model
op_train_model >> op_build_serving_model
op_build_serving_model >> op_deploy_serving_model

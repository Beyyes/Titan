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

# Project Name Parameter
# project_name = 'cibn_distributed_train'
project_name = os.path.splitext(os.path.basename(__file__))[0]

# HDFS Parameter
hdfs_host = '10.190.190.185'
hdfs_port = '9000'

# Offline Job Parameter
jar_hdfs_path = '/cibn/jar/'
jar_local_path = '/home/dladmin/cibn/offline'
spark_offline_cmd = 'HADOOP_USER_NAME=qiozou /usr/local/spark/bin/spark-submit --class "DataPreprocess" --master yarn --deploy-mode cluster {} {} {} --driver-memory 3g --executor-memory 3g --executor-cores 4 --num-executors 4'
offline_input = 'hdfs://' + hdfs_host + ':' + hdfs_port + '/offline/input/' + project_name + '/Large/'
offline_output = 'hdfs://' + hdfs_host + ':' + hdfs_port + '/offline/output/' + project_name + '/Large/'
offline_dir = '/offline/output/' + project_name + '/Large/'

# Model Parameter
model_version = '0.7'
model_name = 'CibnPredictor'

# M
model_memory = 200

# Rest Server Job Submit Parameter
portal_user_name = "admin"
portal_user_password = "password"

# Yarn train job parameter
train_code_dir = 'hdfs://' + hdfs_host + ':' + hdfs_port + '/path/code/cibn/code'
train_job_name = "tensorflow-hdfs-dist-cibn"
train_task_num = 6
train_cpu_num = 4
train_gpu_num = 0
train_memory_mb = 2048
train_cmd = "pip install -r requirements.txt && python train_dist.py \
    --batch_size=500 --train_file_path=%s --train_output_dir=%s --show_step=10 --num_epochs=1"

train_file_path = 'hdfs://' + hdfs_host + ':' + hdfs_port + "/path/test/train_data.ffm"
eval_file_path = 'hdfs://' + hdfs_host + ':' + hdfs_port + "/path/test/train_data_val.ffm"
eval_job_name = "tensorflow-hdfs-cibn-eval"
eval_cpu_num = 4
eval_gpu_num = 0
eval_memory_mb = 2048
eval_cmd = "pip install -r requirements.txt && python eval.py \
    --batch_size=500 --eval_file_path=%s --train_output_dir=%s"

# Docker Image - transparent to user
docker_repo = 'deeplearningrepo'
docker_username = 'deeplearningrepo'
docker_password = 'asdlkj!12345'

# docker_repo = 'neilbao'
# docker_username = 'neilbao'
# docker_password = 'dockerhub1986'

log = logging.getLogger(__name__)

args = {'owner': 'airflow',
        'start_date': airflow.utils.dates.days_ago(2),
        'provide_context': True}

dag = DAG(project_name, schedule_interval='@once', default_args=args)


def on_bootstrap(**kwargs):
    tracking_id = wu.get_uuid()
    wu.put_xcom_value(kwargs['ti'], tracking_id=tracking_id)


def on_spark(**kwargs):
    download_offline_binary()
    jar_files = os.path.join(jar_local_path, 'data-preprocess_2.11-1.0.jar')
    cmd = spark_offline_cmd.format(jar_files, offline_input, offline_output)
    sp.check_call(cmd, shell=True)
    shutil.rmtree(jar_local_path)


def download_offline_binary(host=hdfs_host):
    shutil.rmtree(jar_local_path)
    jar_files = os.path.join(jar_hdfs_path, '*')
    mkdir(jar_local_path)
    wu.copy_hdfs_to_local(jar_files, jar_local_path)
    return jar_local_path


def on_train_model_yarn(**kwargs):
    """
    1. create job config - json file
    2. submit job config
    3. query job status
    4. fail the job if status == 'FAILED'
    """

    """
    hdfs_path = os.path.join(offline_dir, "/*.csv")
    hdfs_files = glob(hdfs_path)
    train_file_path = hdfs_files[0]
    """

    train_job_guid = wu.append_uuid(train_job_name)
    train_output_dir = wu.get_train_output_dir(portal_user_name, train_job_guid)
    wu.put_xcom_value(kwargs['ti'], train_output_dir=train_output_dir)
    train_cmd2 = train_cmd % (train_file_path, train_output_dir)
    job_config = wu.create_dist_tensorflow_job_config(train_job_guid, train_code_dir, train_task_num, train_memory_mb,
                                                      train_cpu_num, train_gpu_num, train_cmd2)
    log.info("job config %s" % job_config)
    wu.submit_job(portal_user_name, portal_user_password, job_config)
    wu.poll_job(train_job_guid)


def on_eval_model_yarn(**kwargs):
    """
    1. create job config - json file
    2. submit job config
    3. query job status
    4. fail the job if status == 'FAILED'
    """

    eval_job_guid = eval_job_name + '-' + wu.get_uuid()
    train_output_dir = wu.get_xcom_value(kwargs['ti'], 'train_output_dir', 'op_train_model')
    eval_cmd2 = eval_cmd % (eval_file_path, train_output_dir)
    job_config = wu.create_single_tensorflow_job_config(eval_job_guid, train_code_dir, 1, eval_memory_mb, eval_cpu_num,
                                                        eval_gpu_num, eval_cmd2)
    log.info("job config %s" % job_config)
    wu.submit_job(portal_user_name, portal_user_password, job_config)
    wu.poll_job(eval_job_guid)


def on_download_train_output_model(**kwargs):
    working_dir = wu.make_task_working_dir(**kwargs)

    train_output_dir = wu.get_xcom_value(kwargs['ti'], 'train_output_dir', 'op_train_model')
    final_eval_model_file_hdfs = os.path.join(train_output_dir, "final_eval_model.txt")
    wu.copy_hdfs_to_local(final_eval_model_file_hdfs, working_dir)

    final_eval_model_file_local = os.path.join(working_dir, "final_eval_model.txt")
    with open(final_eval_model_file_local, 'r') as f:
        final_model_path = f.read().strip()
    log.info("final model path %s", final_model_path)

    checkpoint_path = os.path.join(working_dir, "checkpoint")
    with open(checkpoint_path, 'w') as f:
        final_model_name = final_model_path.split('/')[-1]
        f.write("model_checkpoint_path: \"%s\"" % final_model_name)

    train_output_files = final_model_path + '*'
    log.info("copy from %s to local %s" % (train_output_files, working_dir))
    wu.copy_hdfs_to_local(train_output_files, working_dir)


def on_build_serving_model(**kwargs):
    """
    1. give model image name
    2. feed (mount tracking dir to /model, model name, model version, docker repo) and run seldon python model wrapper docker command
    3. go build dir run docker image build command
    4. login dockerhub with username and password
    5. push docker image to dockerhub
    """
    working_dir = wu.make_task_working_dir(**kwargs)
    download_working_dir = wu.get_xcom_value(kwargs['ti'], 'working_dir', 'op_download_train_output_model')
    wu.cpdir(download_working_dir, working_dir)
    model_files = os.path.join(train_code_dir, '*')
    wu.copy_hdfs_to_local(model_files, working_dir)

    # generate build files
    cmd = 'docker run --rm -v {}:/model deeplearningrepo/core-python-wrapper:0.7 /model {} {} {} --force'.format(
        working_dir, model_name, model_version, docker_repo)
    sp.check_call(cmd, shell=True)

    build_dir = os.path.join(working_dir, 'build')
    model_image = ('{}/{}:{}').format(docker_repo, model_name.lower(), model_version)
    cmd = 'docker build --force-rm=true -t {} {}'.format(model_image, '.')
    sp.check_call(cmd, cwd=build_dir, shell=True)

    cmd = 'docker login -u {} -p {}'.format(docker_username, docker_password)
    sp.check_call(cmd, shell=True)

    cmd = 'docker push {}'.format(model_image)
    sp.check_call(cmd, cwd=build_dir, shell=True)


def on_deploy_serving_model(**kwargs):
    working_dir = wu.make_task_working_dir(**kwargs)
    deploy_file_path = os.path.join(working_dir, 'deploy.json')
    model_image = ('{}/{}:{}').format(docker_repo, model_name.lower(), model_version)

    cibn_model_deploy_template_path = '/home/dladmin/airflow/templates/model_deploy.json.template'

    with open(cibn_model_deploy_template_path, 'r') as f:
        template_data = f.read()
        content = jinja2.Template(template_data).render(model_name=model_name.lower(),
                                                        model_version=model_version.lower(), model_image=model_image,
                                                        model_memory=model_memory)

    with open(deploy_file_path, 'w') as f:
        f.write(content)

    log.info(deploy_file_path)
    cmd = 'kubectl apply -f {} -n seldon'.format(deploy_file_path)

    sp.check_call(cmd, shell=True)


def on_end(**kwargs):
    app_tracking_dir = wu.get_app_tracking_dir(**kwargs)
    log.info("rm tmp dir %s" % app_tracking_dir)
    # wu.rm_tmp_dir(app_tracking_dir)


def foo(**kwargs):
    log.info('foo')


op_bootstrap = PythonOperator(task_id='op_bootstrap', dag=dag, python_callable=on_bootstrap)
# op_spark = PythonOperator(task_id='op_spark', dag=dag, python_callable=on_spark)
op_train_model = PythonOperator(task_id='op_train_model', dag=dag, python_callable=on_train_model_yarn)
op_eval_model = PythonOperator(task_id='op_eval_model', dag=dag, python_callable=on_eval_model_yarn)
op_download_train_output_model = PythonOperator(task_id='op_download_train_output_model', dag=dag,
                                                python_callable=on_download_train_output_model)
op_build_serving_model = PythonOperator(task_id='op_build_serving_model', dag=dag,
                                        python_callable=on_build_serving_model)
op_deploy_serving_model = PythonOperator(task_id='op_deploy_serving_model', dag=dag,
                                         python_callable=on_deploy_serving_model)
op_end = PythonOperator(task_id='op_end', dag=dag, python_callable=on_end)
# op_bootstrap >> op_spark
# op_spark >> op_train_model

op_bootstrap >> op_train_model
op_train_model >> op_eval_model
op_eval_model >> op_download_train_output_model
op_download_train_output_model >> op_build_serving_model
op_build_serving_model >> op_deploy_serving_model
op_deploy_serving_model >> op_end

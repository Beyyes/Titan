# uncompyle6 version 3.1.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.12 (default, Dec  4 2017, 14:50:18) 
# [GCC 5.4.0 20160609]
# Embedded file name: /home/neilbao/airflow/dags/example_cibn.py
# Compiled at: 2018-04-04 08:55:35
from __future__ import print_function
import sys, os, subprocess as sp, uuid
from time import sleep
import logging
import jinja2
import shutil
import glob
import requests as req
import json


titan_env = os.environ.get('TITAN_ENV', 'prod')

if titan_env == "dev":
    platform_hdfs_uri = "hdfs://10.190.148.125:9000/"
    platform_job_rest_api_path = "http://10.190.148.125:9186/api/v1/jobs"
    platform_token_rest_api_path = "http://10.190.148.125:9186/api/v1/token"
    platform_tensorflow_runtime_image = "neilbao/pai.run.tensorflow"
elif titan_env == "prod":
    platform_hdfs_uri = "hdfs://10.190.190.185:9000/"
    platform_job_rest_api_path = "http://10.190.190.185:30186/api/v1/jobs"
    platform_token_rest_api_path = "http://10.190.190.185:30186/api/v1/token"
    platform_tensorflow_runtime_image = "neilbao/pai.run.tensorflow"
else:
    raise ValueError("bad titan env: " + titan_env)

log = logging.getLogger(__name__)


def get_hdfs_uri():
    return platform_hdfs_uri

#hack for root permission temporary
def rm_tmp_dir(tmp_dir, rm_dir=True):
    if not tmp_dir.startswith("/tmp/"): return
    cmd = "docker run --rm -v {}:/model ubuntu rm -rf /model".format(tmp_dir) 
    sp.call(cmd, shell=True) #be careful
    if rm_dir:
        cmd = "rmdir {}".format(tmp_dir)
        sp.check_call(cmd, shell=True)

def get_uuid():
    return str(uuid.uuid4())


def get_app_tracking_dir(**kwargs):
    ti = kwargs['ti']
    application_id = get_xcom_value(ti, key='application_id', task_ids='op_bootstrap')
    if application_id == None: application_id = str(ti.dag_id)
    tracking_id = get_xcom_value(ti, key='tracking_id', task_ids='op_bootstrap')
    if application_id == None or tracking_id == None: raise Exception('application id or tracking id is None')
    tmp_path = ('/tmp/{}/{}').format(application_id, tracking_id)
    return tmp_path

def make_task_working_dir(**kwargs):
    tracking_dir = get_app_tracking_dir(**kwargs)
    ti = kwargs['ti']
    working_dir = os.path.join(tracking_dir,  ti.task_id + '-' + get_uuid().split('-')[-1])
    log.info("task working dir %s", working_dir)
    mkdir(working_dir)
    put_xcom_value(ti, working_dir=working_dir)
    return working_dir

def mkdir(path):
    if os.path.exists(path): return
    sp.check_call(('mkdir -p {}').format(path), shell=True)

def cpdir(src, dest):
    files = os.path.join(src, '*')
    sp.check_call(('cp -rf {} {}').format(files, dest), shell=True)

def make_app_tracking_dir(**kwargs):
    tmp_path = get_app_tracking_dir(**kwargs)
    mkdir(tmp_path)
    return tmp_path
 
def copy_hdfs_to_local(hdfs_path, local_dst):
    hdfs_path = hdfs_path.strip()
    if hdfs_path.startswith('hdfs://'):
        cmd = 'hdfs dfs -copyToLocal -f {} {}'.format(hdfs_path, local_dst)
    else:
        cmd = 'hdfs dfs -copyToLocal -f {}{} {}'.format(platform_hdfs_uri, hdfs_path, local_dst)
    sp.check_call(cmd, shell=True)
    return local_dst

def glob_hdfs(hdfs_path):
    cmd = "hdfs dfs -ls {}".format(hdfs_path)
    r = sp.check_output(cmd, shell=True)
    r = r.strip('\n').split('\n')
    r = map(lambda x: 'hdfs://' + x.split('hdfs://')[-1],r)
    return r

def get_xcom_value(ti, key, task_ids):
    value = ti.xcom_pull(key=key, task_ids=task_ids)
    log.info("get xcom key:%s, task_ids: %s, value: %s" %(key, task_ids, value))
    return value 

def put_xcom_value(ti, **kwargs):
    for k,v in kwargs.items():
        ti.xcom_push(key=k, value=v)
        log.info("put xcom key:%s, value: %s" %(k, v))

def get_token(user_name, password):
    data = {"username":user_name, "password": password, "expiration": 60 * 60 * 24 }
    r = req.post(platform_token_rest_api_path, data=data)
    if r.status_code == 200:
        return r.json()["token"]
    err_msg = "user name [%s] get token failed: %s" % (user_name,r.json())
    raise Exception(err_msg)

def get_job(job_name):
    r = req.get(platform_job_rest_api_path + '/' + job_name)
    return r.json()

def append_uuid(job_name):
    job_uuid = job_name + "-" + get_uuid() 
    return job_uuid

def get_train_output_dir(user_name, job_name):
    train_output_dir = "%sOutput/%s/%s/" % (get_hdfs_uri(), user_name, job_name)
    return train_output_dir

def create_dist_tensorflow_job_config(job_name, code_dir, task_num, memory_mb, cpu_num, gpu_num, cmd):
    job_config = {}
    job_config["jobName"] = job_name
    job_config["image"] = platform_tensorflow_runtime_image
    job_config["codeDir"] = code_dir if code_dir.startswith('hdfs://') else get_hdfs_uri() + code_dir

    cd = code_dir.strip('/').split('/')[-1]

    task_roles = []
    ps_role = {}
    ps_role["name"] = "ps"
    ps_role["taskNumber"] = 1
    ps_role["cpuNumber"] = 2
    ps_role["memoryMB"] = 1024
    ps_role["gpuNumber"] = 0
    ps_role["command"] = "cd " + cd + " && " +  cmd + " --ps_hosts=$PAI_TASK_ROLE_ps_HOST_LIST --worker_hosts=$PAI_TASK_ROLE_worker_HOST_LIST --job_name=ps --task_index=$PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX"

    worker_role = {}
    worker_role["name"] = "worker"
    worker_role["taskNumber"] = task_num
    worker_role["cpuNumber"] = cpu_num
    worker_role["memoryMB"] = memory_mb
    worker_role["gpuNumber"] = gpu_num
    worker_role["command"] = "cd " + cd + " && " + cmd + " --ps_hosts=$PAI_TASK_ROLE_ps_HOST_LIST --worker_hosts=$PAI_TASK_ROLE_worker_HOST_LIST --job_name=worker --task_index=$PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX"


    task_roles.append(ps_role)
    task_roles.append(worker_role)

    job_config["taskRoles"] = task_roles
    job_config["killAllOnCompletedTaskNumber"] = task_num
    job_config["retryCount"] = 0
    return job_config

def create_single_tensorflow_job_config(job_name, code_dir, task_num, memory_mb, cpu_num, gpu_num, cmd):
    job_config = {}
    job_config["jobName"] = job_name
    job_config["image"] = platform_tensorflow_runtime_image
    job_config["codeDir"] = code_dir if code_dir.startswith('hdfs://') else get_hdfs_uri() + code_dir

    cd = code_dir.strip('/').split('/')[-1]

    task_roles = []
    worker_role = {}
    worker_role["name"] = "worker"
    worker_role["taskNumber"] = task_num
    worker_role["cpuNumber"] = cpu_num
    worker_role["memoryMB"] = memory_mb
    worker_role["gpuNumber"] = gpu_num
    worker_role["command"] = "cd " + cd + " && " + cmd


    task_roles.append(worker_role)

    job_config["taskRoles"] = task_roles
    return job_config

def submit_job(user_name, password, job_config):
    job_name = job_config["jobName"]
    data = json.dumps(job_config)
    token = get_token(user_name, password)
    log.info("job %s" % data)
    log.info("token %s" % token)
    r = req.put(platform_job_rest_api_path + '/' + job_name, data=data, headers={'Content-Type':'application/json', 'Authorization': 'Bearer %s' % token})
    log.info("res %s" % r.json())
    if r.status_code != 202:
        err_msg = "submit job %s failed, %s" % (job_name,r.json())
        raise Exception(err_msg)
    return job_name

def get_job_status(job):
    return job['jobStatus']['state']

def poll_job(job_name, interval_sec = 60):
    while True:
        job = get_job(job_name)
        status = get_job_status(job).lower()

        if status == "succeeded":
            log.info("job %s succeeded" % job_name)
            break
        if status == "failed" or status == "stopped":
            msg = "job %s stopped" % job_name
            raise Exception(msg)
        if status == "waiting":
            log.info("job %s is waiting" % job_name)
        else:
            log.info("job %s is running" % job_name)
        sleep(interval_sec)


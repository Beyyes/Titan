# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from builtins import str
import dill
import inspect
import os
import pickle
import subprocess as sp
import sys
import types
import uuid
import logging
import requests as req
import json
import shutil
import glob
import jinja2

from time import sleep
from airflow.exceptions import AirflowException
from airflow.models import BaseOperator, SkipMixin
from airflow.utils.decorators import apply_defaults
from airflow.utils.file import TemporaryDirectory

from textwrap import dedent

log = logging.getLogger(__name__)

class TitanHelper:

    @staticmethod 
    def get_uuid():
        return str(uuid.uuid4())

    @staticmethod
    def put_xcom_value(ti, **kwargs):
        for k,v in kwargs.items():
            ti.xcom_push(key=k, value=v)
            log.info("put xcom key:%s, value: %s" %(k, v))
   
    @staticmethod
    def get_xcom_value(ti, key, task_ids, check_none=True):
        value = ti.xcom_pull(key=key, task_ids=task_ids)
        if check_none and not value:
            raise AirflowException('value is None for key: `%s`, task_ids: `%s`' % (key, task_ids))
        log.info("get xcom key:%s, task_ids: %s, value: %s" %(key, task_ids, value))
        return value
     
    @staticmethod
    def append_uuid(job_name):
        job_uuid = job_name + "-" + TitanHelper.get_uuid()
        return job_uuid

    @staticmethod
    def get_app_tracking_dir(**kwargs):
        ti = kwargs['ti']
        application_id = str(ti.dag_id)
        tracking_id = TitanHelper.get_xcom_value(ti, key='tracking_id', task_ids='op_bootstrap')
        if application_id == None or tracking_id == None:
            raise AirflowException('application id or tracking id is None')
        tmp_path = ('/tmp/{}/{}').format(application_id, tracking_id)
        return tmp_path


    @staticmethod
    def make_task_working_dir(**kwargs):
        tracking_dir = TitanHelper.get_app_tracking_dir(**kwargs)
        ti = kwargs['ti']
        working_dir = os.path.join(tracking_dir,  ti.task_id + '-' + TitanHelper.get_uuid().split('-')[-1])
        log.info("task working dir %s", working_dir)
        TitanHelper.mkdir(working_dir)
        TitanHelper.put_xcom_value(ti, working_dir=working_dir)
        return working_dir


    #hack for root permission temporary
    @staticmethod
    def rm_tmp_dir(tmp_dir, rm_dir=True):
        if not tmp_dir.startswith("/tmp/"): return
        cmd = "docker run --rm -v {}:/model ubuntu rm -rf /model".format(tmp_dir)
        sp.call(cmd, shell=True) #be careful
        if rm_dir:
            cmd = "rmdir {}".format(tmp_dir)
            sp.check_call(cmd, shell=True)

    @staticmethod
    def mkdir(path):
        if os.path.exists(path): return
        sp.check_call(('mkdir -p {}').format(path), shell=True)

    @staticmethod
    def cpdir(src, dest):
        files = os.path.join(src, '*')
        sp.check_call(('cp -rf {} {}').format(files, dest), shell=True)

    @staticmethod
    def make_app_tracking_dir(**kwargs):
        tmp_path = TitanHelper.get_app_tracking_dir(**kwargs)
        TitanHelper.mkdir(tmp_path)
        return tmp_path

    @staticmethod
    def copy_hdfs_to_local(hdfs_path, local_dst):
        hdfs_path = hdfs_path.strip()
        if hdfs_path.startswith('hdfs://'):
            cmd = 'hdfs dfs -copyToLocal {} {}'.format(hdfs_path, local_dst)
        else:
            cmd = 'hdfs dfs -copyToLocal {}{} {}'.format(TitanHelper.get_hdfs_uri(), hdfs_path, local_dst)
        sp.check_call(cmd, shell=True)
        return local_dst

    @staticmethod
    def glob_hdfs(hdfs_path):
        if not hdfs_path.startswith('hdfs://'):
           hdfs_path = TitanHelper.get_hdfs_uri() + hdfs_path  
        cmd = "hdfs dfs -ls {}".format(hdfs_path)
        r = sp.check_output(cmd, shell=True)
        r = r.strip('\n').split('\n')
        r = map(lambda x: 'hdfs://' + x.split('hdfs://')[-1],r)
        return r

    @staticmethod
    def get_hdfs_uri():
        hdfs_uri = os.environ.get('TITAN_HDFS_URI')
        if not hdfs_uri:
            raise AirflowException('`TITAN_HDFS_URI` is not found in env') 
        return hdfs_uri

    @staticmethod
    def get_train_rest_api_root():
        api_root = os.environ.get('TITAN_TRAIN_REST_API_ROOT')
        if not api_root:
            raise AirflowException('`TITAN_TRAIN_REST_API_ROOT` is not found in env')
        return api_root 

    @staticmethod
    def get_train_runtime_image():
        img = os.environ.get('TITAN_TRAIN_RUNTIME_IMAGE')
        if not img:
            raise AirflowException('`TITAN_TRAIN_RUNTIME_IMAGE` is not found in env')
        return img 
    
    @staticmethod
    def create_dist_tensorflow_job_config(job_name, code_dir, task_num, memory_mb, cpu_num, gpu_num, cmd):
        job_config = {}
        job_config["jobName"] = job_name
        job_config["image"] = TitanHelper.get_train_runtime_image()
        job_config["codeDir"] = code_dir if code_dir.startswith('hdfs://') else TitanHelper.get_hdfs_uri() + code_dir

        cd = code_dir.strip('/').split('/')[-1]

        task_roles = []
        ps_role = {}
        ps_role["name"] = "ps"
        ps_role["taskNumber"] = 2
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

    @staticmethod
    def create_single_tensorflow_job_config(job_name, code_dir, task_num, memory_mb, cpu_num, gpu_num, cmd):
        job_config = {}
        job_config["jobName"] = job_name
        job_config["image"] = TitanHelper.get_train_runtime_image()
        job_config["codeDir"] = code_dir if code_dir.startswith('hdfs://') else TitanHelper.get_hdfs_uri() + code_dir

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

    @staticmethod
    def get_job_status(job):
        try:
            s = job['jobStatus']['state']
        except:
            log.error("error job res: %s" % job)
            return None
        return s


    @staticmethod
    def get_job(job_name):
        max_retries = 3
        retry_interval_secs = 5
        api_path = TitanHelper.get_train_rest_api_root() + 'jobs/' + job_name
        retries = 0
        while True:
            r = req.get(api_path)
            job = r.json()
            job_status = TitanHelper.get_job_status(job) 
            if not job_status:
                retries += 1
            else:
                return job
            if retries > max_retries:
                raise AirflowException('call api failed: %s, response: %s' % (api_path, job))
            else:
                sleep(retry_interval_secs)

    @staticmethod
    def get_token(username, password):
        data = {"username":username, "password": password, "expiration": 60 * 60 * 24 }
        api_path = TitanHelper.get_train_rest_api_root() + 'token'
        r = req.post(api_path, data=data)
        if r.status_code == 200:
            return r.json()["token"]
        err_msg = "user name [%s] get token failed: %s" % (username,r.json())
        raise AirflowException(err_msg)

    @staticmethod
    def submit_job(username, password, job_config):
        job_name = job_config["jobName"]
        data = json.dumps(job_config)
        token = TitanHelper.get_token(username, password)
        log.info("job %s" % data)
        log.info("token %s" % token)
        api_path = TitanHelper.get_train_rest_api_root() + 'jobs/' + job_name
        r = req.put(api_path, data=data, headers={'Content-Type':'application/json', 'Authorization': 'Bearer %s' % token})
        log.info("res %s" % r.json())
        if r.status_code != 202:
            err_msg = "submit job %s failed, %s" % (job_name,r.json())
            raise AirflowException(err_msg)
        return job_name

    @staticmethod
    def get_train_output_dir(username, job_name):
        train_output_dir = "%s/Output/%s/%s/" % (TitanHelper.get_hdfs_uri(), username, job_name)
        return train_output_dir

    @staticmethod
    def get_yarn_logs(job):
        logs = []
        if type(job) == dict:
            log = job.get('containerLog')
            if log != None: return log.encode('ascii')

            for it in job.items():
                log = TitanHelper.get_yarn_logs(it[1])
                if log != None:
                    if type(log) == str:
                        logs.append(log)
                    else:
                        logs.extend(log)
        elif type(job) == list:
            for it in job:
                log = TitanHelper.get_yarn_logs(it)
                if log != None:
                    if type(log) == str:
                        logs.append(log)
                    else:
                        logs.extend(log)
        else:
            return None
        return logs


    @staticmethod
    def poll_job(job_name, poll_logs=False, interval_sec = 60):
        while True:
            job = TitanHelper.get_job(job_name)
            status = TitanHelper.get_job_status(job).lower()

            if poll_logs:
                yarn_logs = TitanHelper.get_yarn_logs(job)
                if yarn_logs:
                    return yarn_logs
            if status == "succeeded":
                log.info("job %s succeeded" % job_name)
                break
            if status == "failed" or status == "stopped":
                msg = "job %s stopped" % job_name
                raise AirflowException(msg)
            if status == "waiting":
                log.info("job %s is waiting" % job_name)
            else:
                log.info("job %s is running" % job_name)

            sleep(interval_sec)

        return None

    @staticmethod
    def execute_callable(func, *args, **kwargs):
        if callable(func):
            return func(*args, **kwargs)
        else:
            return func 

class TitanBootstrapOperator(BaseOperator):
    """
    Executes a Python callable
    :param on_end_hook: A reference to an object that is callable
    :type on_end_hook: python callable
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function
    :type op_kwargs: dict
    :param op_args: a list of positional arguments that will get unpacked when
        calling your callable
    :type op_args: list

    """
    template_fields = ('templates_dict',)
    template_ext = tuple()
    ui_color = '#ffefeb'

    @apply_defaults
    def __init__(
            self,
            on_end_hook=None,
            templates_dict=None,
            op_args=None,
            op_kwargs=None,
            *args, **kwargs):
        super(TitanBootstrapOperator, self).__init__(task_id='op_bootstrap', *args, **kwargs)
        self.on_end_hook = on_end_hook
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.templates_dict = templates_dict

    def execute(self, context):
        self.log.info("context: %s" % context)
        context.update(self.op_kwargs)
        self.op_kwargs = context
        tracking_id = TitanHelper.get_uuid() 
        TitanHelper.put_xcom_value(self.op_kwargs['ti'], tracking_id=tracking_id) 

        return_value = self.execute_callable(self.on_end_hook) 
        self.log.info("Done. Returned value was: %s", return_value)
        return return_value

    def execute_callable(self, func):
        return TitanHelper.execute_callable(func, *self.op_args, **self.op_kwargs)

class TitanEndOperator(BaseOperator):
    """
    Executes a Python callable
    :param on_end_hook: A reference to an object that is callable
    :type on_end_hook: python callable
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function
    :type op_kwargs: dict
    :param op_args: a list of positional arguments that will get unpacked when
        calling your callable
    :type op_args: list

    """
    template_fields = ('templates_dict',)
    template_ext = tuple()
    ui_color = '#ffefeb'

    @apply_defaults
    def __init__(
            self,
            on_end_hook=None,
            templates_dict=None,
            op_args=None,
            op_kwargs=None,
            *args, **kwargs):
        super(TitanEndOperator, self).__init__(task_id='op_end', *args, **kwargs)
        self.on_end_hook = on_end_hook
        self.templates_dict = templates_dict
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.templates_dict = templates_dict

    def execute(self, context):
        self.log.info("context: %s" % context)
        context.update(self.op_kwargs)
        self.op_kwargs = context

        return_value = self.execute_callable(self.on_end_hook) 
        app_tracking_dir = TitanHelper.get_app_tracking_dir(**self.op_kwargs)  
        self.log.info('rm tmp dir %s' % app_tracking_dir)
        #TitanHelper.rm_tmp_dir(app_tracking_dir)
        return return_value

    def execute_callable(self, func):
        return TitanHelper.execute_callable(func, *self.op_args, **self.op_kwargs)
        
class TitanTrainOperator(BaseOperator):
    """
    Executes a Python callable
    :param python_callable: A reference to an object that is callable
    :type python_callable: python callable
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function
    :type op_kwargs: dict
    :param op_args: a list of positional arguments that will get unpacked when
        calling your callable
    :type op_args: list
    :param provide_context: if set to true, Airflow will pass a set of
        keyword arguments that can be used in your function. This set of
        kwargs correspond exactly to what you can use in your jinja
        templates. For this to work, you need to define `**kwargs` in your
        function header.
    :type provide_context: bool
    :param templates_dict: a dictionary where the values are templates that
        will get templated by the Airflow engine sometime between
        ``__init__`` and ``execute`` takes place and are made available
        in your callable's context after the template has been applied. (templated)
    :type templates_dict: dict of str
    :param templates_exts: a list of file extensions to resolve while
        processing templated fields, for examples ``['.sql', '.hql']``
    :type templates_exts: list(str)
    """
    template_fields = ('templates_dict',)
    template_ext = tuple()
    ui_color = '#ffefeb'

    @apply_defaults
    def __init__(
            self,
            titan_username,
            titan_password,
            train_conf=None,
            on_end_hook=None,
            op_args=None,
            op_kwargs=None,
            templates_dict=None,
            *args, **kwargs):
        super(TitanTrainOperator, self).__init__(*args, **kwargs)
        if not titan_username:
            raise AirflowException('`titan_username` is None')
        if not titan_password:
            raise AirflowException('`titan_password` is None')

        self.titan_username = titan_username
        self.titan_password = titan_password
        self.train_conf = train_conf
        self.on_end_hook = on_end_hook
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.templates_dict = templates_dict

    def execute(self, context):
        context.update(self.op_kwargs)
        context['templates_dict'] = self.templates_dict
        self.op_kwargs = context
        dag_run = self.op_kwargs.get('dag_run') 
        train_conf = self.execute_callable(self.train_conf)
        conf = dag_run.conf or train_conf

        if not type(conf) == dict:
            raise AirflowException('conf is not dict %s' % conf)

        self.log.info('conf: %s' % conf)

        train_task_num = conf.get('train_task_num', 2)
        train_memory_mb = conf.get('train_memory_mb', 2048)
        train_cpu_num = conf.get('train_cpu_num', 2)
        train_gpu_num = conf.get('train_gpu_num', 0)
        train_job_name = conf.get('train_job_name', 'train')
        train_code_dir = conf.get('train_code_dir')
        train_cmd = conf.get('train_cmd')
        train_dist = conf.get('train_dist', True)

        if train_code_dir == None or train_cmd == None:
            raise AirflowException("`train_code_dir` or `train_cmd` is None")

        train_job_guid = TitanHelper.append_uuid(train_job_name)

        if '{train_output_dir}' in train_cmd:
            train_output_dir = TitanHelper.get_train_output_dir(self.titan_username, train_job_guid)
            TitanHelper.put_xcom_value(self.op_kwargs['ti'], train_output_dir=train_output_dir)
            train_cmd = train_cmd.replace('{train_output_dir}', train_output_dir)

        if train_dist:
            job_config = TitanHelper.create_dist_tensorflow_job_config(train_job_guid, train_code_dir, train_task_num, train_memory_mb, train_cpu_num, train_gpu_num, train_cmd)
        else:
            train_task_num = 1
            job_config = TitanHelper.create_single_tensorflow_job_config(train_job_guid, train_code_dir, train_task_num, train_memory_mb, train_cpu_num, train_gpu_num, train_cmd)

        self.log.info("job config %s" % job_config)
        TitanHelper.submit_job(self.titan_username, self.titan_password, job_config)
        yarn_logs = TitanHelper.poll_job(train_job_guid, poll_logs=True)
        TitanHelper.put_xcom_value(self.op_kwargs['ti'], yarn_logs=yarn_logs)
        TitanHelper.poll_job(train_job_guid)
     
        return_value = self.execute_callable(self.on_end_hook)
        self.log.info("Done. Returned value was: %s", return_value)
        return return_value

    def execute_callable(self, func):
        return TitanHelper.execute_callable(func, *self.op_args, **self.op_kwargs)

class TitanDownloadModelOperator(BaseOperator):
    """
    Executes a Python callable
    :param python_callable: A reference to an object that is callable
    :type python_callable: python callable
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function
    :type op_kwargs: dict
    :param op_args: a list of positional arguments that will get unpacked when
        calling your callable
    :type op_args: list
    :param provide_context: if set to true, Airflow will pass a set of
        keyword arguments that can be used in your function. This set of
        kwargs correspond exactly to what you can use in your jinja
        templates. For this to work, you need to define `**kwargs` in your
        function header.
    :type provide_context: bool
    :param templates_dict: a dictionary where the values are templates that
        will get templated by the Airflow engine sometime between
        ``__init__`` and ``execute`` takes place and are made available
        in your callable's context after the template has been applied. (templated)
    :type templates_dict: dict of str
    :param templates_exts: a list of file extensions to resolve while
        processing templated fields, for examples ``['.sql', '.hql']``
    :type templates_exts: list(str)
    """
    template_fields = ('templates_dict',)
    template_ext = tuple()
    ui_color = '#ffefeb'

    @apply_defaults
    def __init__(
            self,
            download_conf,
            on_end_hook=None,
            op_args=None,
            op_kwargs=None,
            templates_dict=None,
            *args, **kwargs):
        super(TitanDownloadModelOperator, self).__init__(*args, **kwargs)
        self.download_conf = download_conf
        self.on_end_hook = on_end_hook
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.templates_dict = templates_dict

    def execute(self, context):
        context.update(self.op_kwargs)
        context['templates_dict'] = self.templates_dict
        self.op_kwargs = context
        conf = self.execute_callable(self.download_conf)

        if not type(conf) == dict:
            raise AirflowException('conf is not dict %s' % conf)

        self.log.info('conf: %s' % conf)

        working_dir = TitanHelper.make_task_working_dir(**self.op_kwargs)
        train_output_dir = conf.get('train_output_dir')
        final_eval_model_file = conf.get('final_eval_model_file', 'final_eval_model.txt')

        if not train_output_dir: raise AirflowException('`train_output_dir` is None')
        if not final_eval_model_file: raise AirflowException('`final_eval_model_file` is None')
       
        final_eval_model_file_hdfs = os.path.join(train_output_dir, final_eval_model_file) 
        TitanHelper.copy_hdfs_to_local(final_eval_model_file_hdfs, working_dir)
        final_eval_model_file_local = os.path.join(working_dir, final_eval_model_file)
        with open(final_eval_model_file_local, 'r') as f:
            final_model_path = f.read().strip()
        self.log.info("final model path %s", final_model_path)

        checkpoint_path = os.path.join(working_dir, "checkpoint")
        with open(checkpoint_path, 'w') as f:
            final_model_name = final_model_path.split('/')[-1]
            f.write("model_checkpoint_path: \"%s\"" % final_model_name)

        train_output_files = final_model_path + '*'
        self.log.info("copy from %s to local %s" % (train_output_files, working_dir))
        TitanHelper.copy_hdfs_to_local(train_output_files, working_dir)

        return_value = self.execute_callable(self.on_end_hook)
        self.log.info("Done. Returned value was: %s", return_value)
        return return_value

    def execute_callable(self, func):
        return TitanHelper.execute_callable(func, *self.op_args, **self.op_kwargs)


class TitanBuildModelOperator(BaseOperator):
    """
    Executes a Python callable
    :param python_callable: A reference to an object that is callable
    :type python_callable: python callable
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function
    :type op_kwargs: dict
    :param op_args: a list of positional arguments that will get unpacked when
        calling your callable
    :type op_args: list
    :param provide_context: if set to true, Airflow will pass a set of
        keyword arguments that can be used in your function. This set of
        kwargs correspond exactly to what you can use in your jinja
        templates. For this to work, you need to define `**kwargs` in your
        function header.
    :type provide_context: bool
    :param templates_dict: a dictionary where the values are templates that
        will get templated by the Airflow engine sometime between
        ``__init__`` and ``execute`` takes place and are made available
        in your callable's context after the template has been applied. (templated)
    :type templates_dict: dict of str
    :param templates_exts: a list of file extensions to resolve while
        processing templated fields, for examples ``['.sql', '.hql']``
    :type templates_exts: list(str)
    """
    template_fields = ('templates_dict',)
    template_ext = tuple()
    ui_color = '#ffefeb'

    @apply_defaults
    def __init__(
            self,
            build_conf,
            on_end_hook=None,
            op_args=None,
            op_kwargs=None,
            templates_dict=None,
            *args, **kwargs):
        super(TitanBuildModelOperator, self).__init__(*args, **kwargs)
        self.build_conf = build_conf
        self.on_end_hook = on_end_hook
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.templates_dict = templates_dict

    def execute(self, context):
        context.update(self.op_kwargs)
        context['templates_dict'] = self.templates_dict
        self.op_kwargs = context
        conf = self.execute_callable(self.build_conf)

        if not type(conf) == dict:
            raise AirflowException('conf is not dict %s' % conf)

        self.log.info('conf: %s' % conf)

        docker_repo = conf.get('docker_repo')
        docker_username = conf.get('docker_username')
        docker_password = conf.get('docker_password')
        download_working_dir = conf.get('download_working_dir')
        train_code_dir = conf.get('train_code_dir')
        model_name = conf.get('model_name') 
        model_version = conf.get('model_version')

        if not docker_repo: raise AirflowException('`docker_repo` is None')
        if not docker_username: raise AirflowException('`docker_username` is None')
        if not docker_password: raise AirflowException('`docker_password` is None')
        if not download_working_dir: raise AirflowException('`download_working_dir` is None')
        if not train_code_dir: raise AirflowException('`train_code_dir` is None')
        if not model_name: raise AirflowException('`model_name` is None')
        if not model_version: raise AirflowException('`model_version` is None')
        
        working_dir = TitanHelper.make_task_working_dir(**self.op_kwargs)
        TitanHelper.cpdir(download_working_dir, working_dir)
        model_files = os.path.join(train_code_dir, '*')
        TitanHelper.copy_hdfs_to_local(model_files, working_dir)

        # generate build files
        cmd = 'docker run --rm -v {}:/model deeplearningrepo/core-python-wrapper:0.7 /model {} {} {} --force'.format(working_dir, model_name, model_version, docker_repo)
        sp.check_call(cmd, shell=True)

        build_dir = os.path.join(working_dir, 'build')
        model_image = ('{}/{}:{}').format(docker_repo, model_name.lower(), model_version)
        cmd = 'docker build --force-rm=true -t {} {}'.format(model_image, '.')
        sp.check_call(cmd, cwd=build_dir, shell=True)

        cmd = 'docker login -u {} -p {}'.format(docker_username, docker_password)
        sp.check_call(cmd, shell=True)

        cmd = 'docker push {}'.format(model_image)
        sp.check_call(cmd, cwd=build_dir, shell=True)

        return_value = self.execute_callable(self.on_end_hook)
        self.log.info("Done. Returned value was: %s", return_value)
        return return_value

    def execute_callable(self, func):
        return TitanHelper.execute_callable(func, *self.op_args, **self.op_kwargs)

seldon_deploy_model_template="""{
    "apiVersion": "machinelearning.seldon.io/v1alpha1",
    "kind": "SeldonDeployment",
    "metadata": {
        "labels": {
            "app": "seldon"
        },
        "name": "seldon-deployment-{{deployment_id}}"
    },
    "spec": {
        "annotations": {
            "project_name": "{{deployment_id}}",
            "deployment_version": "0.1"
        },
        "name": "{{deployment_id}}-deployment",
        "oauth_key": "{{client_id}}",
        "oauth_secret": "{{client_secret}}",
        "predictors": [
            {
                "componentSpec": {
                    "spec": {
                        "containers": [
                            {
                                "image": "{{model_image}}",
                                "imagePullPolicy": "Always",
                                "name": "{{model_name}}",
                                "resources": {
                                    "requests": {
                                        "memory": "{{model_memory}}Mi"
                                    }
                                }
                            }
                        ],
                        "terminationGracePeriodSeconds": 20
                    }
                },
                "graph": {
                    "children": [],
                    "name": "{{model_name}}",
                    "endpoint": {
                        "type" : "REST"
                    },
                    "type": "MODEL"
                },
                "name": "{{predictor_id}}",
                "replicas": {{replicas}},
                "annotations": {
                "predictor_version" : "0.1"
                }
            }
        ]
    }
}
"""

seldon_deploy_abtest_template="""{
     "apiVersion": "machinelearning.seldon.io/v1alpha1",
     "kind": "SeldonDeployment",
     "metadata": {
         "labels": { "app": "seldon" },
         "name": "seldon-deployment-{{deployment_id}}"
     },

     "spec": {
         "annotations": { "project_name": "Prediction", "deployment_version": "v1" },

         "name": "{{deployment_id}}-deployment",
         "oauth_key": "{{client_id}}",
         "oauth_secret": "{{client_secret}}",
         "predictors": [
             {
                 "componentSpec":
                 {
                     "spec": {
                     "containers":
                     [
                         {
                             "image": "{{model_image}}",
                             "imagePullPolicy": "Always",
                             "name": "{{model_name}}",
                             "resources": { "requests": { "memory": "{{model_memory}}Mi" } }
                         },

                         {
                             "image": "{{model_image_b}}",
                             "imagePullPolicy": "Always",
                             "name": "{{model_name_b}}",
                             "resources": { "requests": { "memory": "{{model_memory_b}}Mi" } }
                         }
                     ],
                     "terminationGracePeriodSeconds": 20
                     }
                 },
                 "name": "{{predictor_id}}",
                 "replicas": {{replicas}},
                 "annotations": { "predictor_version": "v1" },
                 "graph":
                 {
                     "name": "random-ab-test",
                     "endpoint":{},
                     "implementation":"RANDOM_ABTEST",
                     "parameters": [ { "name":"ratioA", "value":"{{ratio_a}}", "type":"FLOAT" } ],
                     "children":
                     [
                         { "name": "{{model_name}}", "endpoint":{ "type":"REST" }, "type":"MODEL", "children":[] },
                         { "name": "{{model_name_b}}", "endpoint":{ "type":"REST" }, "type":"MODEL", "children":[] }
                     ]
                 }
             }
         ]
   }
 }
"""

class TitanDeployModelOperator(BaseOperator):
    """
    Executes a Python callable
    :param python_callable: A reference to an object that is callable
    :type python_callable: python callable
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function
    :type op_kwargs: dict
    :param op_args: a list of positional arguments that will get unpacked when
        calling your callable
    :type op_args: list
    :param provide_context: if set to true, Airflow will pass a set of
        keyword arguments that can be used in your function. This set of
        kwargs correspond exactly to what you can use in your jinja
        templates. For this to work, you need to define `**kwargs` in your
        function header.
    :type provide_context: bool
    :param templates_dict: a dictionary where the values are templates that
        will get templated by the Airflow engine sometime between
        ``__init__`` and ``execute`` takes place and are made available
        in your callable's context after the template has been applied. (templated)
    :type templates_dict: dict of str
    :param templates_exts: a list of file extensions to resolve while
        processing templated fields, for examples ``['.sql', '.hql']``
    :type templates_exts: list(str)
    """
    template_fields = ('templates_dict',)
    template_ext = tuple()
    ui_color = '#ffefeb'

    @apply_defaults
    def __init__(
            self,
            deploy_conf,
            on_end_hook=None,
            op_args=None,
            op_kwargs=None,
            templates_dict=None,
            *args, **kwargs):
        super(TitanDeployModelOperator, self).__init__(*args, **kwargs)
        self.deploy_conf = deploy_conf
        self.on_end_hook = on_end_hook
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.templates_dict = templates_dict

    def execute(self, context):
        context.update(self.op_kwargs)
        context['templates_dict'] = self.templates_dict
        self.op_kwargs = context
        conf = self.execute_callable(self.deploy_conf)

        if not type(conf) == dict:
            raise AirflowException('conf is not dict %s' % conf)

        self.log.info('conf: %s' % conf)

        docker_repo = conf.get('docker_repo')
        model_name = conf.get('model_name') 
        model_version = conf.get('model_version')
        model_memory = conf.get('model_memory')
        replicas = conf.get('replicas', 1)
        deployment_id = conf.get('deployment_id')
        client_id = conf.get('client_id')
        client_secret = conf.get('client_secret')
        predictor_id = conf.get('predictor_id')
        model_abtest = conf.get('model_abtest', False)

        if model_abtest:
            model_name_b = conf.get('model_name_b')
            model_version_b = conf.get('model_version_b')
            model_memory_b = conf.get('model_memory_b')
            ratio_a = conf.get('ratio_a')
            
            if not model_name_b: raise AirflowException('`model_name_b` is None')
            if not model_version_b: raise AirflowException('`model_version_b` is None')
            if not model_memory_b or not type(model_memory_b) == int or model_memory_b < 200:
                raise AirflowException('`model_memory_b` is invalid')
            if not ratio_a or not type(ratio_a) == float or ratio_a <= 0 or ratio_a > 1:
                raise AirflowException('`ratio_a` is not valid')

        if not deployment_id: raise AirflowException('`deployment_id` is None')
        if not client_id: raise AirflowException('`client_id` is None')
        if not client_secret: raise AirflowException('`client_secret` is None')
        if not predictor_id: raise AirflowException('`predictor_id` is None')
        if not docker_repo: raise AirflowException('`docker_repo` is None')
        if not model_name: raise AirflowException('`model_name` is None')
        if not model_version: raise AirflowException('`model_version` is None')
        if not model_memory or not type(model_memory) == int or model_memory < 200:
            raise AirflowException('`model_memory` is invalid')
        
        working_dir = TitanHelper.make_task_working_dir(**self.op_kwargs)
        deploy_file_path = os.path.join(working_dir, 'deploy.json')
        model_image = ('{}/{}:{}').format(docker_repo, model_name.lower(), model_version)

        #self.log.info('template = %s' % seldon_deploy_model_template)
        if model_abtest:
            model_image_b = ('{}/{}:{}').format(docker_repo, model_name_b.lower(), model_version_b)
            content = jinja2.Template(seldon_deploy_abtest_template).render(\
                deployment_id=deployment_id.lower(),\
                predictor_id=predictor_id.lower(),\
                client_id=client_id.lower(),\
                client_secret=client_secret.lower(),\
                replicas=replicas,\
                ratio_a=ratio_a,\
                model_name=model_name.lower(),\
                model_name_b=model_name_b.lower(),\
                model_version=model_version.lower(),\
                model_version_b=model_version_b.lower(),\
                model_image=model_image,\
                model_image_b=model_image_b,\
                model_memory=model_memory,\
                model_memory_b=model_memory_b)
        else:
            content = jinja2.Template(seldon_deploy_model_template).render(\
                deployment_id=deployment_id.lower(),\
                predictor_id=predictor_id.lower(),\
                client_id=client_id.lower(),\
                client_secret=client_secret.lower(),\
                replicas=replicas,\
                model_name=model_name.lower(),\
                model_version=model_version.lower(),\
                model_image=model_image,\
                model_memory=model_memory)

        with open(deploy_file_path, 'w') as f:
            f.write(content)

        self.log.info(deploy_file_path)
        cmd = 'kubectl apply -f {} -n seldon'.format(deploy_file_path)
        sp.check_call(cmd, shell=True)

        return_value = self.execute_callable(self.on_end_hook)
        self.log.info("Done. Returned value was: %s", return_value)
        return return_value

    def execute_callable(self, func):
        return TitanHelper.execute_callable(func, *self.op_args, **self.op_kwargs)


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
import subprocess
import sys
import types
import uuid
import logging

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

th = TitanHelper

class TitanBootstrapOperator(BaseOperator):
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

    """
    template_fields = ('templates_dict',)
    template_ext = tuple()
    ui_color = '#ffefeb'

    @apply_defaults
    def __init__(
            self,
            python_callable=None,
            provide_context=False,
            templates_dict=None,
            templates_exts=None,
            op_args=None,
            op_kwargs=None,
            *args, **kwargs):
        super(TitanBootstrapOperator, self).__init__(*args, **kwargs)
        if python_callable and not callable(python_callable):
            raise AirflowException('`python_callable` param must be callable')
        self.python_callable = python_callable
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.provide_context = provide_context
        self.templates_dict = templates_dict
        if templates_exts:
            self.template_ext = templates_exts


    def execute(self, context):
        self.log.info("context: %s" % context)
        context.update(self.op_kwargs)
        self.op_kwargs = context
        tracking_id = TitanHelper.get_uuid() 
        TitanHelper.put_xcom_value(self.op_kwargs['ti'], tracking_id=tracking_id) 

        return_value = self.execute_callable() if self.python_callable else None
        self.log.info("Done. Returned value was: %s", return_value)
        return return_value

    def execute_callable(self):
        return self.python_callable(*self.op_args, **self.op_kwargs)

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
            python_callable,
            op_args=None,
            op_kwargs=None,
            provide_context=False,
            templates_dict=None,
            templates_exts=None,
            *args, **kwargs):
        super(TitanTrainOperator, self).__init__(*args, **kwargs)
        if not callable(python_callable):
            raise AirflowException('`python_callable` param must be callable')
        self.python_callable = python_callable
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.provide_context = provide_context
        self.templates_dict = templates_dict
        if templates_exts:
            self.template_ext = templates_exts

    def execute(self, context):
        if self.provide_context:
            context.update(self.op_kwargs)
            context['templates_dict'] = self.templates_dict
            self.op_kwargs = context

        return_value = self.execute_callable()
        self.log.info("Done. Returned value was: %s", return_value)
        return return_value

    def execute_callable(self):
        return self.python_callable(*self.op_args, **self.op_kwargs)


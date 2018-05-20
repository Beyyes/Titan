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

import json

from airflow.exceptions import AirflowException
from airflow.models import DagRun, DagBag
from airflow.utils import timezone
from airflow.utils.state import State


def clear_dag(dag_id, execution_date, only_failed=False, only_running=False):

    dagbag = DagBag()

    if dag_id not in dagbag.dags:
        raise AirflowException("Dag id {} not found".format(dag_id))

    dag = dagbag.get_dag(dag_id)

    if not execution_date:
		raise AirflowException("Exection_date is None")

    assert timezone.is_localized(execution_date)

    count = dag.clear(
                start_date=execution_date,
                end_date=execution_date,
                only_failed=only_failed,
                only_running=only_running,
                include_subdags=True)

    return count

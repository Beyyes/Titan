import argparse
import commands
import yaml
import os
import logging
import logging.config
import subprocess

logger = logging.getLogger(__name__)

def copy_js():
    config_file_path = "config/cluster-config.yaml"
    with open(config_file_path, "r") as f:
        raw_config = yaml.load(f)

    master = ""
    for host in raw_config['host-list']:
        if host['role'] == 'master':
            master = host['hostname']
            break

    # Airflow
    file = open("ui/TempAirflow.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/Airflow.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempGrafana.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/Grafana.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempGrafanaAlert.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/GrafanaAlert.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempGrafanaDashboardImport.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/GrafanaDashboardImport.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempGrafanaDashboardManager.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/GrafanaDashboardManager.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempGrafanaDashboardMetrics.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/GrafanaDashboardMetrics.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempGrafanaDashboardNew.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/GrafanaDashboardNew.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempGrafanaDashboardNewFolder.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/GrafanaDashboardNewFolder.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempGrafanaDataSource.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/GrafanaDataSource.js", "w+")
    wfile.write(new_content)

    file = open("ui/TempKubernetes.js", "r")
    content = file.read()
    new_content = content.replace("DEFINE-MASTER-IP", master)
    wfile = open("ui/output/Kubernetes.js", "w+")
    wfile.write(new_content)

    cmd = "cp ui/output/* ../titan.ui/src/routes/Dashboard/"
    commands.getoutput(cmd)

if __name__ == '__main__':
    copy_js()
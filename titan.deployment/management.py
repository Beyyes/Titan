import argparse
import commands
import yaml
import os
import logging
import logging.config
import subprocess
import ui_deploy
#from ruamel import yaml

logger = logging.getLogger(__name__)

class Management:
    def __init__(self):
        return

    def k8s_deploy(self):
        print(log("deploy k8s cluster using kubeadm, this may take a few minutes"))
        cmd = "cd ../titan.deployment/kubernetes/ && " \
              "sudo pip install paramiko && " \
              "sudo pip install pyyaml"
        execute_shell(cmd, "Pip installing paramiko, pyyaml meets error!")
        cmd = "cd ../titan.deployment/kubernetes/ &&  " \
              "sudo python deploy.py -a deploy"
        execute_shell(cmd, "Deploy k8s cluster meets error!")

        # deploy k8s dashboard
        print(log("Deploy k8s dashboard"))
        cmd = "cd kubernetes/dashboard && sh create_k8s_dashboard.sh"
        execute_shell(cmd, "Deploy k8s dashboard meets error!")
        print ("You can access k8s dashboard by port 30280\r\n")

    def k8s_clean(self):
        print(log("Uninstall k8s cluster"))
        cmd = "cd ../titan.deployment/kubernetes/ && sudo python deploy.py -a reset"
        execute_shell(cmd, "Clean k8s cluster meets error!")
        cmd = "cd kubernetes/script && sh reset_k8s.sh"
        execute_shell(cmd, "Clean k8s environment meets error!")

    # maybe we also need a k8s service/deployment cleaning script
    def k8s_dashboard_deploy(self):
        print(log("Deploy k8s dashboard"))
        cmd = "cd kubernetes/dashboard && sh create_k8s_dashboard.sh"
        output = commands.getoutput(cmd)
        print(output)
        print ("\r\nYou can access k8s dashboard by port 30280")

    def pai_deploy(self):
        print(log("deploy PAI service, this may take some minutes"))
        # with open("config/cluster-config.yaml", "r") as k8s_cluster_file:
        #     yaml_obj = yaml.load(k8s_cluster_file.read())
        #
        #     host_list = yaml_obj["host-list"]
        #     host_list[0]['username'] = 'xxxxx'
        #     print(host_list)
        #
        #
        # # write to yaml
        # with open("config/service-config/cluster-configuration-tmp.yaml", "w") as pai_cluster_file:
        #     # yaml_obj = yaml.load(pai_cluster_file.read())
        #     # pai_cluster_file["machine-list"] = 0
        #     yaml.dump(yaml_obj, pai_cluster_file)

        # a cluster-configuration is needed
        configpath = os.getcwd() + "/config/service-config"
        cmd = "sudo rm -rf pai && " \
              "git clone https://github.com/Beyyes/pai && " \
              "cd pai/pai-management && " \
              "sudo pip install kubernetes &&" \
              "git checkout deploy_for_titan_prod && " \
              "sudo python deploy.py -d -p " + configpath
        execute_shell(cmd, "Deploy pai meets error!")

        print(log("deploy seldon"))
        cmd = "cd seldon && sudo sh start.sh"
        execute_shell(cmd, "Setup seldon meets error!")

    def pai_clean(self):
        print(log("clean PAI service, this may take some minutes"))
        cmd = "cd pai/pai-management && git checkout deploy_for_titan_prod && sudo python cleanup-service.py"
        execute_shell(cmd, "Clean PAI service meets error!")

        print(log("clean seldon services"))
        cmd = "cd seldon && sudo sh cleanup.sh"
        execute_shell(cmd, "Clean seldon meets error!")

    def airflow_deploy(self):
        print(log("deploy airflow"))
        print(log("Installing pip, python-software-properties, software-properties-common, gcc, pip"))
        cmd = "cd ../titan.workflow && " \
              "sudo apt-get install python-setuptools -y && " \
              "curl -LO https://files.pythonhosted.org/packages/ae/e8/2340d46ecadb1692a1e455f13f75e596d4eab3d11a57446f08259dee8f02/pip-10.0.1.tar.gz &&" \
              "tar -xzvf pip-10.0.1.tar.gz && " \
              "cd pip-10.0.1 && " \
              "sudo python setup.py install && " \
              "sudo pip install setuptools --user --upgrade && " \
              "sudo apt-get -y install python-software-properties && " \
              "sudo apt-get -y install software-properties-common && " \
              "sudo apt-get -y install gcc make build-essential libssl-dev libffi-dev python-dev"
        commands.getoutput(cmd)

        print(log("Installing airflow using source code"))
        cmd = "cd ../titan.workflow && sudo python setup.py install"
        execute_shell(cmd, "Installing airflow using source code meets error!")

        print("\r\n >>>>>> AIRFLOW_HOME has been set to $HOME/airflow, you need do these three things:\r\n"
              "1) run 'sudo apt-get install mysql-server -y' to install mysql, set the password\r\n"
              "2) create database airflow in mysql\r\n"
              "3) replace `sql_alchemy_conn = mysql://root:password@localhost/airflow` to $HOME/airflow/airflow.cfg\r\n"
              "4) set LocalExecutor to $HOME/airflow/airflow.cfg\r\n"
              "5) create dags folder in $HOME/airflow\r\n")

    def airflow_start(self):
        print(log("start airflow, make sure you have done the prerequisites in airflow-deploy"))
        cmd = "cd airflow && sh start.sh"
        execute_shell(cmd, "Starting airflow meets error!")

    def airflow_clean(self):
        print(log("Uninstall airflow"))
        print(log("systemctl stop airflow-webserver and airflow-scheduler, remove ~/airflow is optional"))
        cmd = "sudo systemctl stop airflow-webserver &&" \
              "sudo systemctl stop airflow-scheduler"
        execute_shell(cmd, "Meeing errors in stop airflow!")
        print("\r\nYou can access airflow web by master-ip:18880!\r\n")

    # a parameter of port is needed, port 8000 may be conflict with others
    def ui_deploy(self):
        print(log("deploy Titan UI, this may take a few minutes"))
        ui_deploy.copy_js()
        cmd = "cd ../titan.ui/ && sudo sh start.sh"
        execute_shell(cmd, "unable to stop titan ui")
        print("\r\nYou can access Titan UI by: master-ip:8000")

    def ui_clean(self):
        print(log("stop Titan UI"))
        cmd = "sudo lsof -i:8000 | awk '{print $2}'"
        pids = commands.getoutput(cmd)
        print("Kill Titan UI process\r\n")
        pids = pids.split("\n")
        print(pids)

        count = 0
        for pid in pids:
            if count > 0:
                cmd = "sudo kill -9 " + pid
                print(cmd)
                commands.getoutput(cmd)
            count += 1

        print("\r\nKill Titan UI process successfully!")

    def add_node(self, node_file):
        print(log("Add new node"))
        with open(node_file, "r") as f:
            raw_config = yaml.load(f)

        for node in raw_config['machine-list']:
            hostname = node['hostname']
            hostip = node['hostip']
            # token = commands.getoutput("sudo kubeadm token create")
            # hash = commands.getoutput("openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | "
            #                           "openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -hex | sed 's/^.* //'")
            # join_cmd = "kubeadm join {0}:6443 --token {1} " \
            #            "--discovery-token-ca-cert-hash sha256:{2}".format(hostip, token, hash)
            # execute_shell(join_cmd, "Join new node meets error!")
            #
            # label_nodes = "kubectl label nodes {0} machinetype=gpu && " \
            #               "kubectl label nodes {1} node-exporter=true && " \
            #               "kubectl label nodes {2} yarnrole=worker && " \
            #               "kubectl label nodes {3} hdfsrole=worker".format(hostname, hostname, hostname, hostname)
            # execute_shell(label_nodes, "Labels new node meets error!")

            yaml_config = commands.getoutput("kubectl get configmap host-configuration -o yaml")
            yaml_config = yaml.load(yaml_config)
            print(yaml_config)
            # print(yaml_config["data"])
            # print(yaml_config["data"]["host-configuration.yaml"])
            #content = yaml_config["data"]["host-configuration.yaml"]
            append_node = "    \n\n{0}:\n    ip: {1}\n    hostname: spark-master\n    " \
                   "dataFolder: \n    machinetype: gpu\n    hdfsrole: worker\n    yarnrole: worker\n".format(hostname, hostip)
            yaml_config["data"]["host-configuration.yaml"] = yaml_config["data"]["host-configuration.yaml"] + append_node
            print(yaml_config)
            with open("host-configuration.yaml", "w+") as f:
                yaml.dump(yaml_config, f, default_flow_style=False)
            f.close()

            # config_command = "kubectl create configmap host-configuration --from-file=host-configuration/ --dry-run -o yaml | kubectl replace -f -"
            # execute_shell(config_command, "Modify new node configmap meets error!")

        print("\r\n add new node successfully!")

    def delete_node(self, node_file):
        return

    def test(self):
        host_config = commands.getoutput("kubectl get configmap host-configuration -o yaml")
        with open("host-configuration.yaml", "w+") as f:
            f.write(host_config)
        f.close()
        config_shell = "kubectl delete configmap host-configuration_tmp && " \
                       "kubectl create configmap host-configuration_tmp --from-file=host-configuration.yaml"

    def all_deploy(self):
        print('all-deploy')
        self.k8s_deploy()
        self.airflow_deploy()
        self.ui_deploy()

def log(log):
    return ("\r\n>>>>>>>>>>>>>>>>>>>>>>> {0} <<<<<<<<<<<<<<<<<<<<<<<\r\n").format(log)

def execute_shell(shell_cmd, error_msg):
    try:
        subprocess.check_call(shell_cmd, shell=True)
    except subprocess.CalledProcessError:
        logger.error(error_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', required=True, default=None,
                        help="action to execute. select one from 'k8s', 'ui', 'airflow' and 'all'")
    parser.add_argument('-f', '--file', default=None, help="An yamlfile with the nodelist to maintain")
    args = parser.parse_args()

    management = Management()
    if args.action == 'k8s-deploy':
        management.k8s_deploy()
    elif args.action == 'k8s-clean':
        management.k8s_clean()
    elif args.action == 'k8s-dashboard':
        management.k8s_dashboard_deploy()
    elif args.action == 'pai-deploy':
        management.pai_deploy()
    elif args.action == 'pai-clean':
        management.pai_clean()
    elif args.action == 'airflow-deploy':
        management.airflow_deploy()
    elif args.action == 'airflow-start':
        management.airflow_start()
    elif args.action == 'airflow-clean':
        management.airflow_clean()
    elif args.action == 'ui-deploy':
        management.ui_deploy()
    elif args.action == 'ui-clean':
        management.ui_clean()
    elif args.action == 'add-node':
        management.add_node(args.file)
    elif args.action == 'all':
        management.all_deploy()
    else:
        print("Error parameter for ACTION")

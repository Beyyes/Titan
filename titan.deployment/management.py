import argparse
import commands
import yaml
import os
import logging
import logging.config

class Management:
    def __init__(self):
        return

    def kubeadm_install(self):
        # install kubeadm
        return

    def k8s_deploy(self):
        # deploy k8s cluster
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> deploy k8s cluster using kubeadm, this may take a few minutes <<<<<<<<<<<<<<<<<<<<<<<\r\n")

        cmd = "cd ../titan.deployment/kubernetes/ && " \
              "sudo pip install paramiko && " \
              "sudo pip install pyyaml"
        commands.getoutput(cmd)
        cmd = "cd ../titan.deployment/kubernetes/ &&  " \
              "sudo python deploy.py -a deploy"
        output = commands.getoutput(cmd)
        print(output)

        # deploy k8s dashboard
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> deploy k8s dashboard <<<<<<<<<<<<<<<<<<<<<<<\r\n")
        cmd = "cd kubernetes/dashboard && sh create_k8s_dashboard.sh"
        output = commands.getoutput(cmd)
        print(output)
        print ("You can access k8s dashboard by port 30280\r\n")

    def k8s_reset(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> uninstall k8s cluster <<<<<<<<<<<<<<<<<<<<<<<\r\n")

        cmd = "cd ../titan.deployment/kubernetes/ && sudo python deploy.py -a reset"
        output = commands.getoutput(cmd)
        print(output)
        cmd = "cd kubernetes/script && sh reset_k8s.sh"
        output = commands.getoutput(cmd)
        print(output)

    # maybe we also need a k8s service/deployment cleaning script
    def k8s_dashboard_deploy(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> deploy k8s dashboard <<<<<<<<<<<<<<<<<<<<<<<\r\n")
        cmd = "cd kubernetes/dashboard && sh create_k8s_dashboard.sh"
        output = commands.getoutput(cmd)
        print(output)
        print ("\r\nYou can access k8s dashboard by port 30280")

    def pai_deploy(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> deploy PAI service, this may take some minutes <<<<<<<<<<<<<<<<<<<<<<<\r\n")
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
              "git checkout deploy_for_titan_prod && " \
              "sudo python deploy.py -d -p " + configpath
        output = commands.getoutput(cmd)
        print(output)

        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> deploy seldon <<<<<<<<<<<<<<<<<<<<<<<\r\n")
        cmd = "cd seldon && sh start.sh"
        output = commands.getoutput(cmd)
        print(output)

    def pai_clean(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> clean PAI service, this may take some minutes <<<<<<<<<<<<<<<<<<<<<<<\r\n")

        cmd = "cd pai/pai-management && git checkout deploy_for_titan_prod && sudo python cleanup-service.py"
        output = commands.getoutput(cmd)
        print(output)

        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> clean seldon services <<<<<<<<<<<<<<<<<<<<<<<\r\n")
        cmd = "cd seldon && sh cleanup.sh"
        output = commands.getoutput(cmd)
        print(output)

    def airflow_deploy(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> deploy airflow <<<<<<<<<<<<<<<<<<<<<<<\r\n")

        print("\r\n >>>>>> Installing pip, python-software-properties, software-properties-common, gcc")
        cmd = "cd ../titan.workflow && " \
              "sudo apt-get install python-setuptools -y && " \
              "curl -LO https://files.pythonhosted.org/packages/ae/e8/2340d46ecadb1692a1e455f13f75e596d4eab3d11a57446f08259dee8f02/pip-10.0.1.tar.gz &&" \
              "tar -xzvf pip-10.0.1.tar.gz && " \
              "cd pip-10.0.1 && " \
              "sudo python setup.py install && " \
              "pip install setuptools --user --upgrade && " \
              "sudo apt-get -y install python-software-properties && " \
              "sudo apt-get -y install software-properties-common && " \
              "sudo apt-get -y install gcc make build-essential libssl-dev libffi-dev python-dev"
        commands.getoutput(cmd)
        #path = os.getcwd()
        #cmd = "export AIRFLOW_HOME=" + path
        #commands.getoutput(cmd)

        print("\r\n >>>>>> Installing airflow using source code")
        cmd = "cd ../titan.workflow &&" \
              "sudo python setup.py install &&" \
              "sudo apt-get install mysql-server -y"
        output = commands.getoutput(cmd)
        print(output)
        print("\r\n >>>>>> AIRFLOW_HOME has been set to $HOME/airflow, you need install mysql using 'sudo apt-get install mysql-server' "
              "and input the username and password and set LocalExecutor to $HOME/airflow/airflow.cfg !! \r\n")


    def airflow_start(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> start airflow, before start, make sure you have set the Executor and MySQL auth to airflow.cfg, and create Dags file"
              " in ~/airflow/dags <<<<<<<<<<<<<<<<<<<<<<<")

        print(commands.getoutput("sudo pip install kubernetes\r\n"))
        print(commands.getoutput("sudo service mysql restart\r\n"))
        print(commands.getoutput("sudo apt-get install libmysqlclient-dev -y\r\n"))
        print(commands.getoutput("sudo pip install mysqlclient"))
        # print(commands.getoutput("sudo service mysql restart"))
        # print(commands.getoutput("sudo service mysql restart"))
        # cmd = "sudo pip install kubernetes && " \
        #       "sudo apt-get update && " \
        #       "sudo service mysql restart && " \
        #       "sudo apt-get install libmysqlclient-dev && " \
        #       "sudo pip install mysqlclient"
        # output = commands.getoutput(cmd)
        # print(output)

        print(commands.getoutput("airflow && cd ~/airflow"))
        print(commands.getoutput("mkdir dags"))
        print(commands.getoutput("sudo mkdir -p /usr/lib/systemd/system"))
        print(commands.getoutput("sudo cp airflow/airflow-webserver.service /usr/lib/systemd/system"))
        print(commands.getoutput("sudo cp airflow/airflow-scheduler.service /usr/lib/systemd/system"))
        print(commands.getoutput("sudo systemctl start airflow-webserver"))
        print(commands.getoutput("sudo systemctl start airflow-scheduler"))
        # cmd = "airflow && cd ~/airflow && " \
        #       "mkdir dags && " \
        #       "sudo mkdir -p /usr/lib/systemd/system && " \
        #       "sudo cp airflow/airflow-webserver.service /usr/lib/systemd/system  && " \
        #       "sudo cp airflow/airflow-scheduler.service /usr/lib/systemd/system && " \
        #       "sudo systemctl start airflow-webserver && " \
        #       "sudo systemctl start airflow-scheduler"
        # output = commands.getoutput(cmd)
        # print(output)

    def airflow_clean(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> uninstall airflow <<<<<<<<<<<<<<<<<<<<<<<")

        print(">>>>>> remove ~/airflow and systemctl stop airflow-webserver and airflow-scheduler")
        cmd = "rm -rf ~/airflow &&" \
              "sudo systemctl stop airflow-webserver &&" \
              "sudo systemctl stop airflow-scheduler"
        output = commands.getoutput(cmd)
        print(output)

    # a parameter of port is needed, port 8000 may be conflict with others
    def ui_deploy(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> deploy Titan UI <<<<<<<<<<<<<<<<<<<<<<<")
        cmd = "cd ../titan.ui/ && " \
              "sh start.sh"
        # output = commands.getoutput(cmd)
        # print(output)
        #
        # cmd = '"nohup npm start &"'
        output = commands.getoutput(cmd)
        print(output)
        print("\r\nYou can access Titan UI by: master-ip:8000")

    def ui_clean(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> stop Titan UI <<<<<<<<<<<<<<<<<<<<<<<")

        cmd = "lsof -i:8000 | awk '{print $2}'"
        pids = commands.getoutput(cmd)
        print("Kill Titan UI process\r\n")
        pids = pids.split("\n")
        print(pids)

        count = 0
        for pid in pids:
            if count > 0:
                cmd = "kill " + pid
                print(cmd)
                commands.getoutput(cmd)
            count += 1

        print("\r\nKill Titan UI process successfully!")

    def add_node(self):
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> add new node <<<<<<<<<<<<<<<<<<<<<<<")


        token = commands.getoutput("sudo kubeadm token create")
        hash = commands.getoutput("openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | "
                                  "openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -hex | sed 's/^.* //'")
        # bjag6l.tgr33e1wxkieoop1

        print("Kill Titan UI process\r\n")

        print("\r\n add new node successfully!")

    def all_deploy(self):
        print('all-deploy')
        self.k8s_deploy()
        self.airflow_deploy()
        self.ui_deploy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', required=True, default=None,
                        help="action to execute. select one from 'k8s', 'ui', 'airflow' and 'all'")
    args = parser.parse_args()

    management = Management()
    if args.action == 'k8s-deploy':
        management.k8s_deploy()
    elif args.action == 'k8s-clean':
        management.k8s_reset()
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
        management.add_node()
    elif args.action == 'all':
        management.all_deploy()
    else:
        print("Error parameter for ACTION")

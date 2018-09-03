import argparse
import commands
import yaml
import os


class Management:
    def __init__(self):
        # print('init')
        return

    def kubeadm_install(self):
        # install kubeadm
        cmd = "cd ../titan.deployment/kubernetes/ && " \
              "sudo ./deploy.py -a deploy"
        output = commands.getoutput(cmd)
        print(output)

    def k8s_deploy(self):
        # deploy k8s cluster
        print(">>>>>>>>>>>>>>>>>>>>>>> deploy k8s cluster using kubeadm <<<<<<<<<<<<<<<<<<<<<<<")
        cmd = "cd ../titan.deployment/kubernetes/ && " \
              "sudo python deploy.py -a deploy"
        output = commands.getoutput(cmd)
        print(output)

        # deploy k8s dashboard
        print("\r\n>>>>>>>>>>>>>>>>>>>>>>> deploy k8s dashboard <<<<<<<<<<<<<<<<<<<<<<<")
        cmd = "cd kubernetes/dashboard && " \
              "kubectl create -f dashboard-rbac.yaml && " \
              "kubectl create -f dashboard-controller.yaml && " \
              "kubectl create -f dashboard-service.yaml"
        output = commands.getoutput(cmd)
        print(output)

    def k8s_reset(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> reset k8s cluster <<<<<<<<<<<<<<<<<<<<<<<")
        cmd = "cd ../titan.deployment/kubernetes/ && " \
              "sudo python deploy.py -a reset"
        output = commands.getoutput(cmd)
        print(output)

    # maybe we also need a k8s service/deployment cleaning script
    def k8s_dashboard_deploy(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> deploy k8s dashboard <<<<<<<<<<<<<<<<<<<<<<<")
        cmd = "cd kubernetes/dashboard && " \
              "kubectl create -f dashboard-rbac.yaml && " \
              "kubectl create -f dashboard-controller.yaml && " \
              "kubectl create -f dashboard-service.yaml"
        output = commands.getoutput(cmd)
        print(output)
        print ("You can access k8s dashboard by port 30280\r\n")

    def pai_deploy(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> deploy PAI service, this may take some minutes <<<<<<<<<<<<<<<<<<<<<<<")
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

        configpath = os.getcwd() + "/config/service-config"
        # a cluster-configuration is needed
        cmd = "git clone https://github.com/Beyyes/pai && " \
              "cd pai/pai-management && " \
              "git checkout deploy_for_titan_prod && " \
              "sudo python deploy.py -d -p " + configpath
        output = commands.getoutput(cmd)
        print(output)

    def pai_clear(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> clean PAI service, this may take some minutes <<<<<<<<<<<<<<<<<<<<<<<")

        # a cluster-configuration is needed
        cmd = "cd pai/pai-management && " \
              "git checkout deploy_for_titan_prod && " \
              "sudo python cleanup-service.py"
        output = commands.getoutput(cmd)
        print(output)

    def airflow_deploy(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> deploy airflow <<<<<<<<<<<<<<<<<<<<<<<")

        print(">>>>>> Installing pip, python-software-properties, software-properties-common, gcc")
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

        print(">>>>>> Installing airflow using source code")
        cmd = "cd ../titan.workflow &&" \
              "sudo python setup.py install"
        output = commands.getoutput(cmd)
        print(output)

        print(">>>>>> AIRFLOW_HOME has been set to $HOME/airflow, and mysql will be installed next, you should input "
              "the username and password to $HOME/airflow/airflow.cfg !")
        cmd = "airflow &&" \
              "cd ~/airflow && " \
              "pip install kubernetes && " \
              "sudo apt-get update && " \
              "sudo apt-get install mysql-server && " \
              "service mysql restart && " \
              "sudo apt-get install libmysqlclient-dev && " \
              "pip install mysqlclient"
        output = commands.getoutput(cmd)
        print(output)

        # print(">>>>>> Installing mysql-server, mysql-client \r\n")
        # cmd = "sudo apt-get install mysql-server mysql-client && " \
        #       "sudo apt-get install libmysqlclient-dev && "
        # output = commands.getoutput(cmd)
        # print(output)

        print(">>>>>> Pip Installing mysqlclient, airflow[mysql,crypto,password] \r\n")
        cmd = "sudo mkdir -p /usr/lib/systemd/system && " \
              "sudo cp airflow/airflow-webserver.service /usr/lib/systemd/system  && " \
              "sudo cp airflow/airflow-scheduler.service /usr/lib/systemd/system && " \
              "sudo systemctl start airflow-webserver && " \
              "sudo systemctl start airflow-scheduler"
        output = commands.getoutput(cmd)
        print(output)

    def airflow_uninstall(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> uninstall airflow <<<<<<<<<<<<<<<<<<<<<<<")

        print(">>>>>> Uninstalling airflow")
        cmd = "rm -rf ~/airflow"
        output = commands.getoutput(cmd)
        print(output)

        print(">>>>>> Pip Installing mysqlclient, airflow[mysql,crypto,password] \r\n")
        cmd = "sudo pip install mysqlclient && " \
              "sudo pip install airflow[mysql,crypto,password]"
        output = commands.getoutput(cmd)
        print(output)

    # a parameter of port is needed, port 8000 may be conflict with others
    def ui_deploy(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> deploy Titan UI <<<<<<<<<<<<<<<<<<<<<<<")
        cmd = "cd ../titan.ui/ && " \
              "sh start.sh"
        # output = commands.getoutput(cmd)
        # print(output)
        #
        # cmd = '"nohup npm start &"'
        output = commands.getoutput(cmd)
        print(output)
        print("\r\n You can access Titan UI by: master-ip:8000")

    def ui_stop(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> stop Titan UI <<<<<<<<<<<<<<<<<<<<<<<")

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

    # single node shell, not k8s deployment
    def airflow_deploy_shell(self, airflow_home):
        cmd = "export AIRFLOW_HOME=" + airflow_home + \
              "apt-get install mysql-server mysql-client && " \
              "kubectl create -f airflow-deployment.yaml && " \
              "kubectl create -f airflow-service.yaml"
        output = commands.getoutput(cmd)
        print(output)

    def seldon_deploy(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> deploy seldon <<<<<<<<<<<<<<<<<<<<<<<")
        cmd = "cd ../titan.deployment/seldon/script && " \
              "sudo ./install_seldon.sh"
        output = commands.getoutput(cmd)
        print(output)

    def grafana_deploy(self):
        cmd = "cd config/grafana && " \
              "sh node-label.sh && " \
              "kubectl create -f grafana-deployment.yaml && " \
              "kubectl create -f grafana-service.yaml"
        output = commands.getoutput(cmd)
        print(output)

    def all_deploy(self):
        print('all-deploy')
        self.k8s_deploy()
        self.airflow_deploy()
        self.grafana_deploy()
        self.ui_deploy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', required=True, default=None,
                        help="action to execute. select one from 'k8s', 'ui', 'airflow' and 'all'")
    args = parser.parse_args()

    management = Management()
    if args.action == 'k8s-deploy':
        management.k8s_deploy()
    elif args.action == 'k8s-reset':
        management.k8s_reset()
    elif args.action == 'k8s-dashboard':
        management.k8s_dashboard_deploy()
    elif args.action == 'pai-deploy':
        management.pai_deploy()
    elif args.action == 'pai-clear':
        management.pai_clear()
    elif args.action == 'airflow-deploy':
        management.airflow_deploy()
    elif args.action == 'airflow-uninstall':
        management.airflow_uninstall()
    elif args.action == 'grafana':
        management.grafana_deploy()
    elif args.action == 'ui-deploy':
        management.ui_deploy()
    elif args.action == 'ui-stop':
        management.ui_stop()
    elif args.action == 'all':
        management.all_deploy()
    else:
        print("Error parameter for ACTION")

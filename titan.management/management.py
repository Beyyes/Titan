
import argparse
import commands

class Management:
    def __init__(self):
        print('a')

    def k8s_deploy(self):
        cmd = "cd ../titan.deployment/kubernetes/ && " \
              "sudo ./deploy.py -a deploy"
        output = commands.getstatusoutput(cmd)
        print(output)

    def k8s_reset(self):
        cmd = "cd ../titan.deployment/kubernetes/ && " \
              "sudo ./deploy.py -a reset"
        output = commands.getstatusoutput(cmd)
        print(output)

    # maybe we also need a k8s service/deployment cleaning script
    def k8s_dashboard_deploy(self):
        cmd = "cd config/dashboard && " \
              "kubectl create -f dashboard-rbac.yaml && " \
              "kubectl create -f dashboard-controller.yaml && " \
              "kubectl create -f dashboard-service.yaml"
        output = commands.getstatusoutput(cmd)
        print(output)

    # a parameter of port is needed, port 8000 may be conflict with others
    def ui_deploy(self):
        cmd = "cd ../titan.ui/ && " \
              "npm install && " \
              "npm start"
        output = commands.getstatusoutput(cmd)
        print(output)

    def seldon_deploy(self):
        cmd = "cd ../titan.deployment/seldon/script && " \
              "sudo ./install_seldon.sh"
        output = commands.getstatusoutput(cmd)
        print(output)

    def airflow_deploy(self):
        cmd = "cd config/airflow && " \
              "sh node-label.sh && " \
              "kubectl create -f airflow-deployment.yaml && " \
              "kubectl create -f airflow-service.yaml"
        output = commands.getstatusoutput(cmd)
        print(output)

    # single node shell, not k8s deployment
    def airflow_deploy_shell(self, airflow_home):
        cmd = "export AIRFLOW_HOME=" + airflow_home + \
              "apt-get install mysql-server mysql-client && " \
              "kubectl create -f airflow-deployment.yaml && " \
              "kubectl create -f airflow-service.yaml"
        output = commands.getstatusoutput(cmd)
        print(output)

    def grafana_deploy(self):
        cmd = "cd config/grafana && " \
              "sh node-label.sh && " \
              "kubectl create -f grafana-deployment.yaml && " \
              "kubectl create -f grafana-service.yaml"
        output = commands.getstatusoutput(cmd)
        print(output)

    def pai_deploy(self):
        # a cluster-configuration is needed
        cmd = "./deploy.py -d -p /cluster-configuration/ -s"

    def mysql_deploy(self):
        # a cluster-configuration is needed
        cmd = "./deploy.py -d -p /cluster-configuration/ -s"

    def all_deploy(self):
        print('all-deploy')
        self.k8s_deploy()
        self.airflow_deploy()
        self.grafana_deploy()
        self.ui_deploy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', required=True, default=None, help="action to execute. select one from 'k8s', 'ui', 'airflow' and 'all'")
    args = parser.parse_args()

    management = Management()
    if args.action == 'k8s-deploy':
        management.k8s_deploy()
    elif args.action == 'k8s-reset':
        management.k8s_reset()
    elif args.action == 'k8s-dashboard':
        management.k8s_dashboard_deploy()
    elif args.action == 'airflow':
        management.airflow_deploy()
    elif args.action == 'grafana':
        management.grafana_deploy()
    elif args.action == 'ui':
        management.ui_deploy()
    elif args.action == 'all':
        management.all_deploy()
    else:
        print("Error parameter for ACTION")
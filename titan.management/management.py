
import argparse
import commands

class Management:
    def __init__(self):
        print('a')

    def k8s_deploy(self):
        cmd = "cd ../titan.deployment/kubernetes/ && sudo ./deploy.py -a deploy"
        output = commands.getstatusoutput(cmd)
        print(output)

    def ui_deploy(self):
        cmd = "cd ../titan.deployment/kubernetes/ && sudo ./deploy.py -a deploy"
        output = commands.getstatusoutput(cmd)
        print(output)

    def seldon_deploy(self):
        cmd = "cd ../titan.deployment/seldon/script && sudo ./install_seldon.sh"
        output = commands.getstatusoutput(cmd)
        print(output)

    def all_deploy(self):
        print('all-deploy')
        self.k8s_deploy()
        self.ui_deploy()

    def reset(self):
        print('reset')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--deploy', required=True, default=None, help="action to execute. select one from 'k8s', 'ui', 'airflow' and 'all'")
    args = parser.parse_args()

    management = Management()
    if args.action == 'k8s':
        management.k8s_deploy()
    elif args.action == 'ui':
        management.ui_deploy()
    elif args.action == 'all':
        management.all_deploy()
    elif args.action == 'reset':
        management.reset()
    else:
        print("Error parameter for ACTION")
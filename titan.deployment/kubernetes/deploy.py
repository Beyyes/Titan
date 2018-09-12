from configReader import ConfigReader
from remoteTool import RemoteTool
from os import listdir
from os.path import isfile, join
import subprocess
import argparse
import logging
import commands

class Deployment:

    def __init__(self, filePath=""):
        configReader = ConfigReader(filePath)
        self.hosts = configReader.parse()
        self.remoteTool = RemoteTool()
        self.script_folder = "init_k8s_scrpts"
        self.join_cmd = ""

    def pack_script(self):
        tar_cmd = "tar -cvf init_ script.tar script"
        subprocess.check_call(tar_cmd, shell=True)

    def transferScripts(self, host, scripts_folder = ""):
        if scripts_folder == "":
            scripts_folder = "script"
        scripts = [f for f in listdir(scripts_folder) if isfile(join(scripts_folder, f))]
        dst_path = "/home/{0}/{1}".format(host.username, self.script_folder)
        # mkdir on remote host
        self.remoteTool.execute_cmd(host, "mkdir -p {0}".format(dst_path))

        for script in scripts:
            self.remoteTool.sftp_paramiko("{0}/{1}".format(scripts_folder, script), "{0}/{1}".format(dst_path, script),
                                          host)
        # set exec mode
        self.remoteTool.execute_cmd(host, "chmod +x {0}/*".format(dst_path))

    def deployMaster(self):
        host = self.hosts['master'][0]
        self.transferScripts(host)
        cmd = "cd /home/{0}/{1}/ && sudo ./init_master.sh {2}".format(host.username, self.script_folder, host.username)
        output = self.remoteTool.execute_cmd(host, cmd)
        self.join_cmd = self.extract_join_cmd(output['out'])
        cmd = "kubectl label nodes {0} k8s-master=true".format(host.hostname)
        print(cmd)
        output = self.remoteTool.execute_cmd(host, cmd)
        # we need store

        # clear(host)

    def deploySlaves(self):
        for host in self.hosts['slave']:
            self.transferScripts(host)
            prepare_cmd = "cd /home/{0}/{1}/ && sudo ./prepare_env.sh {2}".format(host.username, self.script_folder, host.username)
            print("Execute prepare script: " + prepare_cmd)
            self.remoteTool.execute_cmd(host, prepare_cmd)
            join_cmd = "sudo {0}".format(self.join_cmd)
            self.remoteTool.execute_cmd(host, join_cmd)

    def clear(self, host):
        dst_path = "/home/{0}/{1}".format(host.username, self.script_folder)
        self.remoteTool.execute_cmd(host, "rm -rf {0}".format(dst_path))

    def extract_join_cmd(self, content):
        lines = content.split('\r')
        for line in lines:
            if line.strip().startswith("kubeadm join"):
                return line.strip()

    def deploy(self):
        self.deployMaster()
        self.deploySlaves()
        print("\nSuccessfully deploy kubernetes on cluster")

    def reset_cluster(self):
        reset_cmd = "sudo kubeadm reset -f"
        for host in self.hosts['master']:
            self.remoteTool.execute_cmd(host, reset_cmd)
            clean_cmd = "cd /home/{0}/{1}/ && sudo ./reset_k8s.sh".format(host.username, self.script_folder)
            self.remoteTool.execute_cmd(host, clean_cmd)
        for host in self.hosts['slave']:
            self.remoteTool.execute_cmd(host, reset_cmd)
            clean_cmd = "cd /home/{0}/{1}/ && sudo ./reset_k8s.sh".format(host.username, self.script_folder)
            self.remoteTool.execute_cmd(host, clean_cmd)
        print("\nSuccessfully clear kubernetes on cluster")

    def add_node(self):
        token = commands.getoutput("sudo kubeadm token create")
        hash = commands.getoutput("openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | "
                                  "openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -hex | sed 's/^.* //'")

        # for host in self.hosts['slave']:
        #     join_cmd = "sudo kubeadm join {0}:6443 --token {1} " \
        #                "--discovery-token-ca-cert-hash sha256:{2}".format(hostip, token, hash)
        #     self.remoteTool.execute_cmd(host, join_cmd)
        #
        # print("\nSuccessfully add new node to kubernetes cluster")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', required=True, default=None, help="action to execute. select one from 'deploy' and 'reset'")
    args = parser.parse_args()

    deployment = Deployment()
    if args.action == 'deploy':
        deployment.deploy()
    elif args.action == 'reset':
        deployment.reset_cluster()
    elif args.action == 'add':
        deployment.add_node()
    else:
        print("Error parameter for ACTION")




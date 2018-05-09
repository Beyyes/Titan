
import paramiko

hosts = ['stcvl-001', 'stcvl-002', 'stcvl-003', 'stcvl-004', 'stcvl-005']


def execute(host, username, passwd, cmd):
    port = 22
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port, username=username, password=passwd)
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    stdin.write(passwd + '\n')
    stdin.flush()
    print("Executing the command on host [{0}]: {1}".format(host, cmd))
    for response_msg in stdout:
        print(response_msg.strip('\n'))
    ssh.close()
    return True


if __name__ == '__main__':
    cmd1 = "sudo kubeadm reset"
    cmd2 = "sudo kubeadm join 10.190.190.185:6443 --token 4y72i5.c2wvi3d23zxs8hxp --discovery-token-ca-cert-hash sha256:487823c70982c5c6d2ac4ce9161fc38792b4eff9ea76647a76b4b1087b22791a";
    for host in hosts:
        execute(host, "dladmin", "Pass_word", cmd2)
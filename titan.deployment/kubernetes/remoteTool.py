import paramiko
from configReader import *

class RemoteTool:
    def __init__(self):
        pass

    def execute_cmd(self, host, cmd):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host.ip, port=host.port, username=host.username, password=host.password)
        stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
        stdin.write(host.password + '\n')
        stdin.flush()
        print("Executing the command on host [{0}]: {1}".format(host.hostname, cmd))
        output = dict()
        output['out'] = ""
        output['err'] = ""
        for response_msg in stdout:
            output['out'] += response_msg.strip('\n')
            print(response_msg.strip('\n'))
        for response_msg in stderr:
            output['err'] += response_msg.strip('\n')
        ssh.close()
        print("Exec done.")
        return output

    def sftp_paramiko(self, src, dst, host):
        print("transfer file from {0} to {1}".format(src, dst))
        # Put the file to target Path.
        transport = paramiko.Transport((host.ip, host.port))
        transport.connect(username=host.username, password=host.password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(src, dst)
        sftp.close()

        transport.close()
        return True


if __name__ == '__main__':
    transport = paramiko.Transport(("spark-master", 22))
    transport.connect(username="dladmin", password="Pass_word")
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put("README.md", "README.md")
    sftp.close()

    transport.close()
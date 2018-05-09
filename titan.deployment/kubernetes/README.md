## Using Titan to Deploy Kubernetes

You can use Titan to deploy a Kubernetes cluster automatically. And here is the steps.

### Prerequirements

Before running the script, please ensure you have a python environment and `paramiko` and `pyyaml` has been installed. 

After install python. Using following commands to install `paramiko` and `pyyaml`

```
pip install paramiko
pip install pyyaml
```

### Prepare configuration file

The configuration file `cluster-config.yaml` illustrate the cluster information for kubernetes. It includes all the hosts
 used to deploy the cluster and specific configurations for each host. Here is the example of the configuration file.
 
```
host-list:
- hostname: host1         # hostname of the host
  ip: 192.168.1.100       # ip of the host
  port: 22
  role: master            # role in k8s cluster. select one in "master" and "slave"
  username: dladmin       # username for current host
  password: Pass_word     # password for current host

- hostname: host2
  ip: 192.168.1.101
  port: 22
  role: slave             # set to "slave" if current host is one of the workers in k8s cluster
  username: dladmin
  password: Pass_word
    
- hostname: host3
  ip: 192.168.1.102
  port: 22
  role: slave
  username: dladmin
  password: Pass_word
    
  # if you have more hosts, add them one by one
  # ...

```

### Deploy

Put the configuration file in `./titan.deployment/kubernetes/cluster-config.yaml`. Then start deploy using following command

```bash
python ./deploy.py -a deploy
```

### Reset the cluster

If the configuration file is fine. Just run the command
```bash
python ./deploy.py -a reset
```
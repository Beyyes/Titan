import yaml


class ConfigReader:

    def __init__(self):
        self.config_file_path = 'cluster-config.yaml'
        self.hosts = dict()
        self.hosts['master'] = []
        self.hosts['slave'] = []
        pass

    def parse(self):
        with open(self.config_file_path, "r") as f:
            raw_config = yaml.load(f)

        for host in raw_config['host-list']:
            self.hosts[host['role']].append(HostConfig(host))
        return self.hosts

class HostConfig:

    def __init__(self, host):
        self.config_dict = host
        self.hostname = host['hostname']
        self.ip = host['ip']
        self.port = host['port']
        self.username = host['username']
        self.password = host['password']
        self.role = host['role']

    def __repr__(self):
        return str(self.config_dict)


if __name__ == '__main__':
    configReader = ConfigReader()
    hosts = configReader.parse()
    print(hosts)
FROM ubuntu:16.04

RUN apt-get -y update && \
    apt-get -y install \
      nano \
      vim \
      joe \
      wget \
      curl \
      jq \
      gawk \
      psmisc \
      python \
      python-yaml \
      python-jinja2 \
      python-paramiko \
      python-dev \
      python-pip \
      python-mysqldb \
      openjdk-8-jre \
      openjdk-8-jdk \
      openssh-server \
      openssh-client \
      git \
      mysql-server \
      libmysqlclient-dev \
      mysql-client && \
      mkdir -p /airflow &&
      pip install mysqlclient \
      airflow[mysql,crypto,password]

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV AIRFLOW_HOME=/airflow

COPY airflow-setup.sh /airflow-setup.sh
CMD ['airflow-setup.sh']
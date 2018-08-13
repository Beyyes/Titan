FROM ubuntu:16.04

ENV AIRFLOW_HOME=/airflow

RUN apt-get -y update && \
    apt-get -y install \
      nano \
      vim \
      joe \
      wget \
      curl \
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
      git \&&
      mkdir -p /airflow

# chmod +x airflow-setup.sh

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

COPY ./airflow-setup.sh /airflow-setup.sh
CMD ['airflow-setup.sh']
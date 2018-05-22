# Setup subtree

+ Add Connection
  * git remote add airflow https://github.com/iamtouchskyer/titan-airflow.git
  * git remote add grafana https://github.com/iamtouchskyer/titan-grafana.git
+ Enlist
  * git subtree add --prefix=projects/airflow airflow master
  * git subtree add --prefix=projects/grafana grafana master
+ Pull
  * git subtree pull --prefix=projects/airflow airflow master
  * git subtree pull --prefix=projects/grafana grafana master
+ Push
  * git subtree push --prefix=projects/airflow airflow master
  * git subtree push --prefix=projects/grafana grafana master


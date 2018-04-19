/usr/local/spark/bin/spark-submit --class "DataPreprocess" --master yarn --deploy-mode cluster ./target/scala-2.11/data-preprocess_2.11-1.0.jar "hdfs://spark-master:9000/small_data/" "hdfs://spark-master:9000/small_output/" --driver-memory 3g --executor-memory 3g --executor-cores 2 --num-executors 4

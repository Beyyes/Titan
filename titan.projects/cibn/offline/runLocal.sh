install/sbt package
rm -r ./output2/*
/usr/local/spark/bin/spark-submit --class "DataPreprocess" ./target/scala-2.11/data-preprocess_2.11-1.0.jar "file:///home/qiozou/src/spark-video-bp/spark-video/data1/" "file:////home/qiozou/src/spark-video-bp/spark-video/output2/"

import scala.util.control._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import utils.{ DataLoader }
import videoCategoryIndexWeight.VideoCategoryIndexWeight
import videoCategoryVector.VideoCategoryVector
import vidEncoding.VidEncoding
import videoImpression.VideoImpression
import vidTrainData.VidTrainData
import vidRec.VidRec
import vidPreData.VidPreData
import scala.sys.process._


object DataPreprocess {
	def Process(inputFolder: String, outputFolder:String) {
		//set environment
		val spark =SparkSession
			.builder
			.appName("data preprocess")
			.getOrCreate()
		val sc = spark.sparkContext
		val sqlContext = new SQLContext(sc)
                sqlContext.setConf("spark.sql.shuffle.partitions", "1000")
		spark.conf.set("spark.sql.broadcastTimeout",  36000)
		import sqlContext.implicits._
		import spark.implicits._

		//Set variable
		val rawVideoDataFile = String.format("%svideo_info_full_large.txt", inputFolder)
		val rawPlayEventsFullFile = String.format("%sraw_play_events_full_large.csv", inputFolder)
		//Load Input data
		val rawVideoDataSchemaString = "vid,vname,videotype,series,updatenum,duration,storyplot,issueyear,studio,language,category,area,director,star,actor,dubbing,showhost,showguest,singer,presenter,taginfo,score,award,goodsname,goodsid,goodsbrand,goodsprice"
		val rawVideoDataDF = DataLoader.LoadDataToDF(rawVideoDataFile, rawVideoDataSchemaString, spark, sqlContext)
		val videoDataDF = rawVideoDataDF.select("videotype", "vid", "category", "taginfo", "vname")
		//--------------------------------VideoCategoryIndexWeight------------------
		val categoryIndexWeight = VideoCategoryIndexWeight.Process(rawVideoDataDF, spark, sqlContext)
		//categoryIndexWeight.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%scategory_index_weight", outputFolder))
		//--------------------------------VideoCategoryVector------------------
		
		val video_vector_dist_df = VideoCategoryVector.Process(videoDataDF, categoryIndexWeight, spark, sqlContext)
		//video_vector_dist_df.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%svideo_vector_dist_df", outputFolder))
		//--------------------------------VidEncoding------------------		
		val (vidEncodingIdDF, vidHistEncoding, maxEncoding) = VidEncoding.Process(videoDataDF, spark, sqlContext)
		//vidEncodingIdDF.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%svidEncodingIdDF", outputFolder))
		//vidHistEncoding.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%svidHistEncoding", outputFolder))
		//maxEncoding.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%smaxEncoding", outputFolder))
		//--------------------------------VidImpression------------------	
		val (hidVidTrainImpressionLabel, hidVidValImpressionLabel, hidVidTestImpressionLabel, vidImpression, vidHidHourDF, vidHidHourFinal) = VideoImpression.Process(rawVideoDataDF, rawPlayEventsFullFile, spark, sqlContext)	
		//hidVidTrainImpressionLabel.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%shidVidTrainImpressionLabel", outputFolder))
		//hidVidValImpressionLabel.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%shidVidValImpressionLabel", outputFolder))
		//hidVidTestImpressionLabel.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%shidVidTestImpressionLabel", outputFolder))
		//vidImpression.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%svidImpression", outputFolder))
		//vidHidHourDF.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%svidHidHourDF", outputFolder))
		//vidHidHourFinal.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%svidHidHourFinal", outputFolder))
		//--------------------------------VidTrainData-----------------
		val (hidVidHistEncoding, trainData, valData, testData) = VidTrainData.Process(vidEncodingIdDF, vidHistEncoding, hidVidTrainImpressionLabel,hidVidValImpressionLabel, hidVidTestImpressionLabel, spark, sqlContext)
		//hidVidHistEncoding.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%shidVidHistEncoding", outputFolder))
		trainData.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%strain_data", outputFolder))
		valData.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%sval_data", outputFolder))
		testData.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%stest_data", outputFolder))
		//--------------------------------VidRec-----------------        
        	val vid_rec_step2 = VidRec.Process(vidHidHourFinal, vidImpression, video_vector_dist_df, spark, sqlContext)
		//vid_rec_step2.repartition(1).write.format("com.databricks.spark.csv").mode("append").save(String.format("%svid_rec_step2", outputFolder))
		//--------------------------------VidPreData-----------------
		val preData = VidPreData.Process(vidEncodingIdDF, hidVidHistEncoding, vid_rec_step2, spark, sqlContext)  
		preData.write.format("com.databricks.spark.csv").mode("append").save(String.format("%spre_data", outputFolder))
		
		//val indexWeightFileHDFS = "hdfs://spark-master:9000/video/"
		//var isResult = Seq("/usr/local/hadoop/bin/hdfs","dfs","-put", outputFolder, indexWeightFileHDFS).!!
		spark.stop()
   	}
   //Main
   	def main(args: Array[String]) {		
		Process(args(0),args(1))
   	}
}

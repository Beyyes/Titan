package videoImpression
//Video Impression
//Input File: raw_video_data.txt raw_play_events_full.txt
//Output File: train/val/test/sel
import java.sql.{Timestamp}
import utils.{ DataLoader, Helper }
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{LongType, TimestampType}

object VideoImpression {
    private val impression_threshold_str = "60000" //60s
	private val impression_threshold = impression_threshold_str.toLong
	private val cutoff_time_train_start = "2018-01-01 00:00:00.000"
	private val cutoff_time_train_end = "2018-01-18 00:00:00.000"
	private val cutoff_time_val = "2018-01-21 00:00:00.000"
	private val cutoff_time_test = "2018-01-23 00:00:00.000"
	private val cutoff_time_sel_start = "2018-01-01 00:00:00.000"
	private val cutoff_time_sel_end = "2018-01-18 00:00:00.000"

    def filterByTime(inputDF: DataFrame, startTime: String, endTime: String, columns: Array[String]):DataFrame = {
        val cusCol = columns :+ "servertime"
        val outputDF = inputDF
            .select(cusCol.map(col(_)): _*)
            .withColumn("temptime", unix_timestamp(col("servertime"), "MM/dd/yyyy HH:mm:ss"))
            .filter((unix_timestamp(lit(startTime)) < col("temptime")))
            .filter((unix_timestamp(lit(endTime)) > col("temptime")))
        outputDF
    }

    def countIf(inputDF: DataFrame, columns: Array[String]):DataFrame = {
        //.select(col("hid"), col("vid"), col("realplaytime"), col("temp").cast(TimestampType))
        //.withColumnRenamed("temp", "servertime")
        val cusCol =  columns:+ "temp"
        var coder: (Long => Int)=(arg:Long) => { if (arg > impression_threshold) 1 else 0 }
        var sqlfunc = udf(coder)
        val outputDF = inputDF
            .withColumn("temp", sqlfunc(col("realplaytime")))
            .select(cusCol.map(col(_)): _*)
            .groupBy(columns.map(col(_)): _*)
            .agg(Map("temp" -> "sum"))
            .withColumnRenamed("sum(temp)", "impression_count")
        outputDF
    }

    def getLabel(inputDF: DataFrame, startTime: String, endTime: String):DataFrame = {
        var coder: (Int => Int)=(arg:Int) => { if (arg > 0) 1 else 0 }
        var sqlfunc = udf(coder)
        val cusCol = Array("hid", "vid")
        val cusFilterCol = cusCol :+ "realplaytime"
        val tempDF = filterByTime(inputDF, startTime, endTime, cusFilterCol)
        val outputDF = countIf(tempDF, cusCol)
            .withColumn("label", sqlfunc(col("impression_count")))
            .sort("vid")

        outputDF
    }

    def selectImpression(inputDF: DataFrame, startTime: String, endTime: String):DataFrame = {
        val cusCol = Array("videotype", "vid")
        val cusFilterCol = cusCol :+ "realplaytime"
        val tempDF = filterByTime(inputDF, startTime, endTime, cusFilterCol)
        val outputDF = countIf(tempDF, cusCol)
            .withColumnRenamed("impression_count", "vid_impression_count")

        outputDF 
    }

    def Process(rawVideoDataDF: DataFrame, rawPlayEventsFullFile: String, spark: SparkSession, sqlContext:SQLContext):Tuple6[DataFrame,DataFrame,DataFrame,DataFrame,DataFrame,DataFrame] = {
        //Load Input data
        val videoVidDF = rawVideoDataDF.select("videotype", "vid").dropDuplicates()

        //basic filter
        videoVidDF.registerTempTable("videoVid")
        rawVideoDataDF.registerTempTable("videoData")
        val playEventsDF = spark.read.format("csv").option("header", "true").load(rawPlayEventsFullFile)
        val filteredPalyEventsDF = playEventsDF.filter("vid != '0' AND cmdid == '13' AND playevent == 'end' AND errorcode == 'NULL' AND realplaytime != '0'")
        filteredPalyEventsDF.registerTempTable("filterPlayEvents")
        
        val hidVidImpressionPre = sqlContext.sql("""SELECT servertime,
		           hid,
                   realplaytime,
		           a.vid,
		           b.videotype,
                   starttime,
                   endtime
		    FROM filterPlayEvents AS a
		         INNER JOIN
		             videoVid AS b
		         ON a.vid == b.vid""")

        val hidVidImpressionRDD = hidVidImpressionPre.rdd.map(row => {
            val originalRow = row.toSeq.toList
            val servertime = originalRow(0).toString
            val day = servertime.split(" ")(0)
            val timeUnit = servertime.split(" ")(2)
            var hour = servertime.split(" ")(1).split(":")(0)
            if (timeUnit == "PM") {
                hour = (hour.toInt + 12).toString;
            }
            val orignalPlayTime = originalRow(2).toString
            val startTime = originalRow(5).toString
            val endTime = originalRow(6).toString
            val realPlayTime = Helper.TransformRealPlayTime(orignalPlayTime, startTime, endTime)

            Row.fromSeq(originalRow :+ day :+ hour :+ realPlayTime)
        }).map(row => Row(row(0), row(7), row(8), row(1), row(3), row(4), row(9)))
       	
        val hidVidImpressionString = "servertime,day,hour,hid,vid,videotype,realplaytime"
        val hidVidImpressionTypeMap = Map("realplaytime" -> LongType)
        val hidVidImpressionSchema = DataLoader.getSchemaByString(hidVidImpressionString, hidVidImpressionTypeMap)
        val hidVidImpressionDF = sqlContext.createDataFrame(hidVidImpressionRDD, hidVidImpressionSchema)
        //start===========================train,val,test label=======================
        //issue: duplicate record may increase impression_count
        //train impression
        val hidVidTrainImpressionLabel = getLabel(hidVidImpressionDF, cutoff_time_train_start, cutoff_time_train_end)
        //hidVidTrainImpressionLabel.repartition(1).write.format("com.databricks.spark.csv").save(String.format("%strain_hid_impression_label", outputFolder))
        val hidVidValImpressionLabel = getLabel(hidVidImpressionDF, cutoff_time_train_end, cutoff_time_val)
        //hidVidValImpressionLabel.repartition(1).write.format("com.databricks.spark.csv").save(String.format("%sval_hid_impression_label", outputFolder))
        val hidVidTestImpressionLabel = getLabel(hidVidImpressionDF, cutoff_time_val, cutoff_time_test)
        //hidVidTestImpressionLabel.repartition(1).write.format("com.databricks.spark.csv").save(String.format("%stest_hid_impression_label", outputFolder))
        //end===========================train,val,test label=======================
        //selection impression
        var vidImpression = selectImpression(hidVidImpressionDF, cutoff_time_sel_start, cutoff_time_sel_end)
        val videotypeImpression = vidImpression.groupBy("videotype")
            .agg(Map("vid_impression_count" -> "sum"))
            .withColumnRenamed("sum(vid_impression_count)", "total_impression_count")
        vidImpression.registerTempTable("vidImpression")
        videotypeImpression.registerTempTable("videotypeImpression")
        vidImpression = sqlContext.sql("""SELECT a. *,
                b.total_impression_count,
                (vid_impression_count * 1.0 / b.total_impression_count) AS impression_score
            FROM vidImpression AS a
                INNER JOIN
                    videotypeImpression AS b
                ON a.videotype == b.videotype""")
        
        val cusCol = Array("vid", "hid", "hour", "videotype")

        val tempHidVidImpressionDF = filterByTime(hidVidImpressionDF, cutoff_time_sel_start, cutoff_time_sel_end, cusCol :+ "realplaytime")
        var vidHidHourDF = countIf(tempHidVidImpressionDF, cusCol)
        vidHidHourDF = vidHidHourDF.sort(desc("impression_count"))
        vidHidHourDF.registerTempTable("vidHidHour")

        var vidHidHourFinal = sqlContext.sql("""SELECT vid,hid,hour,videotype,impression_count,
            ROW_NUMBER() OVER(PARTITION BY hid,hour,videotype ORDER BY impression_count DESC) As index
            FROM vidHidHour""")
         vidHidHourFinal = vidHidHourFinal.filter("index < 11").select("vid","hid","hour","videotype","impression_count")
        //vidHidHourFinal.repartition(1).write.format("com.databricks.spark.csv").save(String.format("%svid_hid_hour",outputFolder))
        (hidVidTrainImpressionLabel, hidVidValImpressionLabel, hidVidTestImpressionLabel, vidImpression, vidHidHourDF, vidHidHourFinal)
    }  
}

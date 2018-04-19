package vidTrainData
//Vid Encoding
//Input File: raw_video_data.txt;
//Output File: vid_encoding.txt;vid_hist_encoding.txt;max_encoding.txt
import scala.util.control._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import utils.{ DataLoader }

object VidTrainData {
    def generateData(hidImpLabel: DataFrame, sqlContext: SQLContext):DataFrame={
        hidImpLabel.registerTempTable("hid_imp_label")
        var outputData = sqlContext.sql("""SELECT a.label,
                a.vid,
                a.hid,
                b.encoding AS ve,
                e.encoding AS hve
            FROM hid_imp_label AS a
                INNER JOIN
                    vid_encoding As b
                ON a.vid == b.vid
                INNER JOIN
                    hid_vid_hist_encoding As e
                ON a.hid == e.hid""")
        var coder3 = (label: String,ve: String,hve: String,hid: String,vid: String) => String.format("%s %s %s%%%s_%s", label, ve, hve, hid, vid)
        val sqlfunc = udf(coder3)
        outputData = outputData.withColumn("r", sqlfunc(col("label"), col("ve"), col("hve"), col("hid"), col("vid"))).select("r")
        outputData
    }
	def Process(vidEncodingDF: DataFrame, vidHistEncodingDF: DataFrame,trainImpressionDF: DataFrame, valImpressionDF: DataFrame, testImpressionDF: DataFrame, spark: SparkSession, sqlContext:SQLContext): Tuple4[DataFrame,DataFrame,DataFrame,DataFrame] = {
        //define string func
        val coder1 = (encodingtype: String,encodingid: String) => String.format("%s:%s:1", encodingtype, encodingid)
        val sqlfunc1 = udf(coder1)
		
        val vidEncoding = vidEncodingDF.withColumn("encoding", sqlfunc1(col("encodingtype"), col("encodingid")))
            .select("vid", "encoding")
        trainImpressionDF.registerTempTable("hid_imp_label")
        vidHistEncodingDF.registerTempTable("vid_hist_encoding")
        var hidVidHistEncoding = sqlContext.sql("""SELECT hid,
                encodingtype,
                encodingid,
                a.impression_count,
                a.vid AS vid
            FROM hid_imp_label AS a
                INNER JOIN
                    vid_hist_encoding AS b
                ON a.vid == b.vid""")
        val coder2= (encodingtype: String,encodingid: String,impression_count: String) => String.format("%s:%s:%s", encodingtype, encodingid, impression_count)
        val sqlfunc2 = udf(coder2)
        val coder3= ()
        hidVidHistEncoding = hidVidHistEncoding.withColumn("encoding", sqlfunc2(col("encodingtype"), col("encodingid"), col("impression_count")))
            .select("hid", "encoding")
        hidVidHistEncoding.registerTempTable("hid_vid_hist_encoding_pre")
        hidVidHistEncoding = sqlContext.sql("""SELECT hid,concat_ws(' ', collect_set(encoding)) as encoding
            FROM hid_vid_hist_encoding_pre group by hid""").sort("hid")
        //hidVidHistEncoding.repartition(1).write.format("com.databricks.spark.csv").save(String.format("%shid_vid_hist_encoding", outputFolder))
        hidVidHistEncoding.registerTempTable("hid_vid_hist_encoding")
        vidEncoding.registerTempTable("vid_encoding")
        
        //train data
        val trainData = generateData(trainImpressionDF, sqlContext)
        //trainData.repartition(1).write.format("com.databricks.spark.csv").save(String.format("%strain_data", outputFolder))
        //val data
        val valData = generateData(valImpressionDF, sqlContext)
        //valData.repartition(1).write.format("com.databricks.spark.csv").save(String.format("%sval_data", outputFolder))
        //test data
        val testData = generateData(testImpressionDF, sqlContext)
        //testData.repartition(1).write.format("com.databricks.spark.csv").save(String.format("%stest_data", outputFolder))
        (hidVidHistEncoding, trainData, valData, testData)
   	}
}

package vidPreData
//Vid predata
//Input File: raw_video_data.txt;
//Output File: vid_encoding.txt;vid_hist_encoding.txt;max_encoding.txt
import scala.util.control._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._

object VidPreData {
    def Process(vidPreDataDF: DataFrame, vidHistEncodingDF: DataFrame, vidRecStep2DFPre: DataFrame, spark: SparkSession, sqlContext:SQLContext):DataFrame = {        //define string func
        var coder1 = (arg1 : String, arg2 : String) => String.format("%s:%s:1", arg1, arg2)
        var sqlfunc = udf(coder1)       
        val vidPreData = vidPreDataDF.withColumn("encoding", sqlfunc(col("encodingtype"), col("encodingid")))
            .select("vid", "encoding")
        //Pre Data
        vidPreData.registerTempTable("vid_encoding")
        vidHistEncodingDF.registerTempTable("hid_vid_hist_encoding")
	vidRecStep2DFPre.registerTempTable("vid_rec_pre")
        val vidRecStep2DF = sqlContext.sql("""SELECT hid,vid,concat_ws(';', collect_set(hour)) as hours
            FROM vid_rec_pre group by hid,vid""")
        vidRecStep2DF.registerTempTable("vid_rec")
	var preData = sqlContext.sql("""SELECT 0 AS label,
                a.vid,
                a.hid,
                a.hours,
                b.encoding AS ve,
                e.encoding AS hve
            FROM vid_rec AS a
                INNER JOIN
                    vid_encoding AS b
                ON a.vid == b.vid
                INNER JOIN
                    hid_vid_hist_encoding AS e
                ON a.hid == e.hid""")
        var coder2 = (col1: String,col2: String,col3: String) => String.format("%s %s %s", col1, col2, col3)
        var sqlfunc2 = udf(coder2)

        var coder3 = (col1: String,col2: String,col3: String,col4: String) => String.format("%s%%%s_%s_%s", col1, col2, col3, col4)
        var sqlfun3 = udf(coder3)


        preData = preData.withColumn("r", sqlfunc2(col("label"), col("ve"), col("hve")))
            .withColumn("tag", sqlfun3(col("r"), col("hid"), col("vid"), col("hours")))
            .sort("tag")
	    .select("tag")

       // var cnt = preData.count()
       // println(s"final count:${cnt}")
        preData
       }
}

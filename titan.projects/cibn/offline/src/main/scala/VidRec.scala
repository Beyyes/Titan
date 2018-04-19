package vidRec
//Vid Rec
//Input File: raw_video_data.txt;
//Output File: vid_encoding.txt;vid_hist_encoding.txt;max_encoding.txt
import scala.util.control._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{LongType, DoubleType}
import org.apache.spark.sql.functions._

object VidRec {
    def Process(vidHidHourDF: DataFrame, vidImpressionDF: DataFrame, vidVectorDistDF: DataFrame, spark: SparkSession, sqlContext:SQLContext):DataFrame = {        
        val vidImpressionScore = vidImpressionDF.select("vid", "impression_score")        
        vidVectorDistDF.registerTempTable("vid_vector_dist")
        vidImpressionScore.registerTempTable("vid_impression_score")
        var vidDistImpression = sqlContext.sql("""SELECT a.leftvid as vid, a.rightvid as vid_rec, a.dist, b.impression_score FROM vid_vector_dist AS a INNER JOIN vid_impression_score AS b ON a.rightvid == b.vid""")
        
        //var output_impression_score = vidDistImpression.sort(desc("dist"), desc("impression_score"))

        var vidRankscore = vidDistImpression.select(col("vid"), col("vid_rec"), (col("impression_score") * col("dist")).as("rankscore"))

        vidRankscore.registerTempTable("vid_rank_score")
        var vidTopn_temp = sqlContext.sql("""SELECT vid, vid_rec, rankscore,
            ROW_NUMBER() OVER(PARTITION BY vid ORDER BY rankscore DESC) As index
            FROM vid_rank_score""")
        var vidTopn = vidTopn_temp.filter("index < 101").select("vid","vid_rec","rankscore")

        val vidHidHour = vidHidHourDF.select("vid", "hid", "hour", "videotype")

        vidHidHour.registerTempTable("vid_hid_hour")
        vidTopn.registerTempTable("vid_topn")
        var vidRec = sqlContext.sql("""SELECT a.hid, a.hour, a.vid, a.videotype, b.vid_rec, b.rankscore FROM vid_hid_hour AS a INNER JOIN vid_topn AS b ON a.vid == b.vid""")

        var oldVidList = vidRec.filter(col("videotype").equalTo("movie")).select("hid", "hour", "vid")
        var vidHashset = vidRec.select("hid", "hour", "vid_rec", "rankscore")
        vidHashset.registerTempTable("vid_hash_set")
        oldVidList.registerTempTable("old_vid_list")
        var vidList = sqlContext.sql("""SELECT t1.hid as hid, t1.hour as hour, vid_rec as vid, rankscore FROM vid_hash_set t1 LEFT JOIN old_vid_list t2 ON t2.vid = t1.vid_rec and t2.hid = t1.hid and t2.hour = t1.hour WHERE t2.vid IS NULL""")
        
        vidList.registerTempTable("vid_list")
        var vidlist_temp = sqlContext.sql("""SELECT hid, hour, vid, rankscore,
            ROW_NUMBER() OVER(PARTITION BY hid, hour ORDER BY rankscore DESC) As index
            FROM vid_list""")
        var vid_list = vidlist_temp.filter("index < 101").select("hid","hour","vid")
	var vid_rec_step2_output = vid_list.sort("hid", "hour")
        vid_rec_step2_output
       }
}

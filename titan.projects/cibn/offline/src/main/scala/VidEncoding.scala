package vidEncoding
//Vid Encoding
//Input File: raw_video_data.txt;
//Output File: vid_encoding.txt;vid_hist_encoding.txt;max_encoding.txt
import scala.io.Source
import scala.util.control._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, LongType}
import utils.{ DataLoader }
import org.apache.spark.sql.functions._

object VidEncoding {
	def Process(videoDataDF: DataFrame, spark: SparkSession, sqlContext:SQLContext): Tuple3[DataFrame,DataFrame,DataFrame] = {		
		val distinctVidDF = videoDataDF.select("vid").dropDuplicates().sort("vid");
		val vidEncodingIdRDD = distinctVidDF.rdd.zipWithIndex().map(row => {
			val originalRow = row._1.toSeq.toList
			val index = row._2 + 1
			Row.fromSeq(originalRow :+ 1 :+ index )
		}).map(row => Row(row(0), row(1), row(2)))
		val vidEncodingIdSchemaString = "vid,encodingtype,encodingid"
		val vidEncodingIdTypeMap = Map("encodingtype" -> IntegerType, "encodingid" -> LongType)
		val vidEncodingIdSchema = DataLoader.getSchemaByString(vidEncodingIdSchemaString, vidEncodingIdTypeMap)
		val vidEncodingIdDF = sqlContext.createDataFrame(vidEncodingIdRDD, vidEncodingIdSchema)
		// vidEncodingIdDF.repartition(1).write.format("com.databricks.spark.csv").save(vidEncodingFile)
		val vidEncodingIdResDF = vidEncodingIdDF // temp prolbem if = is reference
		vidEncodingIdDF.agg(Map("encodingid" -> "max")).withColumnRenamed("max(encodingid)", "max_encodingid")
		vidEncodingIdDF.registerTempTable("videncoding")
		var maxEncoding = sqlContext.sql("""SELECT MAX(encodingid) AS max_encodingid
    		FROM videncoding""")
		maxEncoding.registerTempTable("maxencoding")
		val vidHistEncoding = sqlContext.sql("""SELECT vid,
				2 AS encodingtype,
				encodingid + max_encodingid AS encodingid
			FROM videncoding
				CROSS JOIN
					maxencoding""")
		//vidHistEncoding.repartition(1).write.format("com.databricks.spark.csv").save(vidHistEncodingFile)
		vidHistEncoding.registerTempTable("vidhistencoding")
		maxEncoding = sqlContext.sql("""SELECT MAX(encodingid) AS encodingid
    		FROM vidhistencoding""")
		//maxEncoding.repartition(1).write.format("com.databricks.spark.csv").save(maxEncodingFile)
		(vidEncodingIdResDF, vidHistEncoding, maxEncoding)
   	}
}

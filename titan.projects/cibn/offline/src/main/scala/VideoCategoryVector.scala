package videoCategoryVector
//Video Category Vector
//Input File: category_index_weight.txt;raw_video_data.txt;
//Temp File: vid_vector.txt
//Output File: vid_vector_dist.txt
import scala.io.Source
import scala.util.control._
import java.util.Base64 
import java.io.File
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import utils.{ DataLoader, Helper }

object CategoryVectorProcessor {
	def GetVideoTypeMaxVectorLength(indexWeightDF: DataFrame): Map[String, Int] = {
		var vectorLengthDict:Map[String, Int] = Map() 
		var videotype="";
		val inputs = indexWeightDF.rdd.toLocalIterator.toArray
		for (input <- inputs) {
			val line = input.toString()
			if(line != null && !line.isEmpty()) {
				val tokens = line.replace("[", "").replace("]", "").split(",", 0);
				if(tokens(0) != videotype )
				{
					videotype = tokens(0)
					val vectorLength = tokens(2).toInt+1
					vectorLengthDict += (videotype -> vectorLength)
				}
			}
		}
		vectorLengthDict
	}

	def LoadCategoryIndexWeight(indexWeightDF: DataFrame): Map[String, Tuple2[Int, Byte]] = {
		var indexWeightDict:Map[String, Tuple2[Int, Byte]] = Map()
		val inputs = indexWeightDF.rdd.toLocalIterator.toArray
		for (input <- inputs) {
			val line = input.toString()
			if(line != null && !line.isEmpty()) {
				val tokens = line.replace("[", "").replace("]", "").split(",", 0);
				val index = tokens(2).toInt
				val weight = tokens(3).toByte
				val key = "%s_%s".format(tokens(0), tokens(1))
				val indexWeight = new Tuple2(index, weight)
				indexWeightDict += (key -> indexWeight)
			}
		}
		indexWeightDict
	}

	def Process(videoDataDF:DataFrame, indexWeightDF:DataFrame, sqlContext:SQLContext): DataFrame = {

		val indexWeightDict = LoadCategoryIndexWeight(indexWeightDF)
		val videotypeVectorLengthDict = GetVideoTypeMaxVectorLength(indexWeightDF)

		val videoDataRDD = videoDataDF.rdd.map(row => {
			val originalRow = row.toSeq.toList
			val videotype = originalRow(0).toString
			
			var category = ""
			if (originalRow(2) != null) {
				category = originalRow(2).toString;
			}
			val taginfo = originalRow(3).toString
			var vname = originalRow(4).toString

			if (vname != "theatre") {
				vname = null
			}
			val categoryList = Helper.GetCategoryList(category, taginfo, vname)
			val vectorLength = videotypeVectorLengthDict.getOrElse(videotype, 0)
			val vector = Array.fill[Byte](vectorLength)(0)
			for(c <- categoryList)
			{
				val key = "%s_%s".format(videotype, c)				
				var indexWeight = indexWeightDict.getOrElse(key, null)		    
				if (indexWeight != null) {
					vector(indexWeight._1) = (vector(indexWeight._1).toInt + indexWeight._2.toInt).toByte
				}
			}
			
			val base64vector = Base64.getEncoder.encodeToString(vector)
			Row.fromSeq(originalRow :+ base64vector)
		}).map(row => Row(row(0), row(1), row(2), row(3), row(4), row(5)))
		val videoDataSchemaString = "videotype,vid,category,taginfo,vname,base64vector"
		val videoDataSchema = DataLoader.getSchemaByString(videoDataSchemaString)
		val videoDataRes = sqlContext.createDataFrame(videoDataRDD, videoDataSchema)
		videoDataRes
	}
}

object VideoCategoryVector {	
	def Process(videoDataDF: DataFrame, indexWeightDF: DataFrame, spark: SparkSession, sqlContext: SQLContext):DataFrame = {		
		var video_data = CategoryVectorProcessor.Process(videoDataDF, indexWeightDF, sqlContext)
		video_data = video_data.select("videotype", "vid", "base64vector").dropDuplicates()
		// output temp file
		video_data.registerTempTable("videodata")
		val video_join_vector = sqlContext.sql("""SELECT a.vid AS vid_left,
			a.base64vector AS base64vector_left,
			b.vid AS vid_right,
			b.base64vector AS base64vector_right
		FROM videodata AS a
			INNER JOIN
				videodata AS b
			ON a.videotype == b.videotype""")
		
		val distSchemaString = "leftvid,rightvid,dist"
		val distSchema = DataLoader.getSchemaByString(distSchemaString)
		val video_vector_dist = video_join_vector.rdd.map(row => {
			val originalRow = row.toSeq.toList
			val base64vector_left = originalRow(1).toString 
			val base64vector_right = originalRow(3).toString
			val dist = Helper.GetVectorDist(base64vector_left, base64vector_right)
			Row.fromSeq(originalRow :+ dist)
		}).map(row => Row(row(0), row(2), row(4).toString))

		var video_vector_dist_df = sqlContext.createDataFrame(video_vector_dist, distSchema)
		video_vector_dist_df = video_vector_dist_df.dropDuplicates()
		video_vector_dist_df	
   	}    
}

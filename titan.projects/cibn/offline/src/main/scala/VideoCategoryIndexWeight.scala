package videoCategoryIndexWeight
//Video Category Index Weight
//Input File: video_info_full.txt
//Temp File: category_item_count.txt
//Output File: category_index_weight.txt
import utils.{ DataLoader, Helper }
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType}

object VideoCategoryIndexWeight {
    def GetCategoryWeight(category: String):Int = {
        var res = 2
        if (category == "亲子" || category == "越剧" || category == "京剧" || category == "少儿") {
            res = 4
        }
        res
    }
    // category split processor
    def Process(rawVideoDataDF: DataFrame, spark: SparkSession, sqlContext: SQLContext):DataFrame = {        //Load Input data
        val videoTypeDF = rawVideoDataDF.select("videotype", "vname", "category", "taginfo")
        //calculate category item count
        val categoryItemRdd = videoTypeDF.rdd.map(row => {
            val originalRow = row.toSeq.toList
            val videotype = originalRow(0).toString
            var vname = originalRow(1).toString
            val category = originalRow(2).toString
            val taginfo = originalRow(3).toString            

            if (vname != "theatre") {
                vname = null
            }
            val categoryList = Helper.GetCategoryList(category, taginfo, vname)
            
            val res = categoryList.map(c => Row.fromSeq(originalRow :+ c))        
            res  
        }).flatMap(row => row).map(row => Row(row(0), row(2), row(3), row(4)))

        val categoryItemSchemaString = "videotype,category,taginfo,category_item"
	    val categoryItemSchema = DataLoader.getSchemaByString(categoryItemSchemaString)
        val categoryItemDF = sqlContext.createDataFrame(categoryItemRdd, categoryItemSchema)
        val categoryItemCount = categoryItemDF.select("videotype", "category_item")
            .withColumnRenamed("category_item", "category")
            .groupBy("videotype", "category")
            .agg(Map("category" -> "count"))
            .withColumnRenamed("count(category)", "count")
            .filter("count > 1") //drop trail category
        // temp output
        // categoryItemCount.repartition(1).write.format("com.databricks.spark.csv").save(categoryItemCountFile)
        //calculate index weight
        categoryItemCount.registerTempTable("categoryItemCount")
        val categoryItemCountWithIndex = sqlContext.sql("""SELECT videotype,
            category,
            count,
            ROW_NUMBER() OVER(PARTITION BY videotype ORDER BY count DESC)
            AS index FROM categoryItemCount""")
        val categoryIndexWeightRDD = categoryItemCountWithIndex.rdd.map(row => {
            val originalRow = row.toSeq.toList
            val category = originalRow(1).toString
            val index = originalRow(3).toString.toInt - 1
            val weight = GetCategoryWeight(category)
            
            Row.fromSeq(originalRow :+ index :+ weight)            
        }).map(row => Row(row(0), row(1), row(4), row(5)))
        val categoryIndexWeightString = "videotype,category,index,weight"
        val categoryIndexWeightTypeMap = Map("index" -> IntegerType, "weight" -> IntegerType)
        val categoryIndexWeightSchema = DataLoader.getSchemaByString(categoryIndexWeightString, categoryIndexWeightTypeMap)
        var categoryIndexWeightDF = sqlContext.createDataFrame(categoryIndexWeightRDD, categoryIndexWeightSchema)
        categoryIndexWeightDF = categoryIndexWeightDF.sort(col("videotype"), col("index").desc)
        categoryIndexWeightDF
    }  
}

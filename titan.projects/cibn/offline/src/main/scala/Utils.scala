package utils

import java.util.Base64 
import java.util.{Date}
import java.text.SimpleDateFormat
import scala.util.control._
import scala.collection.mutable.ListBuffer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DataType}

object DataLoader {
    def getSchemaByString(schemaString: String, typesMap: Map[String, DataType] = Map()): StructType = {
        val fields = schemaString.split(",").map(fieldName => StructField(fieldName, typesMap.getOrElse(fieldName, StringType), nullable=true))
	    val schema = StructType(fields)
        schema
    }

    def LoadDataToDF(dataFile: String, schemaString: String, spark: SparkSession, sqlContext: SQLContext, typesMap: Map[String, DataType] = Map()):DataFrame = {
        //set environment
        //generate schema
        var colNum = schemaString.split(",").length
        var schema = getSchemaByString(schemaString, typesMap)

        //load data
        val sc = spark.sparkContext
        val data = sc.textFile(dataFile)
        val dataRDD = data.map(row => {
            val items = row.split("\\t")
            if (items.length < colNum) {
                colNum = items.length
            }
            var itemArray: Array[String] = new Array[String](colNum)
            for (i <- 0 to colNum-1) {
                itemArray(i) = items(i)
            }
            Row(itemArray: _*)
        })
        val input = sqlContext.createDataFrame(dataRDD, schema)
        input        
    }
}

object Helper {
   private val TheatreCategoryList = Array("越剧", "京剧", "豫剧", "儿童剧", "吕剧", "昆剧", "昆曲", "曲剧", "沪剧", "河南坠子", "淮剧", "琴书", "相声", "绍剧", "评剧", "话剧", "音乐剧", "黄梅戏" )
   private val Separators = "\\||\\s|,"

    def TransformRealPlayTime(realplaytime: String, starttime: String, endtime: String):Long = {
        var res:Long = 1
        val formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        if (realplaytime.length() > 0)
        {
            var st = formatter.parse(starttime);
            var et = formatter.parse(endtime);
            var d = et.getTime() - st.getTime();

            if (d > 0)
            {
                res = d;
            }
        } else {
            res = realplaytime.toLong;
        }
        res
    }

   def GetCategoryList(category: String, taginfo: String, vname:String = null): ListBuffer[String] = {
	var categoryList = ListBuffer("NULL")
	var cats = category.split(Separators)
	var tags = taginfo.split(Separators)
	
	for(c <- cats) {
	   if (c != "NULL") {
		categoryList += c
	   }
	}

	for(t <- tags) {
           if (t != "NULL") {
                categoryList += t
           }
        } 

	if(vname != null) {
	   val loop = new Breaks
	   loop.breakable {
	      for(c <- TheatreCategoryList) {
	       	if (vname contains c) {
		   categoryList += c;
		   loop.break
		}
	      }
	   }
	}
        categoryList
   }

   def GetVectorDist(base64vec1: String, base64vec2: String):Double = {
	val vec1 = Base64.getDecoder().decode(base64vec1)
	val vec2 = Base64.getDecoder().decode(base64vec2)

	var a = 0
	var b = 0
	var c = 0

	for (i <- 0 to vec1.length-1) {
	   a =a + vec1(i)*vec2(i)	
	   b =b + vec1(i)*vec1(i)
	   c =c + vec2(i)*vec2(i)  
	}
	a * 1.0 / (math.sqrt(b*c))
   }
}
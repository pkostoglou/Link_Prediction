import org.apache.log4j.{Level, Logger}
import org.apache.spark
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.Vectors




object Link_Prediction_Part2 {


  def sameAuthor(auth1: String,auth2: String):Int={
    if(auth1==null || auth2 ==null)
      return 0
    val temp1=auth1.split(",")
    val temp2=auth2.split(",")
    var sameAuthors=0
    for( i <- 0 until temp1.length){
      for(j <-0 until temp2.length){
        if(temp1(i).compareTo(temp2(j))==0){
          sameAuthors=sameAuthors+1
        }
      }
    }
    return sameAuthors
  }

  def TFIDF(x: DataFrame,column: String,columnOut: String,numFeatures: Int):DataFrame = {
    val tokenizer = new Tokenizer().setInputCol(column).setOutputCol("words")
    val wordsData = tokenizer.transform(x.na.fill(Map(column -> "")))

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(numFeatures)

    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol(columnOut)
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    return rescaledData.drop("words").drop("rawFeatures").drop(column)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new spark.SparkConf().setAppName("Link Classification Application").setMaster("local[*]")
    val sc = new spark.SparkContext(conf)
    val sparkSession = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .enableHiveSupport()
      .getOrCreate()

    import sparkSession.implicits._
    import sparkSession.sql
    val dfHeaderless = sparkSession.read.format("csv").option("header", "false").load("Data-20181224/node_information.csv")
    val dfNodeInformation = dfHeaderless.toDF("Id", "Year", "Title", "Authors", "Journal", "Abstract")

    val dfNodeInfoProstemp=TFIDF(dfNodeInformation,"Title","fTitle",1000).drop("Title")
    //val dfNodeInfoProstemp=TFIDF(dfNodeInformation,"Abstract","fAbstract",1000).drop("Abstract")

    val Array(training, test) = dfNodeInfoProstemp.randomSplit(Array[Double](0.1, 0.9), 18)

    print(training.count())
    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      //.setInputCol("fAbstract")
      .setInputCol("fTitle")
      .setOutputCol("hashes")
    val model = brp.fit(training)

    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    val hashedDf=model.transform(training)
    val sAuthor= udf(sameAuthor _)
    val c=model.approxSimilarityJoin(hashedDf, hashedDf, 10, "EuclideanDistance")
      .select($"datasetA.Id".alias("idA"),
        $"datasetB.Id".alias("idB"),
        $"EuclideanDistance",
        $"datasetA.Authors".alias("AuthorsA"),
        $"datasetB.Authors".alias("AuthorsB")).filter($"idA"=!=$"idB")
      .withColumn("sameAuthors",sAuthor($"Authors1",$"Authors2")).show()
}





}

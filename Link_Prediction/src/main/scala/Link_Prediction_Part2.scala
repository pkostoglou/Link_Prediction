import org.apache.log4j.{Level, Logger}
import org.apache.spark
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{concat_ws, split, udf}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.storage.StorageLevel





object Link_Prediction_Part2 {

  def getSrcId(Year1: Int, Year2: Int,Id1:String,Id2:String):String={
    if(Year1<Year2){
      return Id2
    }else{
      return Id1
    }
  }

  def getDstId(Year1: Int, Year2: Int,Id1:String,Id2:String):String={
    if(Year1<Year2){
      return Id1
    }else{
      return Id2
    }
  }

  def YearDifference(Year1: Int, Year2: Int):Int={
    return math.abs(Year1 - Year2)
  }

  //cluster1==cluster2 && Year1-Year2<4
  def setlabel(cluster1: Int, cluster2: Int,Year1: Int, Year2: Int):Int={
    if(cluster1==cluster2){
      return 1
    }else{
      return 0
    }
  }

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
    val dfNodeInformation = dfHeaderless.toDF("Id", "Year1", "Title", "Authors", "Journal", "Abstract").withColumn("Year",$"Year1".cast(IntegerType))

    val dfNodeInfoProstemp=TFIDF(dfNodeInformation,"Title","fTitle",1000).drop("Title")
    val dfNodeInfoPros=TFIDF(dfNodeInfoProstemp,"Abstract","fAbstract",10).drop("Abstract")

    //val Array(training, test) = dfNodeInfoPros.randomSplit(Array[Double](0.1, 0.9), 18)

    val assemblerHash = new VectorAssembler().
      setInputCols(Array("fTitle"))
      .setOutputCol("hashVector")
    val hashDf=assemblerHash.transform(dfNodeInfoPros)

    val assembler1 = new VectorAssembler().
      setInputCols(Array("Year")).
      setOutputCol("featuresTemp")

    val assembledDf = assembler1.transform(hashDf)

    //var i=2
    //while(i<10) {
      val kmeans = new KMeans().setK(2).setSeed(1L).setFeaturesCol("featuresTemp")
      val cluster = kmeans.fit(assembledDf)

      // Make predictions
      val predictions = cluster.transform(assembledDf)

      //predictions.show()

      // Evaluate clustering by computing Silhouette score
      val evaluator = new ClusteringEvaluator().setFeaturesCol("featuresTemp")

      val silhouette = evaluator.evaluate(predictions)
      println(s"Silhouette with squared euclidean distance = $silhouette with i = " )
      //i=i+1
    //}
//val predictions=assembledDf
    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(1.0)
      .setNumHashTables(1)
      //.setInputCol("fAbstract")
      //.setInputCol("fTitle")
      .setInputCol("hashVector")
      .setOutputCol("hashes")
    val model = brp.fit(predictions)

    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    val hashedDf=model.transform(predictions)
    val sAuthor= udf(sameAuthor _)
    val difYear = udf(YearDifference _ )
    val setLabel = udf(setlabel _)
    val getDst = udf(getDstId _)
    val getSrc = udf(getSrcId _)
    val c=model.approxSimilarityJoin(hashedDf, hashedDf, 9.0, "EuclideanDistance")
      .select($"datasetA.Id".alias("idA"),
        $"datasetB.Id".alias("idB"),
        $"datasetA.prediction".alias("clusterA"),
        $"datasetB.prediction".alias("clusterB"),
        $"EuclideanDistance",
        $"datasetA.Year".alias("YearA"),
        $"datasetB.Year".alias("YearB"))
        //$"datasetA.Authors".alias("AuthorsA"),
        //$"datasetB.Authors".alias("AuthorsB"))
      .filter($"idA"=!=$"idB")
      .withColumn("label",setLabel($"clusterA",$"clusterB",$"YearA",$"YearB"))
      .withColumn("IdSrc",getSrc($"YearA",$"YearB",$"IdA",$"IdB"))
      .withColumn("IdDst",getDst($"YearA",$"YearB",$"IdA",$"IdB"))
      //.withColumn("IdLabel",concat_ws("\t",$"IdA",$"IdB"))
    c.persist(StorageLevel.MEMORY_AND_DISK)
    //c.show()


    val myRight=c.filter($"label"===1).count()
    println("Evaluation begins ")
    val groundTruth = sparkSession.read.format("csv").option("header", "false").load("Data-20181224/Cit-HepTh.txt").toDF("Id").withColumn("_tmp", split($"Id", "\t")).select(
      $"_tmp".getItem(0).as("Id1"),
      $"_tmp".getItem(1).as("Id2")).drop("_tmp")
    groundTruth.show()
    val allEdges=groundTruth.count()
    val evaluation=groundTruth.join(c,($"IdSrc"===$"Id1"&&$"IdDst"===$"Id2"))
    val setRight=evaluation.filter($"label"===1).count()
    println(c.count(),evaluation.count)
    println(setRight,allEdges)
    val recall = (setRight).toDouble / allEdges
      println(recall)
    val precision = (setRight).toDouble/myRight
    println(precision)
    println(2*precision*recall/(precision+recall))



  }





}

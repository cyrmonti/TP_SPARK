package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover, StringIndexer
  ,OneHotEncoder,VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")
    val path = "/Users/cyril/Desktop/MS BGD/P1/Spark/TP/TP3/prepared_trainingset/"

    val df : DataFrame = spark
      .read
      .option("header", "true")
      .parquet(path)

    //df.show(5)

    /*Tokenizer */
    /*
    Ici l'objectif est de segmenter la colonne text en une liste
    de mots.
    */
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    /*
    On supprime les stopwords car ils n'ont pas d'impact sur le score.
    */
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokens_SWR")


    /* CountVectorizer */
    /*
    Le CountVectorizer va nous permettre de transformer nos données texte
    en données numériques. Plus facile pour l'apprentissage.
    */
    val cvModel= new CountVectorizer()
      .setInputCol("tokens_SWR")
      .setOutputCol("test")


    /** TFIDF */
    /*
    La partie IDF nous permet de mettre des poids sur l'occurence des mots,
    justement pour négliger ceux qui appraissent trop souvent par exemple.
    */
    val idf = new IDF()
      .setInputCol("test")
      .setOutputCol("tfidf")



    /** StringIndexer **/
    /*
    Pour les monnaies ou les pays, le string indexer nous permet de
    catégoriser des données textuelles en données numériques.
    */
    val countryStringIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")
    //.fit(rescaledData)

    val currencyStringIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")
    //.fit(rescaledData)


    // Convertir les variables catégorielles indexées numériquement en flag (variable currency et country)
    val encoder1 = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("countryVec")


    val encoder2 = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currencyVec")


    /* Vector Assembler */
    /*
    Pour les modèles d'apprentissage, Spark préfère travailler avec un vecteur
    plutôt qu'avec plusieurs colonnes.
    */
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa","goal","countryVec","currencyVec"))
      .setOutputCol("features")


    // Modele de classification
    val lr = new LogisticRegression()
      .setElasticNetParam( 0.0 )
      .setFitIntercept( true )
      .setFeaturesCol( "features" )
      .setLabelCol( "final_status" )
      .setStandardization( true )
      .setPredictionCol( "predictions" )
      .setRawPredictionCol( "raw_predictions" )
      .setThresholds( Array ( 0.7 , 0.3 ))
      .setTol( 1.0e-6 )
      .setMaxIter( 300 )


    /** Pipeline **/
    /*
    Notre pipeline va permettre d'automatiser l'ensemble des tâches créées auparavant.
    */
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, countryStringIndexer, currencyStringIndexer,
        encoder1, encoder2, assembler,lr))


    // Train test Split
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)


    /* ParamGrid */
    /*
    Idéal pour tester plusieurs paramètres à notre modèle et trouver
    les paramètres optimaux
    */
    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, Array(55.0,75.0,95.0))
      .addGrid(lr.regParam, Array(10e-2, 10e-4,10e-6,10e-8))
      .build()


    /** F1 Score **/
    /*
    Pour de la catégorisation, la métrique F1 Score est très appropiée
    car elle prend en compte la precision et le recall.
    */
    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")


    /** TrainValidationSplit **/
    /*
    On suit les instruction du TP qui nous demande d'utiliser en chaque point
    de la grille, 70% des données pour l'entrainement et 30% pour la validation.
    */
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(f1Evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val model = trainValidationSplit.fit(training)


    // Evalute best model
    val dfWithPredictions = model.transform(test)
    val bestF1Score = f1Evaluator.evaluate(dfWithPredictions)
    println(s"Best F1 score = $bestF1Score")

    dfWithPredictions.groupBy("final_status", "predictions").count.show

    // Save Model
    //model.write.overwrite().save("myModel")
    print("Saved !")
  }
}

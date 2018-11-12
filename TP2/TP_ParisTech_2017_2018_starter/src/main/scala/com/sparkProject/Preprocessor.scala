package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP
    // on vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation de la SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc et donc aux mécanismes de distribution des calculs.)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Preprocessor")
    // a) charger les données
    var path = "/Users/cyril/Desktop/MS BGD/Spark/TP/TP2/train.csv"
    val df = spark.read.option("header", "true").csv(path)  //reading the header

    // b) nombre de lignes et colonnes
    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    // c) afficher un extrait du dataframe sous forme de tableau
    df.show(5)

    // d) afficher le schema du dataframe (nm des colonnes et le type des données contenues dans chacune d'elles)
    df.printSchema()

    // e) assigner le type Int aux colonnes qui vous semblent contenir des entiers
    /*
    val dfCasted = df
      .withColumn("goal", df$goal.cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    // On vérifie que les modifications sont bien prises en compte
    dfCasted.printSchema()

    // Partie 2 : CLEANING

    // a) afficher une description statistique des colonnes de type Int
    dfCasted.select("goal" ,"backers_count", "final_status").describe().show
  */


  }

}

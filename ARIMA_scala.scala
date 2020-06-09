// Databricks notebook source
//import libraries
import com.cloudera.sparkts.models.ARIMA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import scala.math._

// COMMAND ----------

//create dataframe from csv
val df_raw = spark.sql("select * from aapl_final_csv")
val train_data = df_raw.filter(df_raw("Date") < "2017-03-01T00:00:00.000+0000").filter(df_raw("Date") >= "2013-06-01T00:00:00.000+0000")
val test_data = df_raw.filter(df_raw("Date") < "2018-12-31T00:00:00.000+0000").filter(df_raw("Date") >= "2017-03-01T00:00:00.000+0000")
display(test_data.select("*"))

// COMMAND ----------

// convert dataframe column to dense vector, which is required by ARIMA algorithm
val vtrain = Vectors.dense(train_data.select("Close").rdd.map(r => r(0).asInstanceOf[Float]).map(x=>x.toDouble).collect())
val vtest = Vectors.dense(test_data.select("Close").rdd.map(r => r(0).asInstanceOf[Float]).map(x=>x.toDouble).collect())

// COMMAND ----------

// fit ARIMA model
val p = 3
val d = 1
val q = 2
val arimaModel = ARIMA.fitModel(p, d, q, vtrain)
println("coefficients: " + arimaModel.coefficients.mkString(","))

// COMMAND ----------

//----------------------- add true lead value column to dataframe----------------------
val lead_value = 1

val ref_test = test_data.select("Date", "Close")

// create lead column
val w = Window.orderBy("Date")
val lead_column = lead("Close", lead_value).over(w)
val test_lead = ref_test.withColumn("true_lead", lead_column).na.drop()
display(test_lead)

// COMMAND ----------

//create prediction results in vector format
val pred = Array.ofDim[Float](test_lead.count.toInt)
val a = Array.ofDim[Float](p)

for(i <- 0 to test_lead.count.toInt - 1) {
  if(i < p){
    pred(i) = test_lead.select("true_lead").collect()(i).getFloat(0)
  }
  else{
    for(count <- 0 to p-1){
      a(count) = test_lead.select("Close").collect()(i-p+count).getFloat(0)
    }
    val va = Vectors.dense(a.map(_.toDouble))
    val forecast = arimaModel.forecast(va, lead_value)
    pred(i) = forecast(forecast.size - 1).toFloat
  }
}

// COMMAND ----------

// convert prediction vector to column and add prediction column to dataframe
val Predictions = pred.toSeq.toDF("Prediction_lead")
val test_lead1 = test_lead.withColumn("row_index", monotonically_increasing_id())
val Predictions1 = Predictions.withColumn("row_index1", monotonically_increasing_id())
val pred_summary = test_lead1.join(Predictions1, $"row_index" === $"row_index1", "left_outer").drop("row_index").drop("row_index1")

// COMMAND ----------

display(pred_summary)

// COMMAND ----------

//calculate RSME
val true_lead_RSME = pred_summary.select("true_lead").rdd.map(r => r(0).asInstanceOf[Float]).collect()   //convert vector to array
var RSME = 0.0
for(i <- 0 to true_lead_RSME.size-1){
  RSME = RSME + scala.math.pow(true_lead_RSME(i) - pred(i),2)
}
RSME = scala.math.sqrt(RSME/true_lead_RSME.size.toDouble)

// COMMAND ----------

//Calculate sMAPE
val true_lead_sMAPE = pred_summary.select("true_lead").rdd.map(r => r(0).asInstanceOf[Float]).collect()
var sMAPE = 0.0
for(i <- 0 to true_lead_sMAPE.size-1){
  sMAPE = sMAPE + scala.math.abs(pred(i) - true_lead_sMAPE(i)) / ((scala.math.abs(pred(i)) + scala.math.abs(true_lead_sMAPE(i)))/2)
}
sMAPE = sMAPE / true_lead_sMAPE.size.toDouble

// COMMAND ----------

//junk

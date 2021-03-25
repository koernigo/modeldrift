# Databricks notebook source
def score_quality(df, model_runid):
  stage = "production"
  model_production_uri = f"models:/{model_name}/{stage}"
  print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
  prod_model = mlflow.spark.load_model(model_production_uri)
  #run_details = get_run_details(model_runid)
  #print('Using model version:'+ run_details['runid'])
  predicted_quality = prod_model.transform(df)
  predicted_quality = predicted_quality.select('pid', 'process_time', 'predicted_quality')
  #predicted_quality = predicted_quality.withColumn('model_version', F.lit(run_details['runid']))
  display(predicted_quality)
  return predicted_quality

# COMMAND ----------

def stream_score_quality(df):
  
  #prod_run_details = get_model_production(mlflow_exp_id)
  #predicted_quality = score_quality(df, prod_run_details['runid'])
  predicted_quality = score_quality(df,0)
  
  predict_stream = (predicted_quality.writeStream
  .format("delta")
  .outputMode("append")
  .option("checkpointLocation", predicted_quality_cp_blob)
  .start(predicted_quality_blob))
  
  return predict_stream

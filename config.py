# Databricks notebook source
import json
# Noise for data generator
dg_noise = {"temp_noise": 0.2, "pressure_noise": 0.2, "duration_noise": 0.2}

userid = 'oliver.koernig'

# Data paths (replace with actual locations. Could be directly to S3, Azure blob/ADLS, or these locations mounted locally)
sensor_reading_blob = "/tmp/sensor_reading"
product_quality_blob = "/tmp/product_quality"

predicted_quality_blob = "/tmp/predicted_quality"
predicted_quality_cp_blob = "/tmp/predicted_quality_checkpoint"

# Modeling & MLflow settings
databricks_host=json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())

mlflow_exp_name = "Glassware Quality Predictor"
mlflow_exp_id = databricks_host["tags"]["notebookId"] # Experiment ID equals Notebook ID

model_compare_metric = 'accuracy'

# COMMAND ----------

# MAGIC %run ./utils/viz_utils

# COMMAND ----------

# MAGIC %run ./utils/mlflow_utils

# COMMAND ----------

from pyspark.sql import Window
import pyspark.sql.functions as F

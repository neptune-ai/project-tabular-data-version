import pickle

import neptune.new as neptune
import numpy as np
import pandas as pd
import xgboost as xgb
from neptune.new.integrations.xgboost import NeptuneCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_s3_path = "s3://kamil-projects/tabular-data/dataset-2/train/"
valid_s3_path = "s3://kamil-projects/tabular-data/dataset-2/valid/"
test_s3_path = "s3://kamil-projects/tabular-data/dataset-2/test/"

# (neptune) create run
run = neptune.init(
    project="common/project-tabular-data-version",
    tags=["training", "S3"],
    source_files=["train.py", "environment.yml"],
)

##########################
# S3: track data version #
##########################

# (neptune) track data version
run["data/train"].track_files(train_s3_path)
run["data/valid"].track_files(valid_s3_path)
run["data/test"].track_files(test_s3_path)
run.wait()

# (neptune) prepare data
run["data/train"].download(destination="train")
run["data/valid"].download(destination="valid")
run["data/test"].download(destination="test")
run.wait()

X_train = pd.read_csv("train/X.csv", index_col="Id")
y_train = pd.read_csv("train/y.csv")
X_valid = pd.read_csv("valid/X.csv", index_col="Id")
y_valid = pd.read_csv("valid/y.csv")
X_test = pd.read_csv("test/X.csv", index_col="Id")
y_test = pd.read_csv("test/y.csv")

# (neptune) log train sample
run["data/train_sample"].upload(neptune.types.File.as_html(X_train.head(20)))

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test, label=y_test)

###########################
# XGBoost: model training #
###########################

# define parameters
model_params = {
    "eta": 0.3,
    "gamma": 0.0001,
    "max_depth": 2,
    "colsample_bytree": 0.85,
    "subsample": 0.9,
    "objective": "reg:squarederror",
    "eval_metric": ["mae", "rmse"],
}
evals = [(dtrain, "train"), (dval, "valid")]
num_round = 100

# (neptune-xgboost integration) create neptune_callback to track XGBoost training
base_namespace = "model_training"

neptune_callback = NeptuneCallback(
    run=run,
    base_namespace=base_namespace,
    log_tree=[0, 1, 2],
)

# (neptune) pass neptune_callback to the train function and run training
xgb.train(
    params=model_params,
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=evals,
    callbacks=[neptune_callback],
)

run.sync(wait=True)

# (neptune) download model from the run to make predictions on test data
run[f"{base_namespace}/pickled_model"].download("xgb.model")
with open("xgb.model", "rb") as file:
    bst = pickle.load(file)

test_preds = bst.predict(dtest)

# (neptune) log test scores
run[f"{base_namespace}/test_score/rmse"] = np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_preds))
run[f"{base_namespace}/test_score/mae"] = mean_absolute_error(y_true=y_test, y_pred=test_preds)
run.sync(wait=True)

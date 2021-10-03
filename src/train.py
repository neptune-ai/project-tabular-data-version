import pickle

import neptune.new as neptune
import numpy as np
import xgboost as xgb
from neptune.new.integrations.xgboost import NeptuneCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_namespace = "model_training"

##########################
# part 1: model training #
##########################

# (neptune) create run
run = neptune.init(
    project="common/project-tabular-data-version",
    tags=["training"],
)

# (neptune-xgboost integration) create neptune_callback to track XGBoost training
neptune_callback = NeptuneCallback(
    run=run,
    base_namespace=base_namespace,
    log_tree=[0, 1, 2],
)

# prepare data
X_train = ...
y_train = ...
X_valid = ...
y_valid = ...
X_test = ...
y_test = ...

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test, label=y_test)

# (neptune) track files
run["data/train"].track_files("s3://kamil-projects/tabular-data/dataset-1/train/")
run["data/valid"].track_files("s3://kamil-projects/tabular-data/dataset-1/valid/")
run["data/test"].track_files("s3://kamil-projects/tabular-data/dataset-1/test/")

# (neptune) log train sample
run["data/train_sample"].upload(neptune.types.File.as_html(X_train.head(20)))

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

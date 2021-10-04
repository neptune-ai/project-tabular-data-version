import pickle

import neptune.new as neptune
import numpy as np
import pandas as pd
import xgboost as xgb
from neptune.new.integrations.xgboost import NeptuneCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error

# (neptune) create run that will store re-running metadata
run = neptune.init(
    project="common/project-tabular-data-version",
    tags=["training", "S3", "from-reference"],
    source_files=["train.py", "re-training.py", "../environment.yml"],
)

##########################################
# Fetch data info from the reference run #
##########################################

# (neptune) fetch project
project = neptune.get_project(name="common/project-tabular-data-version")

# (neptune) find reference run
reference_run_df = project.fetch_runs_table(tag="reference").to_pandas()
reference_run_id = reference_run_df["sys/id"].values[0]

# (neptune) resume reference run in the read-only mode
reference_run = neptune.init(
    project="common/project-tabular-data-version",
    run=reference_run_id,
    mode="read-only",
)

# (neptune) download data logged to the reference run
reference_run["data/train"].download(destination="train")
reference_run["data/valid"].download(destination="valid")
reference_run["data/test"].download(destination="test")
reference_run.wait()

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

#######################################
# Assign the same data version to run #
#######################################

run["data/train"].assign(reference_run["data/train"].fetch())
run["data/valid"].assign(reference_run["data/valid"].fetch())
run["data/test"].assign(reference_run["data/test"].fetch())

#######################################
# Fetch params from the reference run #
#######################################

# Fetch the runs parameters
reference_run_params = \
    reference_run["model_training/booster_config/learner/gradient_booster/updater/grow_colmaker/train_param"].fetch()
reference_run.wait()

reference_run_params["objective"] = reference_run["model_training/booster_config/learner/objective/name"].fetch()
reference_run.wait()

reference_run_params["eval_metric"] = ["mae", "rmse"]

# (neptune) close reference run
reference_run.stop()

###########################
# XGBoost: model training #
###########################

evals = [(dtrain, "train"), (dval, "valid")]
num_round = 100

# (neptune-xgboost integration) create neptune_callback to track XGBoost training
neptune_callback = NeptuneCallback(
    run=run,
    base_namespace=base_namespace,
    log_tree=[0, 1, 2],
)

# (neptune) pass neptune_callback to the train function and run training
xgb.train(
    params=reference_run_params,
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

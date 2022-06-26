import argparse
import logging
import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import category_encoders as ce
import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils import resample, shuffle
from xgboost import XGBClassifier

from helpers import read_from_dir, save_to_dir

logger = logging.getLogger("data_prep")
coloredlogs.install(
    fmt="%(asctime)s - %(name)s - %(process)s - %(levelname)s - %(message)s",
    level="INFO",
)

get_col_index = lambda df, cols: [df.columns.get_loc(c) for c in cols if c in df]


def eval_fn(y_test: np.ndarray, y_hat: np.ndarray, model_name: str):
    labels = ["Customer", "Subscriber"]
    cm = confusion_matrix(
        y_true=np.array(["Customer", "Subscriber"])[y_test.astype(int)],
        y_pred=np.array(["Customer", "Subscriber"])[y_hat.astype(int)],
        labels=labels,
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    # plt.show()
    plt.savefig(f"images/{model_name}.png")
    # with open(f"images/{model_name}.pickle", "wb") as f:
    #     pickle.dump(cm, f)

    print("=================================================================")
    print(f"{model_name}")
    print("=================================================================")
    print(f"Confusion matrix:\n {cm}")
    print(f"Accuracy: {accuracy_score(y_test, y_hat)}")
    print(f"Balanced-Accuracy-Score: {balanced_accuracy_score(y_test, y_hat)}")
    print(f"F1-Score: {f1_score(y_test, y_hat)}")
    print(f"Precsion-Score: {precision_score(y_test, y_hat)}")
    print(f"ROC-AUC-Score: {roc_auc_score(y_test, y_hat)}")
    print("=================================================================")
    print("\n")


def main(
    sample_size: float = 1e6,
    fit_xgb: bool = False,
    fit_lr: bool = False,
    upsample: bool = False,
):

    logger.info("Starting main")
    df = read_from_dir("data/output")

    if sample_size > 0:
        logger.info(f"Using sample of size {sample_size}.")
        df = df.sample(int(sample_size), random_state=152)
    else:
        logger.info("Using the complete dataset.")

    df["target"] = (df.usertype == "Subscriber") * 1
    df["weekend"] = df["weekday"] >= 6
    df["birthyear_x_gender"] = (df["birth_year"] == 1969) * (df["gender"] == 0)
    df["age"] = 2018 - df["birth_year"]
    df["age_sq"] = df["age"] ** 2
    df["temp_max_sq"] = df["temp"] ** 2
    df["hour_sq"] = df["hour"] ** 2
    df["month_sq"] = df["month"] ** 2
    df["trip_duration_min_sq"] = df["trip_duration_min"] ** 2
    df["trip_duration_min_sq"] = df["trip_duration_min"] ** 2
    df["trip_suburb"] = df["start_suburb"] + df["end_suburb"]
    df["trip_station"] = df["start_station_id"].astype(str) + df[
        "end_station_id"
    ].astype(str)

    onehot_cols = ["gender", "weather_main"]
    onehot_time_cols = ["month", "hour"]
    target_cols = ["trip_suburb", "trip_station"]
    pass_cols = [
        "trip_duration_min",
        "trip_duration_min_sq",
        "roundtrip",
        "distance",
        "birthyear_x_gender",
        "age",
        "age_sq",
        "hour_sq",
        "month_sq",
        "weekend",
        "temp",
        "temp_max",
        "temp_max_sq",
        "holiday",
        "workingday",
    ]
    drop_cols = list(
        set(df.columns.tolist())
        - set(["target"] + onehot_cols + target_cols + pass_cols)
    )

    logger.info(
        f"Using features: {pass_cols+target_cols+onehot_cols+onehot_time_cols} "
    )
    logger.info(f"Features not used: {drop_cols} ")

    onehot_time_cols_idx = get_col_index(df, onehot_time_cols)
    onehot_cols_idx = get_col_index(df, onehot_cols)
    target_cols_idx = get_col_index(df, target_cols)
    pass_cols_idx = get_col_index(df, pass_cols)
    drop_cols_idx = get_col_index(df, drop_cols)

    X = df.values
    y = df.loc[:, "target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, stratify=y
    )

    if upsample:
        X0 = X_train[y_train == 0, :]
        X1 = X_train[y_train == 1, :]
        n_samples = np.sum(y_train)
        X0_up = resample(X0, random_state=42, n_samples=n_samples, replace=True)
        X_train = np.concatenate([X0_up, X1])
        y_train = np.concatenate([np.zeros(len(X0_up)), np.ones(len(X1))])
        X_train, y_train = shuffle(X_train, y_train, random_state=1242)

    ct = ColumnTransformer(
        transformers=[
            ("onehot_time", OneHotEncoder(drop="first"), onehot_time_cols_idx),
            ("onehot", OneHotEncoder(drop="first"), onehot_cols_idx),
            ("target", ce.MEstimateEncoder(), target_cols_idx),
            # ("pass_cols", "passthrough", pass_cols_idx),
            ("pass_cols", RobustScaler(), pass_cols_idx),
            ("drop_cols", "drop", drop_cols_idx),
        ]
    )

    # Logistic Regression
    if fit_lr:
        logger.info("Fitting Logistic Regression.")
        pipeline = Pipeline(
            steps=[
                ("ct", ct),
                ("lr", LogisticRegression(max_iter=1000, solver="liblinear")),
            ]
        )
        param_grid = {
            # "ct__target__m":[1,2],
            "lr__C": [10**i for i in range(-6, 0)],
            "lr__penalty": ["l2"],
            # "lr__l1_ratio": [0.0, 0.5,1.0],
        }
        clf = GridSearchCV(pipeline, param_grid=param_grid, cv=3, n_jobs=8)
        clf.fit(X_train, y_train)
        logger.info("Fitted Logistic Regression.")
        df_best = pd.DataFrame(clf.cv_results_).sort_values(
            "mean_test_score", ascending=False
        )
        df_best = df_best.head().loc[
            :, ["mean_test_score", "param_lr__C", "param_lr__penalty"]
        ]
        logger.info(f"Best Params:\n {df_best}")
        y_hat = clf.predict(X_test)
        eval_fn(y_test, y_hat, "LogisticRegression")

    # XGBoost
    if fit_xgb:
        logger.info("Fitting XGBoost.")
        pipeline = Pipeline(
            steps=[
                ("ct", ct),
                (
                    "xgb",
                    XGBClassifier(
                        # n_jobs=8,
                        use_label_encoder=False,
                        objective="binary:logistic",
                        eval_metric="error",
                    ),
                ),
            ]
        )
        param_grid = {
            "xgb__n_estimators": [200],
            "xgb__max_depth": [8, 16],
            "xgb__learning_rate": [0.2, 0.4],
            # "xgb__eval_metric": ["logloss"],
            # "xgb__max_leaves": [2,  4,8,16],
            # "xgb__amma": [0.1, 0.2, 0.3,],
            # "xgb__olsample_bytree": [ 0.4, 0.5, ],
            # "xgb__in_child_weight": [1, 2, 3],
            # "xgb__ubsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
        clf = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=8)
        clf.fit(X_train, y_train)
        logger.info("Fitted XGBoost.")
        df_best = pd.DataFrame(clf.cv_results_).sort_values(
            "mean_test_score", ascending=False
        )
        df_best = df_best.head().loc[
            :,
            [
                "mean_test_score",
                "param_xgb__n_estimators",
                "param_xgb__max_depth",
                "param_xgb__learning_rate",
            ],
        ]
        logger.info(f"Best Params: {df_best}")

        y_hat = clf.predict(X_test)
        eval_fn(y_test, y_hat, "XGBoost")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb", action="store_true", help="Run XGBoost")
    parser.add_argument("--lr", action="store_true", help="Run Logistic Regression")
    parser.add_argument("--upsample", action="store_true", help="Balance")
    parser.add_argument(
        "--sample-size",
        default=100000,
        type=int,
        help="Sample size for training",
    )

    args = parser.parse_args()
    main(
        fit_xgb=args.xgb,
        fit_lr=args.lr,
        sample_size=args.sample_size,
        upsample=args.upsample,
    )

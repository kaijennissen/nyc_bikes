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
    confusion_matrix,
    f1_score,
    precision_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils import resample, shuffle
from xgboost import XGBClassifier

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
    print(f"F1-Score: {f1_score(y_test, y_hat)}")
    print(f"Precsion-Score: {precision_score(y_test, y_hat)}")
    print("=================================================================")
    print("\n")


def main(
    sample_size: float = 1e6,
    fit_xgb: bool = False,
    fit_lr: bool = False,
    upsample: bool = False,
):

    logger.info("Starting main")
    if sample_size <= 0:
        logger.info("Using the complete dataset.")
        df = pd.read_parquet("data/processed/df_nyc.parquet")
    else:
        logger.info(f"Using sample of size {sample_size}.")
        df = pd.read_parquet("data/processed/df_nyc.parquet").sample(
            int(sample_size), random_state=152
        )
    df_weather = pd.read_parquet("data/external/weather_history_2018.parquet")
    df_geo = pd.read_parquet("data/processed/df_station.parquet")

    df["target"] = (df.usertype == "Subscriber") * 1
    df["weekend"] = df["weekday"] >= 6

    onehot_cols = ["gender", "weather_main"]
    onehot_time_cols = ["weekday", "month", "hour"]
    target_cols = ["start_suburb", "end_suburb"]
    pass_cols = [
        "trip_duration_min",
        "roundtrip",
        "distance",
        "birth_year",
        "weekend",
        "temp",
        "temp_max",
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
            ("target", ce.TargetEncoder(), target_cols_idx),
            ("pass_cols", "passthrough", pass_cols_idx),
            ("drop_cols", "drop", drop_cols_idx),
        ]
    )

    # Logistic Regression
    if fit_lr:
        logger.info("Fitting Logistic Regression.")
        pipeline = Pipeline(
            steps=[
                ("ct", ct),
                # ("scaler", RobustScaler()),
                ("lr", LogisticRegression(max_iter=100)),
            ]
        )
        param_grid = {
            "lr__C": [0.1, 1, 10, 100],
            "lr__penalty": ["l2", "l1"],
        }
        clf = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=8)
        clf.fit(X_train, y_train)
        logger.info("Fitted Logistic Regression.")
        df_best = pd.DataFrame(clf.cv_results_).sort_values(
            "mean_test_score", ascending=False
        )
        df_best = df_best.head().loc[
            :, ["mean_test_score", "param_lr__C", "param_lr__penalty"]
        ]
        logger.info(f"Best Params: {df_best}")
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
                        n_jobs=8,
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
        clf = GridSearchCV(pipeline, param_grid, cv=5)
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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train(df: pd.DataFrame) -> LogisticRegression:
    y = df["y"]

    X = df[["var_1", "var_2", "var_3"]]

    logreg = LogisticRegression(penalty=None)

    logreg.fit(X, y)

    return logreg

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

Metrics = namedtuple("Metrics", ["auroc", "nll"])


def evaluate(df: pd.DataFrame, model: LogisticRegression):
    y = df["y"]

    X = df[["var_1", "var_2", "var_3"]]

    p = model.predict_proba(X)[:, 1]

    auroc = roc_auc_score(y, p)
    nll = log_loss(y, p)

    return Metrics(auroc=auroc, nll=nll)


if __name__ == "__main__":
    np.random.seed(9562)
    n = 100_000
    df = pd.DataFrame(
        {
            "var_1": np.random.randn(n),
            "var_2": np.random.randn(n),
            "var_3": np.random.randn(n),
        }
    )

    def logistic(x: float):
        return np.exp(x) / (1 + np.exp(x))

    df["y"] = 0.5 + 0.5 * df["var_1"] - 1.2 * df["var_2"] + 0.2 * df["var_3"]
    df["y"] = df["y"].apply(logistic)
    df["y"] = np.random.binomial(1, df["y"])

    logreg = train(df)
    print(logreg.coef_)

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


def train(df: pd.DataFrame) -> LogisticRegression:
    y = df["y"]

    X = df[["var_1", "var_2", "var_3"]]

    logreg = LogisticRegression(penalty=None)

    logreg.fit(X, y)

    return logreg


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

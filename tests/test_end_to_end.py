import numpy as np
import pandas as pd

from ml_pipelines.eval.eval import evaluate
from ml_pipelines.train.train import train


def make_mock_data(n: int, seed: int):
    np.random.seed(seed)
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

    return df


if __name__ == "__main__":
    train_df = make_mock_data(8_000, seed=9562)
    eval_df = make_mock_data(2_000, seed=4352)

    logreg = train(train_df)
    print(logreg.coef_)

    eval_metrics = evaluate(eval_df, logreg)
    print(eval_metrics)

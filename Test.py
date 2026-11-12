import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    import numpy as np

    # Parameters for the "insane" dataframe
    n_rows = 10_000
    n_numeric = 30
    n_categorical = 20
    n_datetime = 10
    n_text = 20

    # Numeric columns
    numeric_data = {
        f"num_{i}": np.random.randn(n_rows) * np.random.randint(1, 100)
        for i in range(n_numeric)
    }

    # Categorical columns with random categories
    categorical_data = {
        f"cat_{i}": np.random.choice([f"group_{j}" for j in range(1, 11)], size=n_rows)
        for i in range(n_categorical)
    }

    # Datetime columns spanning a range of years
    base_date = np.datetime64("2000-01-01")
    datetime_data = {
        f"date_{i}": base_date + np.random.randint(0, 8000, size=n_rows).astype("timedelta64[D]")
        for i in range(n_datetime)
    }

    # Text columns with random strings
    def random_string(length):
        letters = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=length)
        return "".join(letters)

    text_data = {
        f"text_{i}": [random_string(np.random.randint(5, 15)) for _ in range(n_rows)]
        for i in range(n_text)
    }

    # Combine all dictionaries into one DataFrame
    df_insane = pd.DataFrame({**numeric_data, **categorical_data, **datetime_data, **text_data})

    # Introduce some NaN values randomly
    for col in df_insane.columns:
        mask = np.random.rand(n_rows) < 0.01  # ~1% NaNs per column
        df_insane.loc[mask, col] = np.nan

    df_insane
    return


if __name__ == "__main__":
    app.run()

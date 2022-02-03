import numpy as np
import pandas as pd
from itertools import cycle


def caternary(x, a=1):
    return a * np.cosh(x / a)


def caternary_approx(x, a=1):
    return a + x**2 / (2 * a)


def main():
    import plotly.express as px

    x = np.linspace(-2, 2)
    a = [1, 1.4, 1.8]

    df = pd.DataFrame()
    cols = [f"${func}_{{{ai:0.1f}}}$" for func in ["f", "g"] for ai in a]
    for col, ai in zip(cols, cycle(a)):
        df2 = pd.DataFrame({"x": x})
        df2["y"] = caternary(x, ai) if "f" in col else caternary_approx(x, ai)
        df2["line"] = col
        df2["a"] = ai
        df2["form"] = "Exact" if "f" in col else "Approx"
        df = pd.concat([df, df2])

    fig = px.line(
        df,
        x="x",
        y="y",
        line_group="line",
        color="a",
        line_dash="form",
        title="Caternary Demo",
    )
    fig.update_yaxes(range=[0.5, 4])
    fig.show()


if __name__ == "__main__":
    main()

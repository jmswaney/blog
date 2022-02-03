# Steps

1. Create `peotry` conda env with latest version of Python

    ```bash
    conda create -y -n poetry python=3
    conda activate poetry
    python --version # prionts "Python 3.10.0"
    ```

2. Install poetry

    ```bash
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    ```

    Aside: add autocompletions for easier use in your terminal
    [here](https://python-poetry.org/docs/#enable-tab-completion-for-bash-fish-or-zsh)

    ```bash
    # Oh-My-Zsh
    mkdir $ZSH_CUSTOM/plugins/poetry
    poetry completions zsh > $ZSH_CUSTOM/plugins/poetry/_poetry
    ```

    Then add `poetry` to the list of plugins in your `~/.zshrc` file.

    After restarting your shell, running `poetry --version` should print your
    version of poetry to the console. You should also see tab completion listing out
    possible poetry commands if you enabled autocompletion correctly.

    Note that this "global" installation of poetry is similar to `npm` or `yarn` if
    you are familiar to package managers for web development.

3. Create your first poetry project

    ```bash
    poetry new math-demo
    cd math-demo
    ```

    This will build out the basic structure of a python package in poetry's standard
    format in the `math-demo` folder. It should have a subfolder also named
    `math-demo`, which similar to other python packages, is where our source code
    should go. There is also a folder for unit tests and a curious new
    `pyproject.toml` file at the top-level.

    Explain TOML, compare it to how npm / yarn use JSON as a declarative way of
    building out packages in the Javascript ecosystem. It's analogous, but now in Python!

    Dependencies go under `[tool.poetry.dependencies]` in the pyproject.toml file.
    There's even separate places for package dependencies and development-only
    dependencies.

    You can see the dummy tests pass by running `poetry run pytest`

4. Add your dependencies

    Let's add plotly and numpy and do some math!

    ```bash
    poetry add plotly numpy pandas
    ```

    If you are using VSCode on macOS, you can go into your settings and add
    `Library/Caches/pypoetry/virtualenvs` to your `Python: Venv Folders`. This will
    allow us to select the correct Python interpreter and, importantly, let VSCode
    find the packages we have just installed.

    ```bash
    poetry add -D pylint
    ```

5. Setup the tests

    Let's make a module that calculates a caternary curve and a parabolic
    approximation to it. First we can write a test knowing that the curve should
    be symmetric about the y-axis.

    To do this, start by making a new python module called `hyperbolic.py`
    inside `math_demo` next to `__init__.py`.

    Let's write the function we want to test and start with something that will
    definitely not be symmetric so our tests will fail:

    ```python
    # in hyperbolic.py
    def caternary(x):
        return x
    ```

    Great, now let's change out test suite to use this:

    ```python
    # in test_math_demo.py
    from math_demo import hyperbolic
    import numpy as np

    N_POINTS = 10

    def test_caternary_symmetric():
        x = np.asarray([2 ** i for i in range(N_POINTS)])
        assert np.all(hyperbolic.caternary(x) == hyperbolic.caternary(-x))
    ```

    Running `poetry run pytest` should show this test fails as expected.

6. Write the correct caternary equation

    ```python
    # in hyperbolic.py
    import numpy as np

    def caternary(x, a = 1):
        return a * np.cosh(x / a)
    ```

    See that tests pass.

7. Write a parabolic approximation and a script to plot it

    ```python
    # in hyperbolic.py
    def caternary_approx(x, a = 1):
        return a + x ** 2 / (2 * a)

    if __name__ == "__main__":
        import plotly.express as px

        x = np.linspace(-2, 2)
        a = 2

        f = caternary(x, a)
        g = caternary_approx(x, a)

        fig = px.line(x=x, y=f)
        fig.add_trace(px.line(x=x, y=g).data[0])
        fig.show()
    ```

    This approximation comes from a second order Maclaurin series.

    Running:

    ```bash
    poetry run python math_demo/hyperbolic.py
    ```

    will open the plot in your browser to show our caternary curves and decent
    parabolic approximations to them.

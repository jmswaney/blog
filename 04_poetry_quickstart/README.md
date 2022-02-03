# Poetry quickstart: a yarn-like package manager in Python

In this tutorial, we will create a Python package called `math-demo` using the `poetry` Python package manager. We will use `poetry` to scaffold out the `math-demo` package, install our dependencies in a virtual environment, and run scripts for unit testing and making plots.

After completing this tutorial, you will learn how to:

- Install the `poetry` package manager
- Setup a new Python package with `poetry`
- Setup VSCode to recognize `poetry` virtual environments
- Run unit tests and other scripts with `poetry`

## Installing the `poetry` package manager

To install `poetry`, we can use the following command from the documentation:

```zsh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

We can also install `poetry` tab completions to help us remember all subcommands and options from the command line. You can find the instructions for your particular shell [here](https://python-poetry.org/docs/#enable-tab-completion-for-bash-fish-or-zsh). If you are using "Oh-My-Zsh", then you'd run:

```zsh
# Only for Oh-My-Zsh
mkdir $ZSH_CUSTOM/plugins/poetry
poetry completions zsh > $ZSH_CUSTOM/plugins/poetry/_poetry
```

Once this is done, you need to add `poetry` to the list of plugins in your `~/.zshrc` file.

After restarting your terminal, running `poetry --version` should print your version of poetry to the terminal. You should also see tab completion listing out possible `poetry` commands if the tab completion was enabled correctly.

## Creating our `math-demo` package

The `starfox.main:main` part of the `console_scripts` option in `setup.py` is the syntax for referring to this `main` function.

## Installing our dependencies

## Setting up unit tests

## Caternary curves and their parabolic approximations

## Implementing the exact and approximate caternary calaculations

### Writing a failing test

### Making the tests pass

### Adding a parabolic approximation

## Visualizing exact and approximate caternary curves

## Final thoughts

In this tutorial, we learned how to setup a Python package with a CLI called `starfox`. We used `click` to bind command-line arguments to function arguments. We also used `questionary` to create a command-line wizard that simulates a conversation between characters in Star Fox 64.

![barrel-roll](https://i.makeagif.com/media/5-08-2015/ToQiiE.gif)

Do a barrel roll! You deserve it!

## Source Availability

All source materials for this article are available [here](https://github.com/jmswaney/blog/tree/main/04_poetry_quickstart) on my blog GitHub repo.

# Setting up Python Linting and Autoformatting in VSCode

![banner](https://unsplash.com/photos/YuQEEaNOgBA/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8N3x8b3JkZXJ8fDB8fHx8MTY0NDEyMjIzNw&force=true&w=1920)

In this tutorial, we will setup Python linting and automatic code formatting in VSCode. We will use `pylint` to highlight linting errors and `black` for autoformatting our code on save.

Linting and autoformatting are commonly used in Javascript projects and help to enforce consistent code standards across a team of developers working together in a single codebase. For anyone using Python on a team, `pylint` and `black` could help speed up development time, reduce upskilling time of new hires, and make your diffs easier to read for code reviews.

## Setup

First, let's start by making a new virtual environment for this tutorial called `python-linting`.

```zsh
conda create -y -n python-linting python=3.8
conda activate python-linting
```

Next, we'll make sure that we have our linter and formatter installed.

```zsh
conda install -y pylint black
```

Now that we have our tools installed, we can move on to setting them up in VSCode.

## Linting with `pylint`

To enable linting for Python in VSCode, we need to make sure some things are enabled in our settings. Be sure to make these changes to the "User" level (as opposed to the Workspace or folder only) so that it takes effect in any project.

First, open your VSCode Settings UI and search for "Python linting". Make sure that the following setting is checked:

> Python > Linting: **Enabled** \
> `[x]` Whether to lint Python files.

Also specifically allow `pylint` linting by making sure the following setting is checked:

> Python > Linting: **Pylint Enabled** \
> `[x]` Whether to lint Python files using pylint.

You can optionally supress "convention"-level linting messages if you want to only highlight more egregious issues. To disable convention messages, add `--disable=C` to the `pylint` arguments:

> Python > Linting: **Pylint Args** \
> `--disable=C`

Only do this part if you know that you don't want convention linting errors shown.

## Autoformatting with `black`

To setup autoformatting, we need to tell VSCode to use `black` as our formatter for Python code. We can do this in our settings by selecting `black` from the following dropdown selector:

> Python > Formatting: **Provide** \
> `black`

Next, we need to tell VSCode that we'd like to autoformat our code on save of the file. We can do this with the `Format on Save` option:

> Editor: **Format on Save** \
> `[x]` Format a file on save. A formatter must be available, the file must not be saved after delay, and the editor must not be shutting down.

## Result

If everything went well, you should get helpful linting messages while you code and automatic code formatting each time you save your file.

Show a gif of linting and autoformatting working on a dataclass.

## Final thoughts

In this tutorial, we setup `pylint` and `black` for Python linting and autoformatting in VSCode, respectively. Linting and autoformatting can help individual developers move faster be removing time spent debugging and thinking about formatting. They can also help a team of developers work together by enforcing a reasonable set of standards that strive to make their codebase and pull requests easier to read and review.

## Source availability

All source materials for this article are available [here](https://github.com/jmswaney/blog/tree/main/05_python_linting) on my blog GitHub repo.

# Tips for using Jupyter Notebooks with GitHub

Start with some motivation. Jupyter notebooks are awesome, GitHub is awesome, wouldn't it be great if using them together didn't suck?

To machine learning engineers and data scientists, Jupyter notebooks have become a workhorse tool in the toolbox. Notebooks allow researchers to quickly prototype new solutions and make interactive visualizations in a remote computing environment. Unfortunately, the ease-of-use and interactivity of notebooks comes with some trade-offs when using version control.

Version control tools like `git` are powerful collaboration tools that track changes to source code and synchronize local and remote copies of a shared codebase. They allow developers to work together on the same codebase and seamlessly merge their improvements together. These tools also work out of the box with services like GitHub, which provide hosting for the shared source codebase. Unfortunately, most of the user experience of using `git` and GitHub has been designed around text-based source code, and Jupyter notebooks are saved with embedded output media in a JSON format within our familiar `.ipynb` files.

This guide shares some tips and tricks for working with Jupyter notebooks under version control. If you work with notebooks and are on a team of machine learning engineers or data scientists, then you may find this guide particularly useful. As an example, we will make notebooks that compute 50 and 200-day moving averages (MA) of any stock ticker available on Yahoo Finance.

After completing this guide, you will learn how to:

- Setup auto-formatting in notebooks
- Use VSCode's built-in tools for viewing notebook diffs
- Use `nbstripout` with `git` to commit only notebook source
- Use papermill to create a shared notebook library
- Refactor shared notebook code into Python modules
- And fetch some stock market data in Python!

## The `patagonia` Python project

Let's imagine we want to make a notebook that fetches the previous 5 years of closing prices for a given stock ticker, computes the 50-MA and 200-MA, and visualizes all three time series in a single plot.

- Setup virtual environment with all dependencies

```zsh
conda create -y -n python-linting python=3.8 
conda activate python-linting
conda install -y -c conda-forge pylint black jupyterlab jupyterlab_code_formatter
```

Then start a Jupyter server.

```zsh
jupyter lab
```

This will open up Jupyter in your browser.

Create a new notebook and you will see a `Format notebook` option.

### Tip #1: Auto-formatting in Jupyter Notebooks

Before we even use version control, it's helpful to setup auto-formatting to give us some reasonable formatting standards across our notebooks. This will help make our changes easier to read within version control because our code will be committed with consistent formatting that is also designed to make diffs as small as they can be.

Open up the `Advanced Settings Editor` via the Settings dropdown or by pressing `Cmd-,` on Mac.

Here you will see the Jupyterlab Code Formatter settings for customization. We also have a section to add keyboard shortcuts. Let's add a shortcut for formatting the entire notebook.

```json
# Keyboard Shortcuts > User Preferences
{
  "shortcuts": [
    {
      "command": "jupyterlab_code_formatter:format_all",
      "keys": [
        "Ctrl K",
        "Ctrl M"
      ],
      "selector": ".jp-Notebook.jp-mod-editMode"
    }
  ]
}

```

After saving and returning to your notebook, you should see this shortcut works for formatting the entire Jupyter notebook.

Return to Advanced Settings Editor, go to Jupyterlab Code Formatter. Under User Preferences, add the following:

```json
{
  "formatOnSave": true
}
```

After saving and return to your notebook, you should see that simply saving the notebook automatically formats your code.

### Tip #2: Built-in VSCode tools

First of all, mention that VSCode has great tools for looking at notebook diffs.

- Running notebooks inside VSCode
- Viewing notebook diffs in VSCode

### Tip #3: nbstripout

Allows local copies, only commit the code. Refer to autoformatting post.
Combined with the autoformatting, you are left with readable diffs even though
your source code is embedded inside a JSON object. Show before and after diff.

- Setup the git plugin
- Show before and after diff with a simple example

### Tip #4: Papermill input / output folders with gitignore

Papermill is a cool way to execute notebooks.

Gives consistent code formatting across the team of developers. Show before and
after.

### Tip #5: A shared library for all notebooks

It's often useful to factor out code to be re-used across notebooks. This is
generally a better experience while viewing diffs on GitHub too.

## Final thoughts

Jupyter notebooks and `git` are powerful tools for machine learning engineers and data scientists to prototype solutions and collaborate on a shared codebase, respectively. We discussed some tips and tricks for getting the best of both worlds from notebooks and version control in our simple financial example. 

## Source availability

All source materials for this article are available [here](https://github.com/jmswaney/blog/tree/main/06_notebooks_in_github) on my blog GitHub repo.

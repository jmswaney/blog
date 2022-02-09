# Tips for using Jupyter Notebooks with GitHub

Start with some motivation. Jupyter notebooks are awesome, GitHub is awesome, wouldn't it be great if using them together didn't suck?

## Auto-formatting Jupyter Notebooks

### Setup

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

### Adding shortcuts

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

## Notebooks under version control

First of all, mention that VSCode has great tools for looking at notebook diffs.

### Option 1: nbstripout

Allows local copies, only commit the code. Refer to autoformatting post.
Combined with the autoformatting, you are left with readable diffs even though
your source code is embedded inside a JSON object. Show before and after diff.

### Option 2: Papermill input / output folders with gitignore

Papermill is a cool way to execute notebooks.

Gives consistent code formatting across the team of developers. Show before and
after.

### Bonus: A shared library for all notebooks

It's often useful to factor out code to be re-used across notebooks. This is
generally a better experience while viewing diffs on GitHub too.

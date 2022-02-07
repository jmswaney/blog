# Auto-formatting Jupyter Notebooks

## Setup

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

## Adding a keyboard shortcut

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

## Adding format on save

Return to Advanced Settings Editor, go to Jupyterlab Code Formatter. Under User Preferences, add the following:

```json
{
  "formatOnSave": true
}
```

After saving and return to your notebook, you should see that simply saving the notebook automatically formats your code.

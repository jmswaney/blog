# Setting up Python Linting and Autoformatting in VSCode

## Before you start... virtual envs

```bash
poetry add -D pylint black
```

## Linting with pylint

Python > Linting > pylint enabled: check true

To disable convention messages:

Python > Linting > pylint args

--disable=C

## Autoformatting

In settings UI, add Python > Formatting Provider > black

Editor > Format on Save: check true

## Jupyter notebooks

Get it working in notebooks with `black[jupyter]`

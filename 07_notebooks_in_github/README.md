# Jupyter Notebook Development using Github

## Option 1: nbstripout

Allows local copies, only commit the code. Refer to autoformatting post.
Combined with the autoformatting, you are left with readable diffs even though
your source code is embedded inside a JSON object. Show before and after diff.

## Option 2: Papermill input / output folders with gitignore

Papermill is a cool way to execute notebooks.

Gives consistent code formatting across the team of developers. Show before and
after.

## Bonus: A shared library for all notebooks

It's often useful to factor out code to be re-used across notebooks. This is
generally a better experience while viewing diffs on GitHub too.

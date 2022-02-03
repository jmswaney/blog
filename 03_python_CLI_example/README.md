# Creating beautiful command-line interfaces in Python

In this tutorial, we will create a Python package called `starfox` with a command-line interface (CLI) that simulates conversations between players from *Star Fox 64*. We will create the CLI using the `click` package and add a wizard using `questionary` for conveniently choosing which characters to include in our simulated conversation.

After completing this tutorial, you will learn how to:

- Setup a Python package with custom entrypoints
- Use editable mode to see the effects of our changes as we develop
- Define a simple CLI with built-in help messages
- Build a command-line wizard to easily run CLI commands

## Setting up the `starfox` Python package

First, create a folder for this tutorial along with a new file called `setup.py` inside. We'll use the `setuptools` package to help describe our new `starfox` package.

```python
"""starfox setup.py
"""
from setuptools import setup, find_packages

setup(
    name="starfox",
    version="0.1.0",
    description="Do a barrel roll!",
    packages=find_packages(),
    install_requires=[
        "click",
        "questionary"
    ],
    entry_points={
        'console_scripts': ['starfox=starfox.main:main']
    },
    author="Justin Swaney",
    license="MIT"
)
```

Some things to note about this `setup` function call:

- The `find_packages` function will auto-discover our package
- The `install_requires` argument specifies *minimum* dependencies to run
- The `entry_points` argument allows us to name our CLI `starfox` and bind it to a function called `main` within the `starfox/main.py` module (which we still need to write)

Make a folder called "*starfox/*" next to our `setup.py` file. This folder will hold all of the source code for our new package. Inside the starfox folder, create an empty file called `__init__.py`. The existence of this file will cause Python to consider our starfox folder as a package.

> This may seem dubious with our simple example, but an `__init__.py` file allows us to namespace our package modules and execute common initialization code when imported.

Next to the `__init__.py` file in the starfox folder, create a file called `main.py` and define a function called `main` inside.

```python
def main():
    print("Do a barrel roll!")
```

The `starfox.main:main` part of the `console_scripts` option in `setup.py` is the syntax for referring to this `main` function.

## Installing our new package

We still need to edit our `starfox` code, but it would be nice if we could test if we've set things up correctly first. Let's just install it in `editable` mode. This means the package metadata and compiled byte code will be stored within our current folder instead of a distant `site-packages` folder with the rest of the installed packages on the system.

> It's helpful to add `.egg-info` and `__pycache__` to a `.gitignore` file when working in `editable` mode so that they do not pollute the git history.

To install `starfox` in the base environment, we can use pip.

```bash
# In the same folder as setup.py
$ pip install -e .
```

If you want to test this out in a clean environment, you can create a new conda environment and install `starfox` there. Just remember you'll have to activate your new environment if you want to use the `starfox` package.

```bash
# In the same folder as setup.py
$ conda create -y -n starfox python=3.7
$ conda activate starfox
(starfox) $ pip install -e .
```

Finally, we can test if our `starfox` console script works.

```text
$ starfox
Do a barrel roll!
```

## Defining the `starfox` CLI

Now that we have our entry point set up, let's build a CLI that will print random quotes from a given character in the game *Star Fox 64*. To keep things simple, we will only consider the following characters:

- [Fox McCloud](https://www.youtube.com/watch?v=NBYLyRVVS0g)
- [Falco Lombardi](https://www.youtube.com/watch?v=6oC0SW8i6k8)
- [Slippy Toad](https://www.youtube.com/watch?v=_S6X0UN_a_A)
- [Peppy Hare](https://www.youtube.com/watch?v=lQPfQQrUnJM)
- [General Pepper](https://www.youtube.com/watch?v=RxZ5LSNOG7k)
- [Andross](https://www.youtube.com/watch?v=H7ifKpPoIrk)

Create a file called `quotes.py` next to our `main.py` module. Fill it with these character quotes:

```python
"""Quotes from characters in Star Fox 64"""

FOX = [
    "All aircraft report!",
    "I'll go it alone from here",
    "Sorry to jet, but I'm in a hurry"
]

FALCO = [
    "Enemy group behind us!",
    "AaawwwwWW man, I'm gonna have ta BACK OFF",
    "Hey Einstein, I'm on yourrr siiide!!"
]

SLIPPY = [
    "Don't worry, Slippy's here!",
    "Hold A to charge your laser",
    "This baby can take temperatures up to 9000 degrees!"
]

PEPPY = [
    "It's quiet, TOO quiet...",
    "Do a barrel roll!!!",
    "You've got an enemy on your tail!"
]

GENERAL = [
    "It's about time you showed up, Fox. You're the only hope for our world!",
    "Recover our base from the enemy army",
    "Star Fox, we are in your debt"
]

ANDROSS = [
    "Ahhh, the son of James McCloud",
    "I've been waiting for you, Star Fox",
    "Only I have the brains to rule Lylat!"
]

QUOTES = dict(
    fox=FOX,
    falco=FALCO,
    slippy=SLIPPY,
    peppy=PEPPY,
    general=GENERAL,
    andross=ANDROSS
)
```

Inside `main.py`, let's import `QUOTES` and print `QUOTES.keys()` instead. We can test that the characters show up in the terminal.

```python
# Inside main.py
from starfox.quotes import QUOTES

def main():
    print(QUOTES.keys())

# In the terminal
$ starfox
dict_keys(['fox', 'falco', 'slippy', 'peppy', 'general', 'andross'])
```

Now that we have `QUOTES` available, it would be nice if we could select which character we would like to quote directly from the terminal. We can use the `click` package to add a `character` argument to our `main` function and bind it to a command-line argument.

```python
# Inside main.py
import click
from starfox.quotes import QUOTES

@click.command()
@click.argument('character')
def main(character):
    for k, q in enumerate(QUOTES[character.lower()]): 
        print(f'Quote #{k + 1}: {q}')
```

If we try running `starfox` in the terminal, we will get an error because the `character` argument is a required positional argument (named arguments have dashes in front of them). If we also indicate a character in the terminal, we'll see all quotes for that character:

```text
$ starfox falco
Quote #1: Enemy group behind us!
Quote #2: AaawwwwWW man, I'm gonna have ta BACK OFF
Quote #3: Hey Einstein, I'm on yourrr siiide!!
```

The last thing we need to do is randomly sample from these quotes. We can use the `random.sample` built-in to do this.

```python
# Inside main.py
from random import sample
import click
from starfox.quotes import QUOTES

@click.command()
@click.argument('character')
def main(character):
    qs = QUOTES[character.lower()]
    q = sample(qs, k=1)[0]
    print(f'{character.upper()} - "{q}"')
```

Now we'll get random quotes from the given character.

```text
$ starfox slippy
SLIPPY - "Don't worry, Slippy's here!"
$ starfox slippy
SLIPPY - "This baby can take temperatures up to 9000 degrees!"
```

## Creating a command-line wizard

Now that we can print random quotes, let's simulate a conversation between Star Fox 64 characters that the user selects. Ideally, we would re-use the `starfox` CLI code for this conversation feature and still support the previous random quote feature. Fortunately we can do this with `@click.option`, which will make our `character` input an optional named argument. This is convenient because we can alter the behavior of the `starfox` CLI based on whether or not the `character` option is provided (not `None`).

For our simulated conversation, we can prompt the user for which characters to include and how many quotes from each to sample. With that in mind, it would be convenient to re-use the random sampling logic we have already written for this as well as the above mentioned case when `character` is provided. Putting this all together, our `main.py` file would look like this.

```python
from random import sample
import click
import questionary
from starfox.quotes import QUOTES

def quote_character(character):
    """Prints a random quote from a given Star Fox character"""
    qs = QUOTES[character.lower()]
    q = sample(qs, k=1)[0]
    print(f'{character.upper()} - "{q}"')

@click.command()
@click.option('-c', '--character')
def main(character):
    """The `starfox` CLI"""
    if character:
        quote_character(character)
        return
    answers= questionary.form(
        characters = questionary.checkbox("Select characters", choices=QUOTES.keys()),
        n_iter = questionary.text("How long do you want the conversation to be? (int)")
    ).ask()
    for _ in range(int(answers['n_iter'])):
        for c in answers['characters']: 
            quote_character(c)
```

Notice that we have factored out the random quote logic to a function called `quote_character`. We then use this function when the user provides a `character` as well as in our simulated conversation. We have changed our `@click.argument` to a `@click.option` to make it an optional named argument. We also return before getting to the simulated conversation when `character` is specified. If the user were to run `starfox` in the terminal, `character` would be `None` and we would skip right to the `questionary` form.

`questionary` is a Python package for making command-line wizards. Here, we use it to prompt the user for two pieces of information: which characters to inlcude in our conversation and how many quotes to sample from each character. Once we get this information, we simply use our `quote_character` function in a loop that goes around the horn.

Let's check that we didn't break anything from before.

```text
$ starfox -c general
GENERAL - "It's about time you showed up, Fox. You're the only hope for our world!"
```

Looks good. Now let's try our new wizard.

```text
$ starfox

? Select characters (Use arrow keys to move, <space> to select, <a> to toggle, <i> to invert)
   ○ fox
   ● falco
   ● slippy
   ● peppy
   ○ general
 » ● andross

? How long do you want the conversation to be? (int) 2

FALCO - "Hey Einstein, I'm on yourrr siiide!!"
SLIPPY - "This baby can take temperatures up to 9000 degrees!"
PEPPY - "Do a barrel roll!!!"
ANDROSS - "Ahhh, the son of James McCloud"
FALCO - "Hey Einstein, I'm on yourrr siiide!!"
SLIPPY - "This baby can take temperatures up to 9000 degrees!"
PEPPY - "You've got an enemy on your tail!"
ANDROSS - "Only I have the brains to rule Lylat!"
```

Nice! I suggest reading your results out loud to get the full effect.

## Final thoughts

In this tutorial, we learned how to setup a Python package with a CLI called `starfox`. We used `click` to bind command-line arguments to function arguments. We also used `questionary` to create a command-line wizard that simulates a conversation between characters in Star Fox 64.

![barrel-roll](https://i.makeagif.com/media/5-08-2015/ToQiiE.gif)

Do a barrel roll! You deserve it!

## Source Availability

All source materials for this article are available [here](https://github.com/jmswaney/blog/tree/main/03_python_CLI_example) on my blog GitHub repo.

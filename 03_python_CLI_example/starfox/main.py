# from random import sample
# import click
# from starfox.quotes import QUOTES

# @click.command()
# @click.argument('character')
# def main(character):
#     qs = QUOTES[character.lower()]
#     q = sample(qs, k=1)[0]
#     print(f'{character.upper()} - "{q}"')


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
    # wizard
    answers= questionary.form(
        characters = questionary.checkbox("Select characters", choices=QUOTES.keys()),
        n_iter = questionary.text("How long do you want the conversation to be? (int)")
    ).ask()
    for _ in range(int(answers['n_iter'])):
        for c in answers['characters']: 
            quote_character(c)

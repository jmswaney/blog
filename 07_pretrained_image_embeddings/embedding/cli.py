from embedding.io import load, save
from embedding.embed import embed, umap
from embedding.plot import plot_umap, plot_pca
import os
import click

MODEL_TYPES = ['pca', 'resnet']

@click.command()
@click.argument("input")
@click.argument("output")
@click.option("--model", "-m", required=True, type=click.Choice(MODEL_TYPES))
@click.option("--dimensions", "-d", "n_dim", default=32, type=int)
@click.option("--layers", "-l", type=int, default=None)
@click.option("--compress", "-c", is_flag=True)
@click.option("--plot", "-p", is_flag=True)
@click.option("--seed", "-s", type=int, default=None)
@click.option("--batch-size", "-b", type=int, default=None)
def cli(input, output, model, n_dim, layers, compress, plot, seed, batch_size):
    x, y = load(input)
    print("Input shape", x.shape)
    x_embed = embed(x, model, n_dim, batch_size, layers)
    print("Embedded shape", x_embed.shape)
    save(output, x_embed, compress)
    if plot:
        root, ext = os.path.splitext(output)
        name = os.path.basename(root)
        if x_embed.shape[-1] == 2:
            plot_pca(x_embed, y, f"{name}_pca.png")
        else:
            x_umap = umap(x_embed, seed)
            print("UMAP shape", x_umap.shape)
            save(f"{name}_umap{ext}", x_umap, compress)
            plot_umap(x_umap, y, f"{name}_umap.png")
    
import argparse
import random
from typing import List

from skimage.draw import polygon
from skimage.metrics import normalized_root_mse
from deap import base, creator, tools
from PIL import Image
import numpy as np


from pathlib import Path
from tqdm.auto import trange

GENE_SIZE: int = 12 # 4 2D vertices and RGBA


def evaluate(individual: List[float], ref_img: np.ndarray) -> float:
    create_gen_img_array = np.zeros(ref_img.shape, dtype=np.uint8)
    gen_img = Image.fromarray(create_gen_img_array)
    genome_size = len(individual) // GENE_SIZE
    for i in range(genome_size):
        color = np.rint(individual[i + 8:i + 12].copy() * 255).astype(np.uint8)
        poly = individual[i * 12:i * 12 + 8].copy().reshape(-1, 2)
        poly[:, 0] = np.rint(poly[:, 0] * ref_img.shape[0])
        poly[:, 1] = np.rint(poly[:, 1] * ref_img.shape[1])
        create_img_array = np.zeros(ref_img.shape, dtype=np.uint8)
        rr, cc = polygon(poly[:, 0], poly[:, 1], create_img_array.shape)
        create_img_array[rr, cc] = color
        layer = Image.fromarray(create_img_array)
        gen_img.paste(layer, (0, 0), layer)
    gen_array = np.array(gen_img)
    gen_array[:, :, 3] = 255
    return normalized_root_mse(ref_img, gen_array)


def main(input_file: Path, output_file: Path, genome_size: int, pop_size: int, generations: int):
    """
    Entry point and main logic for script to convert image to Excel file
    :param input_file: path of the image file to read
    :param output_file: path of the xlsx file to write
    :param column_width: width of the columns
    :param row_height: height of the rows
    """
    img = Image.open(input_file)
    ref_img_array = np.array(img)


    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=genome_size * GENE_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, ref_img=ref_img_array)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="The name of the image file to read", type=str
    )
    parser.add_argument(
        "output_file", help="The name of the image file to write", type=str
    )
    parser.add_argument(
        "genome_size",
        help="The number of primitive shapes in an individual genome",
        type=int,
        default=10,
    )
    parser.add_argument(
        "pop_size",
        help="The width of the columns in the spreadsheet",
        type=int,
        default=10,
    )
    parser.add_argument(
        "generations",
        help="The height of the rows in the spreadsheet",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    main(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        genome_size=args.genome_size,
        pop_size=args.column_width,
        generations=args.row_height,
    )

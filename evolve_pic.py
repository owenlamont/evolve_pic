import argparse
import random
from typing import List, Tuple

from skimage.draw import polygon
from skimage.metrics import normalized_root_mse
from deap import base, creator, tools, algorithms
from PIL import Image
import numpy as np
import multiprocessing


from pathlib import Path
from tqdm.auto import trange

GENE_SIZE: int = 12 # 4 2D vertices and RGBA


def evaluate(individual: List[float], ref_img: np.ndarray) -> Tuple[float]:
    im_shape = ref_img.shape
    gen_img = express_genome_to_image(np.array(individual), im_shape)
    gen_array = np.array(gen_img)
    gen_array[:, :, 3] = 255
    error = normalized_root_mse(ref_img, gen_array)
    return error.item(),


def express_genome_to_image(individual: np.ndarray, im_shape):
    individual_bounded = np.clip(individual, 0, 1)
    create_gen_img_array = np.zeros(im_shape, dtype=np.uint8)
    gen_img = Image.fromarray(create_gen_img_array)
    genome_size = len(individual_bounded) // GENE_SIZE
    for i in range(genome_size):
        color = np.rint(individual_bounded[i + 8:i + 12].copy() * 255).astype(np.uint8)
        poly = individual_bounded[i * 12:i * 12 + 8].copy().reshape(-1, 2)
        poly[:, 0] = np.rint(poly[:, 0] * im_shape[0])
        poly[:, 1] = np.rint(poly[:, 1] * im_shape[1])
        create_img_array = np.zeros(im_shape, dtype=np.uint8)
        rr, cc = polygon(poly[:, 0], poly[:, 1], create_img_array.shape)
        create_img_array[rr, cc] = color
        layer = Image.fromarray(create_img_array)
        gen_img.paste(layer, (0, 0), layer)
    return gen_img


# toolbox needs to be in global scope for multiprocessing to work on Windows
toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def main(input_file: Path, output_file: Path, genome_size: int, pop_size: int, generations: int):
    """
    Entry point and main logic for script to convert image to Excel file
    :param input_file: path of the image file to read
    :param output_file: path of the xlsx file to write
    :param genome_size: Number of genes in the genome
    :param pop_size: Size of the population to evolve
    :param generations: Number of generations to optimise over
    """
    pool = multiprocessing.Pool(processes=16)
    toolbox.register("map", pool.map)
    img = Image.open(input_file)
    ref_img_array = np.array(img)

    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=genome_size * GENE_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    toolbox.register("evaluate", evaluate, ref_img=ref_img_array)

    hof = tools.HallOfFame(generations)
    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations,
                                   stats=stats, halloffame=hof, verbose=True)
    best = express_genome_to_image(hof[0], ref_img_array.shape)
    best.save(output_file)


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
        pop_size=args.pop_size,
        generations=args.generations,
    )

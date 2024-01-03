import os
import tifffile
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imageprocessing import ImageProcessing


# Define argument parser
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--inputPath", "-ip", type=str, required=True)
argument_parser.add_argument("--outputPath", "-op", type=str, required=True)
args = argument_parser.parse_args()


# Define paths variables
DATA = Path(args.inputPath)
OUTPUT = Path(args.outputPath)


# check if directories exist
if not (Path.exists(DATA) and Path.is_dir(DATA)):
    raise FileNotFoundError("Input directory doesn't exist")


# Check if output dir exists
if not Path.is_dir(OUTPUT):
    raise FileNotFoundError("Output path isn't directory")
if not Path.exists(OUTPUT):
    warnings.warn("Output path doesn't exist. Will be created automatically")
    os.mkdir(OUTPUT)


# Read and process data
ndvi1 = tifffile.imread(DATA / "ndvi1.TIF")
ndvi2 = tifffile.imread(DATA / "ndvi2.TIF")
classes = tifffile.imread(DATA / "classes.TIF")


# Reshape
ndvi1 = ImageProcessing.add_blank(ndvi1, 8192, 8192)
ndvi2 = ImageProcessing.add_blank(ndvi2, 8192, 8192)
classes = ImageProcessing.add_blank(classes, 8192, 8192)
ndvi1.shape += (1,)
ndvi2.shape += (1,)
classes.shape += (1,)


# Split images
ndvi1 = ImageProcessing.split_img(ndvi1, 64).astype("float32")
ndvi2 = ImageProcessing.split_img(ndvi2, 64).astype("float32")
classes = ImageProcessing.split_img(classes, 64).astype("uint8")
ndvi1 = ndvi1.reshape((-1, 64, 64, 1))
ndvi2 = ndvi2.reshape((-1, 64, 64, 1))
classes = classes.reshape((-1, 64, 64, 1))

os.mkdir(OUTPUT / "x_data/")
os.mkdir(OUTPUT / "y_data/")

for i in tqdm(range(ndvi1.shape[0])):
    tifffile.imsave(
        OUTPUT / "x_data/" / (str(i) + ".TIF"), np.concatenate((ndvi1[i], ndvi2[i]), -1)
    )
    tifffile.imsave(OUTPUT / "y_data/" / (str(i) + ".TIF"), classes[i])

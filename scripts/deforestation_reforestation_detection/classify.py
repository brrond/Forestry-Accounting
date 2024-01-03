from pyproj import Proj, transform
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from rasterio import mask as msk
from pathlib import Path
import numpy as np
import tifffile
import rasterio
import argparse
import json
import re
import os


# Define argument parser
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--firstInputPath", "-fip", type=str, required=True)
argument_parser.add_argument("--secondInputPath", "-sip", type=str, required=True)
argument_parser.add_argument("--outputPath", "-op", type=str, required=True)
args = argument_parser.parse_args()


# Define paths variables
FIRST = Path(args.firstInputPath)
SECOND = Path(args.secondInputPath)
OUTPUT = Path(args.outputPath)
FIRST_BANDS = list(
    filter(lambda f: re.compile("B[0-9]*.TIF").search(f), os.listdir(FIRST))
)
SECOND_BANDS = list(
    filter(lambda f: re.compile("B[0-9]*.TIF").search(f), os.listdir(SECOND))
)


# Get coordinate reference system
CRS = rasterio.open(FIRST / FIRST_BANDS[0]).crs


# check if directories exist
if not (Path.exists(FIRST) and Path.is_dir(FIRST)):
    raise FileNotFoundError("First input directory doesn't exist")
if not (Path.exists(SECOND) and Path.is_dir(SECOND)):
    raise FileNotFoundError("Second input directory doesn't exist")


# FInd json file in each dir and check if they exist
f1 = list(
    filter(
        None, [f if "_SR_stac.json" in str(f) else None for f in Path(FIRST).glob("*")]
    )
)
f2 = list(
    filter(
        None, [f if "_SR_stac.json" in str(f) else None for f in Path(SECOND).glob("*")]
    )
)
if f1 is None or len(f1) == 0 or f2 is None or len(f2) == 0:
    raise FileNotFoundError("*_SR_stac.json doesn't exist")


# Parse coordinates
f1 = f1[0]
f2 = f2[0]
coordinates1 = []
coordinates2 = []
with open(f1, "r") as file:
    json_decoder = json.load(file)
    geometry = json_decoder["geometry"]
    coordinates1 = geometry["coordinates"][0]
with open(f2, "r") as file:
    json_decoder = json.load(file)
    geometry = json_decoder["geometry"]
    coordinates2 = geometry["coordinates"][0]


# Get minimum from each pair for each vertex
coordinates = []
for i in range(4):
    coordinates.append([])
    for j in range(2):
        coordinates[-1].append(min(coordinates1[i][j], coordinates2[i][j]))


# Transform coordinates
in_proj = Proj(init="epsg:4326")
out_proj = Proj(init=str(CRS).lower())

new_coordinates = [
    transform(in_proj, out_proj, coord[0], coord[1]) for coord in coordinates
]
p = Polygon(new_coordinates)


# Check if output dir exists
if not (Path.exists(OUTPUT) and Path.is_dir(OUTPUT)):
    raise FileNotFoundError("Output directory doesn't exist")


# Read and process first path
b1 = rasterio.open(FIRST / FIRST_BANDS[1])
r = rasterio.open(FIRST / FIRST_BANDS[3])
nir = rasterio.open(FIRST / FIRST_BANDS[4])

b1 = msk.mask(b1, [p], crop=True)[0]
r = msk.mask(r, [p], crop=True)[0]
nir = msk.mask(nir, [p], crop=True)[0]

b1 = b1[0].astype("float64")
r = r[0].astype("float64")
nir = nir[0].astype("float64")

ndvi1 = np.nan_to_num((nir - r) / (nir + r))
del nir, r

# get ndvi
ndvi1 = ndvi1.astype("float32")

# b1 (aerosol?) > 0.4 => clouds
b1 = np.where(b1 / (2**16) > 0.4, 1, 0).astype("uint8")

# define output classified image
ndvi_classification1 = np.zeros(ndvi1.shape + (3,)).astype("uint8")
classes1 = np.zeros(ndvi1.shape + (1,)).astype("uint8")
ndvi_classification1[(ndvi1 < 0.02) & (ndvi1 != 0)] = [0, 0, 255]  # water body
classes1[(ndvi1 < 0.02) & (ndvi1 != 0)] = 1
ndvi_classification1[b1 == 1] = [255, 255, 255]  # clouds
classes1[b1 == 1] = 2
ndvi_classification1[(ndvi1 >= 0.02) & (ndvi1 < 0.12)] = [
    128,
    128,
    128,
]  # shadow/buildings
classes1[(ndvi1 >= 0.02) & (ndvi1 < 0.12)] = 3
ndvi_classification1[(ndvi1 > 0.15) & (ndvi1 < 0.2)] = [128, 128, 0]  # bair soil/sand
classes1[(ndvi1 > 0.15) & (ndvi1 < 0.2)] = 4
ndvi_classification1[(ndvi1 >= 0.2) & (ndvi1 < 0.4)] = [0, 256, 0]  # low
classes1[(ndvi1 >= 0.2) & (ndvi1 < 0.4)] = 5
ndvi_classification1[(ndvi1 >= 0.4)] = [0, 128, 0]  # huge
classes1[(ndvi1 >= 0.4)] = 6

ndvi_classification1 = ndvi_classification1.astype("uint8")
tifffile.imsave(OUTPUT / "ndvi_classification1.TIF", ndvi_classification1)
tifffile.imsave(OUTPUT / "ndvi1.TIF", ndvi1)
# tifffile.imshow(ndvi_classification1.astype('uint8'), title='ndvi1 classes')


# Read and process second path
b1 = rasterio.open(SECOND / SECOND_BANDS[1])
r = rasterio.open(SECOND / SECOND_BANDS[3])
nir = rasterio.open(SECOND / SECOND_BANDS[4])

b1 = msk.mask(b1, [p], crop=True)[0]
r = msk.mask(r, [p], crop=True)[0]
nir = msk.mask(nir, [p], crop=True)[0]

b1 = b1[0].astype("float64")
r = r[0].astype("float64")
nir = nir[0].astype("float64")

ndvi2 = np.nan_to_num((nir - r) / (nir + r))
del nir, r

# get ndvi
ndvi2 = ndvi2.astype("float32")

# b1 (aerosol?) > 0.4 => clouds
b1 = np.where(b1 / (2**16) > 0.4, 1, 0).astype("uint8")

# define output classified image
ndvi_classification2 = np.zeros(ndvi2.shape + (3,)).astype("uint8")
classes2 = np.zeros(ndvi2.shape + (1,)).astype("uint8")
ndvi_classification2[(ndvi2 < 0.02) & (ndvi2 != 0)] = [0, 0, 255]  # water body
classes2[(ndvi2 < 0.02) & (ndvi2 != 0)] = 1
ndvi_classification2[b1 == 1] = [255, 255, 255]  # clouds
classes2[b1 == 1] = 2
ndvi_classification2[(ndvi2 >= 0.02) & (ndvi2 < 0.12)] = [
    128,
    128,
    128,
]  # shadow/buildings
classes2[(ndvi2 >= 0.02) & (ndvi2 < 0.12)] = 3
ndvi_classification2[(ndvi2 > 0.15) & (ndvi2 < 0.2)] = [128, 128, 0]  # bair soil/sand
classes2[(ndvi2 > 0.15) & (ndvi2 < 0.2)] = 4
ndvi_classification2[(ndvi2 >= 0.2) & (ndvi2 < 0.4)] = [0, 256, 0]  # low
classes2[(ndvi2 >= 0.2) & (ndvi2 < 0.4)] = 5
ndvi_classification2[(ndvi2 >= 0.4)] = [0, 128, 0]  # huge
classes2[(ndvi2 >= 0.4)] = 6

ndvi_classification2 = ndvi_classification2.astype("uint8")
tifffile.imsave(OUTPUT / "ndvi_classification2.TIF", ndvi_classification2)
tifffile.imsave(OUTPUT / "ndvi2.TIF", ndvi2)
# tifffile.imshow(ndvi_classification2.astype('uint8'), title='ndvi2 classes')


# Calc ndiv diff
ndvi_diff = ndvi2 - ndvi1
# tifffile.imshow(ndvi_diff, title='ndvi change')


# Create deforestation map
classes = np.zeros(classes1.shape + (1,))

# de- ,-forestation
# forest change
classes[(classes1 == 6) & (classes2 == 5)] = 1
classes[(classes1 == 5) & (classes2 == 4)] = 1
classes[(classes1 == 4) & (classes2 == 5)] = 1
classes[(classes1 == 5) & (classes2 == 6)] = 1
classes[(classes1 == 5) & (classes1 == 5)] = 1
classes[(classes1 == 6) & (classes1 == 6)] = 1

# water
classes[(classes1 == 1) | (classes2 == 1)] = 4

# no data
classes[(classes1 == 0) | (classes2 == 0)] = 0
classes[(classes1 == 2) | (classes2 == 2)] = 0
classes[(classes1 == 3) | (classes2 == 3)] = 0

deforestation_map = np.zeros(ndvi_classification1.shape)
classes.shape = (
    ndvi_diff.shape[0],
    ndvi_diff.shape[1],
)  # remove last axis
eps = 0.05

deforestation_map[(classes == 1) & (ndvi_diff > eps)] = [0, 255, 0]  # forestation
classes[(classes == 1) & (ndvi_diff > eps)] = 2

deforestation_map[(classes == 1) & (ndvi_diff < -eps)] = [255, 0, 0]  # deforestation
classes[(classes == 1) & (ndvi_diff < -eps)] = 1

deforestation_map[(classes == 1) & (ndvi_diff >= -eps) & (ndvi_diff <= eps)] = [
    255,
    255,
    0,
]  # no change
classes[(classes == 1) & (ndvi_diff >= -eps) & (ndvi_diff <= eps)] = 3

deforestation_map[(classes == 4)] = [0, 0, 255]
deforestation_map[(classes == 0)] = [0, 0, 0]
tifffile.imsave(OUTPUT / "dmap.TIF", deforestation_map.astype("uint8"))

classes = classes.astype("uint8")
tifffile.imsave(OUTPUT / "classes.TIF", classes)

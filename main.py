# using tkinter backend
from gui.tk import *

# to work with Landsat directory
from pathlib import Path
import dateutil.parser
import json
import os
import re

# general purpose libs
import multiprocessing
import numpy as np
import cv2

# coord modules
from pyproj import Proj, transform as trns
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from rasterio import mask as msk
import rasterio


class Loader:
    """
    A class that represents abstraction to load landsat data using rasterio.
    """

    def __init__(self, dir_: Path):
        """
        :param dir_: directory to landsat folder.
        """

        # save directory path and get bands
        self.dir = dir_
        self.bands = list(filter(lambda f: re.compile('B[0-9]*.TIF').search(f), os.listdir(self.dir)))

        # get coordinate reference system
        a = rasterio.open(self.dir / self.bands[0])
        self.crs = a.crs
        del a

    def load(self, band: int):
        """
        A method that allows band obtaining.

        :param band: number of the band to obtain.

        :return: rasterio DatasetReader.
        """

        return rasterio.open(self.dir / self.bands[band])

    def rgb(self, coordinates=None):
        """
        A method that returns RGB image of current satellite snapshot.

        :param coordinates: optional parameter that allows crop to specified coordinates. Coordinates must be in the same CRS.

        :return: numpy array that represents RGB image as unsigned int8 (1 byte) array with HEIGHT x WIDTH x 3.
        """

        # load DatasetReader for blue, green and red channels
        b = self.load(1)
        g = self.load(2)
        r = self.load(3)

        if coordinates is None:
            # TODO: Return full size image
            pass

        # crop to coordinates
        # get actual image [0]
        # *[1] - represents mask transformation (Affine)
        r = msk.mask(r, coordinates, crop=True)[0]
        g = msk.mask(g, coordinates, crop=True)[0]
        b = msk.mask(b, coordinates, crop=True)[0]

        # cast to float64 datatype (the biggest available by default)
        r = r[0].astype('float64')
        g = g[0].astype('float64')
        b = b[0].astype('float64')

        # merge with cv2
        # simply combines three channels into one array
        rgb = cv2.merge((r, g, b))
        rgb = (rgb / rgb.max() * 255).astype('uint8')  # scale image range from [0., 1.] to [0., 255.] and cast to uint8
        return rgb

    def rgb_s(self, coordinates=None):
        """
        A method that returns RGB stretched image of current satellite snapshot.

        :param coordinates: optional parameter that allows crop to specified coordinates. Coordinates must be in the same CRS.

        :return: numpy array that represents RGB stretched image as unsigned int8 (1 byte) array with HEIGHT x WIDTH x 3.
        """

        # method that is used is histogram equalization
        return equalize(self.rgb(coordinates))

    def ndvi(self, coordinates=None):
        """
        A method that returns Normalized Difference Vegetation Index.

        :param coordinates: optional parameter that allows crop to specified coordinates. Coordinates must be in the same CRS.

        :return: numpy array that represents one channel image as float64 (8 byte pixel) array with HEIGHT x WIDTH x 1. Range of each pixel is between [-1., 1.].
        """

        # load red and near-red channels
        r = self.load(3)
        nir = self.load(4)

        if coordinates is None:
            # TODO: Return full size image
            pass

        # crop image
        r = msk.mask(r, coordinates, crop=True)[0]
        nir = msk.mask(nir, coordinates, crop=True)[0]

        # cast to float64 (8 byte pixel)
        r = r[0].astype('float64')
        nir = nir[0].astype('float64')

        # calculate ndvi
        # may give zero division warning, so nan_to_num is used
        ndvi = np.nan_to_num((nir - r) / (nir + r))

        return ndvi

    def ndvi_classes(self, coordinates=None):
        """
        A method that returns classified image based on Normalized Difference Vegetation Index.
        Classes:
        water - (0, 0, 255) - blue
        clouds - (255, 255, 255) - white
        shadow/buildings/not living - (128, 128, 128) - gray
        bare soil/sand - (128, 128, 0) - dark yellow
        low vegetation - (0, 255, 0) - green
        huge vegetation - (0, 128, 0) - dark green

        :param coordinates: optional parameter that allows crop to specified coordinates. Coordinates must be in the same CRS.

        :return: numpy array that represents one channel image as float64 (8 byte pixel) array with HEIGHT x WIDTH x 3.
        """

        # load aerosol band
        b1 = self.load(0)

        if coordinates is None:
            # TODO: Return full size image
            pass

        # crop and cast to float64
        b1 = msk.mask(b1, coordinates, crop=True)[0]
        b1 = b1[0].astype('float64')

        # isn't the most efficient way to get cloud mask
        # if aerosol layer value (from 0 to 1) is higher then 40% then it's cloud
        # TODO: Insert link to prove the idea
        b1 = np.where(b1 / (2 ** 16) > 0.4, 1, 0).astype('uint8')

        ndvi = self.ndvi(coordinates)  # obtain ndvi first

        # assign classes
        ndvi_classification = np.zeros(ndvi.shape + (3,)).astype('uint8')  # create placeholder
        ndvi_classification[(ndvi < 0.02) & (ndvi != 0)] = [0, 0, 255]  # water body
        ndvi_classification[b1 == 1] = [255, 255, 255]  # clouds
        ndvi_classification[(ndvi >= 0.02) & (ndvi < 0.12)] = [128, 128, 128]  # shadow/buildings
        ndvi_classification[(ndvi > 0.15) & (ndvi < 0.2)] = [128, 128, 0]  # bare soil/sand
        ndvi_classification[(ndvi >= 0.2) & (ndvi < 0.4)] = [0, 255, 0]  # low vegetation
        ndvi_classification[(ndvi >= 0.4)] = [0, 128, 0]  # huge vegetation

        # cast to uint8 and return
        return ndvi_classification.astype('uint8')


class MainGUIController:
    """
    A class that manipulates with data. Represents Controller in MVC pattern.
    This class is singleton class and should be used only with main method call.
    """

    def __init__(self):
        # define coordinates to work with
        self.coordinates = [
            # '48.743230, 34.471636',
            # '48.627347, 35.372376',
            # '48.199646, 35.314636',
            # '48.379442, 34.630997'
            '48.238534, 37.697641',
            '47.970177, 38.328160',
            '47.875766, 37.801573',
            '47.981000, 37.561375',
            '48.044346, 37.623734'
        ]
        self.first_path = self.second_path = None  # directory paths
        self.loader1 = self.loader2 = None  # loaders

    def get_coordinates_as_array(self) -> np.ndarray:
        """
        Because coordinates stored as list of strings in special format this method returns coordinates as numpy array.

        :return: numpy array of coordinates (latitude, longitude).
        """

        coordinates = []
        for coord in self.coordinates:
            split = coord.split(', ')
            coordinates.append([float(split[0]), float(split[1])])
        return np.array(coordinates)

    @staticmethod
    def main():
        """
        A method that starts GUI Application.
        """

        app = Application(MainGUIController())
        app.start()

    @staticmethod
    def select_path(path: Path):
        """
        A method that gets and sets important values from Landsat8 directory

        :param path: path to landsat8 directory

        :return: path, row, datetime
        """

        if Path.exists(path) and Path.is_dir(path):
            f = list(filter(None, [f if '_SR_stac.json' in str(f) else None for f in Path(path).glob('*')]))
            if f is None or len(f) == 0:
                return
            f = f[0]
            with open(f, 'r') as file:
                json_decoder = json.load(file)
                properties = json_decoder['properties']
                path = properties['landsat:wrs_path']
                row = properties['landsat:wrs_row']
                dt = properties['datetime']
                dt = dateutil.parser.isoparse(dt)
                return path, row, dt
        raise FileNotFoundError()

    def set_first_path(self, path: Path):
        """
        Sets path to first landsat8 directory.

        :param path: to landsat8 directory
        """

        self.loader1 = Loader(path)  # create loader
        self.first_path = path  # save path

    def set_second_path(self, path):
        """
        Sets path to second landsat8 directory.

        :param path: to landsat8 directory
        """

        self.loader2 = Loader(path)  # create loader
        self.second_path = path  # save path

    def clear_paths(self):
        """
        Clears paths to landsat8 directories and destroys loaders.
        """

        self.loader1 = self.loader2 = None
        self.first_path = self.second_path = None

    def exit(self):
        pass

    def visualize_coordinates(self):
        if len(self.coordinates) == 0:
            Application.error('No coordinates specified')
            return

        p = Polygon(self.get_coordinates_as_array())
        x, y = p.exterior.xy
        plt.fill(y, x)
        plt.show()

    def get_loader(self, loader_index: int) -> Loader:
        if loader_index == 1:
            if self.loader1 is None:
                Application.error('Select path to first directory')
                raise ValueError('No first loader found')
            return self.loader1
        else:
            if self.loader2 is None:
                Application.error('Select path to second directory')
                raise ValueError('No second loader found')
            return self.loader2

    def get_polygon(self, crs) -> Polygon:
        out_proj = Proj(init=str(crs).lower())
        in_proj = Proj(init='epsg:4326')

        new_coords = transform(in_proj, out_proj, self.get_coordinates_as_array())
        if len(new_coords) == 0:
            Application.error('No coordinates specified')
            raise ValueError('No coordinates specified')
        return Polygon(new_coords)

    @staticmethod
    def plot_img_in_another_process(img, title=None):
        multiprocessing.Process(target=plot_img, args=(img, title)).start()

    def get_img(self, method, ps, title):
        try:
            rgb = method(ps)
            self.plot_img_in_another_process(rgb, title)
        except:
            # TODO: Inform user about error
            pass

    def rgb1(self):
        loader = self.get_loader(1)
        p = self.get_polygon(loader.crs)
        self.get_img(loader.rgb, [p], 'RGB 1')

    def rgb2(self):
        loader = self.get_loader(2)
        p = self.get_polygon(loader.crs)
        self.get_img(loader.rgb, [p], 'RGB 2')

    def rgb1_s(self):
        loader = self.get_loader(1)
        p = self.get_polygon(loader.crs)
        self.get_img(loader.rgb_s, [p], 'RGB 1 Stretched')

    def rgb2_s(self):
        loader = self.get_loader(2)
        p = self.get_polygon(loader.crs)
        self.get_img(loader.rgb_s, [p], 'RGB 2 Stretched')

    def ndvi1(self):
        loader = self.get_loader(1)
        p = self.get_polygon(loader.crs)
        self.get_img(loader.ndvi, [p], 'NDVI 1')

    def ndvi2(self):
        loader = self.get_loader(2)
        p = self.get_polygon(loader.crs)
        self.get_img(loader.ndvi, [p], 'NDVI 2')


def plot_img(img, title=None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis(False)
    plt.show()


def equalize(img):
    """
    Histogram Equalization for RGB image.
    https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

    :param img: 3 dimensional tensor HEIGHT x WIDTH x 3 that represents RGB image

    :return: equalized by each channel image
    """

    # apply Histogram Equalization to each channel
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    return img


def transform(in_proj, out_proj, coordinates):
    """
    A method that transforms coordinates from one projection system to another.

    :param in_proj: input projection
    :param out_proj: output projection
    :param coordinates: coordinates to transform

    :return: coordinates in new projection system
    """

    new_coords = []
    for coord in coordinates:
        new_coords.append(trns(in_proj, out_proj, coord[1], coord[0]))
    return new_coords


if __name__ == '__main__':
    MainGUIController.main()

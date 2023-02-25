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
import threading
import numpy as np
import cv2

# coord modules
from pyproj.crs import CRS
from pyproj.transformer import Transformer
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from matplotlib import patches as ptchs
from rasterio import mask as msk
import rasterio

# Machine learning modules
import tensorflow as tf

from scripts.imageprocessing import ImageProcessing

import time

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

    visualizations = ['rgb', 'rgb_s', 'ndvi', 'gndvi', 'ndvi_classes']
    indices = ['ndvi', 'gndvi']

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

    def gndvi(self, coordinates=None):
        """
        A method that returns Green Normalized Difference Vegetation Index.

        :param coordinates: optional parameter that allows crop to specified coordinates. Coordinates must be in the same CRS.

        :return: numpy array that represents one channel image as float64 (8 byte pixel) array with HEIGHT x WIDTH x 1. Range of each pixel is between [-1., 1.].
        """

        # load red and near-red channels
        g = self.load(2)
        nir = self.load(4)

        if coordinates is None:
            # TODO: Return full size image
            pass

        # crop image
        g = msk.mask(g, coordinates, crop=True)[0]
        nir = msk.mask(nir, coordinates, crop=True)[0]

        # cast to float64 (8 byte pixel)
        g = g[0].astype('float64')
        nir = nir[0].astype('float64')

        # calculate gndvi
        # may give zero division warning, so nan_to_num is used
        gndvi = np.nan_to_num((nir - g) / (nir + g))

        return gndvi

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

    def get_method(self, name):
        methods = [self.rgb, self.rgb_s, self.ndvi, self.gndvi, self.ndvi_classes]
        for method in methods:
            if name == method.__name__:
                return method
        return None


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

        self.percentage = 0  # for long-performing task

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
        out_proj = CRS.from_wkt(crs.wkt)
        in_proj = 'epsg:4326'

        new_coords = transform(in_proj, out_proj, self.get_coordinates_as_array())
        if len(new_coords) == 0:
            Application.error('No coordinates specified')
            raise ValueError('No coordinates specified')
        return Polygon(new_coords)

    @staticmethod
    def plot_img_in_another_process(img, title=None, legend_handles=None, cmap=None):
        multiprocessing.Process(target=plot_img, args=(img, title, legend_handles, cmap)).start()

    def get_img(self, method, ps, title, legend_handles=None, cmap=None):
        try:
            img = method(ps)
            self.plot_img_in_another_process(img, title, legend_handles, cmap)
        except:
            Application.error('Coordinates don\'t specified correctly')
            # TODO: Inform user about error
            pass

    '''Next methods can be replaced with reflection'''

    def __call__(self, loader: int, visualization: str):
        """
        Returns callable object for visualization.

        :param loader: specifies image loader (1 or 2)
        :param visualization: specifies visualization type ['rgb', 'rgbs', 'ndvi', 'gndvi', 'ndvi_classes']
        :return: callable object of specified visualization
        """

        return lambda: self.call(loader, visualization)

    def call(self, loader_number, visualization):
        """
        Plots specified visualization in new matplotlib window.

        :param loader_number: specifies image loader (1 or 2)
        :param visualization: specifies visualization type
        """

        AVAILABLE_VISUALIZATIONS = Loader.visualizations
        INDICES = Loader.indices

        visualization = visualization.strip().lower()
        if visualization not in AVAILABLE_VISUALIZATIONS:
            raise ValueError('Visualization must be one of ' + str(AVAILABLE_VISUALIZATIONS))

        if loader_number >= 3 or loader_number <= 0:
            raise ValueError('Loader number must be 1 or 2')
        loader = self.get_loader(loader_number)

        p = self.get_polygon(loader.crs)

        colors = ['k', 'b', 'w', 'tab:gray', 'y', 'tab:green', 'g']
        labels = ['No data', 'Water', 'Clouds', 'Shadow/not living objects', 'Soil/sand', 'Low vegetation',
                  'Huge vegetation']  # define labels
        legend = [ptchs.Patch(color=color, label=label) for label, color in zip(labels, colors)]  # create legend

        method = loader.get_method(visualization)
        self.get_img(method, [p], visualization.upper() + ' ' + str(loader_number),
                     legend_handles=legend if 'classes' in visualization else None,
                     cmap='RdYlGn' if visualization in INDICES else None)

    """
    Next methods are deprecated and should be removed
    """

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

    def gndvi1(self):
        loader = self.get_loader(1)
        p = self.get_polygon(loader.crs)
        self.get_img(loader.gndvi, [p], 'GNDVI 1')

    def gndvi2(self):
        loader = self.get_loader(2)
        p = self.get_polygon(loader.crs)
        self.get_img(loader.gndvi, [p], 'GNDVI 2')

    def classification1(self):
        loader = self.get_loader(1)
        p = self.get_polygon(loader.crs)
        colors = ['k', 'b', 'w', 'tab:gray', 'y', 'tab:green', 'g']
        labels = ['No data', 'Water', 'Clouds', 'Shadow/not living objects', 'Soil/sand', 'Low vegetation', 'Huge vegetation']  # define labels
        legend = [ptchs.Patch(color=color, label=label) for label, color in zip(labels, colors)]  # create legend
        self.get_img(loader.ndvi_classes, [p], 'Classes 1', legend)

    def classification2(self):
        loader = self.get_loader(2)
        p = self.get_polygon(loader.crs)
        colors = ['k', 'b', 'w', 'tab:gray', 'y', 'tab:green', 'g']
        labels = ['No data', 'Water', 'Clouds', 'Shadow/not living objects', 'Soil/sand', 'Low vegetation', 'Huge vegetation']  # define labels
        legend = [ptchs.Patch(color=color, label=label) for label, color in zip(labels, colors)]  # create legend
        self.get_img(loader.ndvi_classes, [p], 'Classes 1', legend)

    """
    deforestation method is called with specific deforestation model passed.
    This method creates new daemon Thread with deforestation_ as target.
    deforestation_ performs:
        NDIV1 & NDVI2 load
        concatenates NDVIs in x (input to model)
        loads specified model
        runs loop (so computer doesn't run out of memory) with percentage modification (self.percentage)
        after prediction phase, concatenates all obtained images into one
        takes only necessary part of the image
        displays it
    """

    def deforestation(self, model_file):
        self.percentage = 0
        foo_thread = threading.Thread(target=lambda: self.deforestation_(model_file))  # create Thread
        foo_thread.daemon = True
        Application.start_progressbar()  # mainloop starts progressbar and always active to user commands
        foo_thread.start()  # starts TF process

    def deforestation_(self, model_file):
        # TODO: Add caching system
        loader1, loader2 = self.get_loader(1), self.get_loader(2)  # get both loaders
        p = self.get_polygon(loader1.crs)  # get coordinates restriction

        # get both nvdi and concat inputs
        ndvi1 = loader1.ndvi([p])
        ndvi2 = loader2.ndvi([p])
        ndvi1.shape = ndvi1.shape[:2] + (1,)
        ndvi2.shape = ndvi2.shape[:2] + (1,)
        x = np.concatenate((ndvi1, ndvi2), -1)

        # load model
        model = tf.keras.models.load_model('assets/models/de_forestation/' + str(model_file))
        original_size = x.shape[:2]
        print(original_size)
        FIRST_TIME_POINT = time.time()
        x = ImageProcessing.add_blank_to_npt(x)  # to correctly process input image must be in shapes of power of two
        resized_size = x.shape[:2]

        # define legend options
        colors = ['k', 'r', 'g', 'y', 'b']  # define matplotlib colors
        labels = ['No data', 'Deforestation', 'Forestation', 'No change', 'Water body']  # define labels
        legend = [ptchs.Patch(color=color, label=label) for label, color in zip(labels, colors)]  # create legend

        if x.shape[0] <= 128 and x.shape[1] <= 128:  # if images is small single run can be performed
            prediction = model.predict(x.reshape((1,) + x.shape))
            self.plot_img_in_another_process(prediction[0], 'De-forestation', legend)
            self.percentage = 1  # although no progressbar is needed, it must be destroyed
        else:
            x = ImageProcessing.split_img(x, 64)  # split images to mini batches
            split_shape = x.shape[:-1]
            x.shape = (-1, 64, 64, 2)  # combine mini batches in array
            y = np.zeros(x.shape[:-1] + (5,))  # create output placeholder
            N = x.shape[0]  # number of batches
            for i in range(0, N, 4):  # actual batch size is 4
                y[i:i+4] = model.predict(x[i:i+4])  # process and save to output placeholder
                self.percentage = i / N  # calculate percentage (for progressbar)
            self.percentage = 1  # after all is done progressbar must be destroyed

            # concat them by pairs
            y.shape = split_shape + (5,)
            y = np.concatenate(np.concatenate(y, 1), 1)  # have no idea, why it works
            assert y.shape[:2] == resized_size  # just in case something went wrong

            y = y.argmax(-1)[:original_size[0], :original_size[1]].astype('uint8')  # obtain necessary part only
            img = np.zeros((y.shape[0], y.shape[1], 3), dtype='uint8')  # create placeholder of image
            colors_in_rgb = [[255, 0, 0], [0, 255, 0], [255, 255, 0], [0, 0, 255]]  # define actual colors
            colors_in_rgb = np.array(colors_in_rgb).astype('uint8')
            for i in range(1, 5):
                img[y == i] = colors_in_rgb[i - 1]
            self.plot_img_in_another_process(img, 'De-forestation', legend)
        print('Elapsed time', time.time() - FIRST_TIME_POINT)


def plot_img(img, title=None, legend_handles=None, cmap=None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    if legend_handles is not None:
        plt.legend(handles=legend_handles)
    if cmap is not None:
        plt.set_cmap(cmap)
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
    transformer = Transformer.from_crs(in_proj, out_proj)
    for coord in coordinates:
        #new_coords.append(transformer.transform(coord[1], coord[0]))
        new_coords.append(transformer.transform(coord[0], coord[1]))
    return new_coords


if __name__ == '__main__':
    MainGUIController.main()

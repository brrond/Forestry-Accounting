from gui.tk import *

from pathlib import Path
import dateutil.parser
import json
import os
import re

import numpy as np

from pyproj import Proj, transform
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from rasterio import mask as msk
import multiprocessing
import rasterio
import cv2


class Loader:
    def __init__(self, dir_):
        self.dir = dir_
        self.bands = list(filter(lambda f: re.compile('B[0-9]*.TIF').search(f), os.listdir(self.dir)))

        a = rasterio.open(self.dir / self.bands[0])
        self.crs = a.crs
        del a

    def load(self, band):
        return rasterio.open(self.dir / self.bands[band])

    def rgb(self, coordinates=None):
        b = self.load(1)
        g = self.load(2)
        r = self.load(3)

        if coordinates is None:
            # TODO: Return full size image
            pass

        r = msk.mask(r, coordinates, crop=True)[0]
        g = msk.mask(g, coordinates, crop=True)[0]
        b = msk.mask(b, coordinates, crop=True)[0]

        r = r[0].astype('float64')
        g = g[0].astype('float64')
        b = b[0].astype('float64')

        rgb = cv2.merge((r, g, b))
        rgb = (rgb / rgb.max() * 255).astype('uint8')
        return rgb

    def rgb_s(self, coordinates=None):
        return equalize(self.rgb(coordinates))

    def ndvi_classes(self, coordinates=None):
        b1 = self.load(0)
        r = self.load(3)
        nir = self.load(4)

        if coordinates is None:
            # TODO: Return full size image
            pass

        r = msk.mask(r, coordinates, crop=True)[0]
        b1 = msk.mask(b1, coordinates, crop=True)[0]
        nir = msk.mask(nir, coordinates, crop=True)[0]

        r = r[0].astype('float64')
        b1 = b1[0].astype('float64')
        nir = nir[0].astype('float64')

        b1 = np.where(b1 / (2 ** 16) > 0.4, 1, 0).astype('uint8')
        ndvi = np.nan_to_num((nir - r) / (nir + r))

        ndvi_classification = np.zeros(ndvi.shape + (3,)).astype('uint8')
        ndvi_classification[(ndvi < 0.02) & (ndvi != 0)] = [0, 0, 255]  # water body
        ndvi_classification[b1 == 1] = [255, 255, 255]  # clouds
        ndvi_classification[(ndvi >= 0.02) & (ndvi < 0.12)] = [128, 128, 128]  # shadow/buildings
        ndvi_classification[(ndvi > 0.15) & (ndvi < 0.2)] = [128, 128, 0]  # bare soil/sand
        ndvi_classification[(ndvi >= 0.2) & (ndvi < 0.4)] = [0, 255, 0]  # low
        ndvi_classification[(ndvi >= 0.4)] = [0, 128, 0]  # huge
        return ndvi_classification.astype('uint8')


class MainController:
    def __init__(self):
        self.coordinates = [
            #'48.743230, 34.471636',
            #'48.627347, 35.372376',
            #'48.199646, 35.314636',
            #'48.379442, 34.630997'
            '48.238534, 37.697641',
            '47.970177, 38.328160',
            '47.875766, 37.801573',
            '47.981000, 37.561375',
            '48.044346, 37.623734'
        ]
        self.first_path = self.second_path = None

    def get_coordinates_as_array(self):
        coordinates = []
        for coord in self.coordinates:
            split = coord.split(', ')
            coordinates.append([float(split[0]), float(split[1])])
        return np.array(coordinates)

    @staticmethod
    def main():
        app = Application(MainController())
        app.mainloop()

    @staticmethod
    def transform(in_proj, out_proj, coordinates):
        new_coords = []
        for coord in coordinates:
            new_coords.append(transform(in_proj, out_proj, coord[1], coord[0]))
        return new_coords

    def select_path(self, path):
        if Path.exists(path) and Path.is_dir(path):
            f = list(filter(None, [f if '_SR_stac.json' in str(f) else None for f in Path(path).glob('*')]))
            if f is None or len(f) == 0:
                # TODO: Inform user about error
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
        # TODO: Inform user about error

    def exit(self):
        pass

    def visualize_coordinates(self):
        if len(self.coordinates) == 0:
            # TODO: Inform user about error
            return

        p = Polygon(self.get_coordinates_as_array())
        x, y = p.exterior.xy
        plt.fill(y, x)
        plt.show()

    def rgb1(self):
        if self.first_path is None:
            # TODO: Inform user about error
            return

        loader = Loader(self.first_path)
        out_proj = Proj(init=str(loader.crs).lower())
        in_proj = Proj(init='epsg:4326')

        new_coords = self.transform(in_proj, out_proj, self.get_coordinates_as_array())
        p = Polygon(new_coords)

        try:
            rgb = loader.rgb([p])
            multiprocessing.Process(target=plot_img, args=(rgb, 'RGB 1')).start()
        except:
            # TODO: Inform user about error
            pass

    def rgb2(self):
        if self.second_path is None:
            # TODO: Inform user about error
            return

        loader = Loader(self.second_path)
        out_proj = Proj(init=str(loader.crs).lower())
        in_proj = Proj(init='epsg:4326')

        new_coords = self.transform(in_proj, out_proj, self.get_coordinates_as_array())
        p = Polygon(new_coords)

        try:
            rgb = loader.rgb([p])
            multiprocessing.Process(target=plot_img, args=(rgb, 'RGB 2')).start()
        except:
            # TODO: Inform user about error
            pass

    def rgb1_s(self):
        if self.first_path is None:
            # TODO: Inform user about error
            return

        loader = Loader(self.first_path)
        out_proj = Proj(init=str(loader.crs).lower())
        in_proj = Proj(init='epsg:4326')

        new_coords = self.transform(in_proj, out_proj, self.get_coordinates_as_array())
        p = Polygon(new_coords)

        try:
            rgb = loader.rgb_s([p])
            multiprocessing.Process(target=plot_img, args=(rgb, 'RGB 1 Stretched')).start()
        except:
            # TODO: Inform user about error
            pass

    def rgb2_s(self):
        if self.second_path is None:
            # TODO: Inform user about error
            return

        loader = Loader(self.second_path)
        out_proj = Proj(init=str(loader.crs).lower())
        in_proj = Proj(init='epsg:4326')

        new_coords = self.transform(in_proj, out_proj, self.get_coordinates_as_array())
        p = Polygon(new_coords)

        try:
            rgb = loader.rgb_s([p])
            multiprocessing.Process(target=plot_img, args=(rgb, 'RGB 1 Stretched')).start()
        except:
            # TODO: Inform user about error
            pass

    def ndvi1(self):
        if self.first_path is None:
            # TODO: Inform user about error
            return

        loader = Loader(self.first_path)
        out_proj = Proj(init=str(loader.crs).lower())
        in_proj = Proj(init='epsg:4326')

        new_coords = self.transform(in_proj, out_proj, self.get_coordinates_as_array())
        p = Polygon(new_coords)

        try:
            ndvi = loader.ndvi_classes([p])
            multiprocessing.Process(target=plot_img, args=(ndvi, 'NDVI 1')).start()
        except:
            # TODO: Inform user about error
            pass

    def ndvi2(self):
        if self.second_path is None:
            # TODO: Inform user about error
            return

        loader = Loader(self.second_path)
        out_proj = Proj(init=str(loader.crs).lower())
        in_proj = Proj(init='epsg:4326')

        new_coords = self.transform(in_proj, out_proj, self.get_coordinates_as_array())
        p = Polygon(new_coords)

        try:
            ndvi = loader.ndvi_classes([p])
            multiprocessing.Process(target=plot_img, args=(ndvi, 'NDVI 2')).start()
        except:
            # TODO: Inform user about error
            pass


def plot_img(img, title=None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis(False)
    plt.show()

def equalize(img):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img


if __name__ == '__main__':
    MainController.main()

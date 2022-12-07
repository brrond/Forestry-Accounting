from gui.tk import *

from pathlib import Path
import dateutil.parser
import json

import numpy as np

from pyproj import Proj, transform
from shapely.geometry import Polygon
from matplotlib import pyplot as plt


class MainController:
    def __init__(self):
        self.coordinates = [
            '48.743230, 34.471636',
            '48.627347, 35.372376',
            '48.199646, 35.314636',
            '48.379442, 34.630997'
        ]
        self.first_path = self.second_path = None

    def get_coordinates_as_array(self):
        coordinates = []
        for coord in self.coordinates:
            coordinates.append((float(coord.split(', ')[0]), float(coord.split(', ')[1])))
        return np.array(coordinates)

    @staticmethod
    def main():
        app = Application(MainController())
        app.mainloop()

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


MainController.main()

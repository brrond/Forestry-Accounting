from gui.tk import *

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

    def select_first_path(self, path):
        # return path and row
        pass

    def select_second_path(self, path):
        # return path and row
        pass

    def clear_selected_paths(self):
        pass

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

from tensorflow import keras
import numpy as np
import tifffile
import os


# Define generator
class MyGenerator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, x_data="x_data/", y_data="y_data/", split=1.0):
        """
        :param x_data: path to x_data folder.
        :param y_data: path to y_data folder.
        :param split:   specify split of train and validation set.
                        Values bigger then 0 means train split.
                        Values less then 0 means validation split.
                        MyGenerator(split=0.8), MyGenerator(split=-0.2) -> returns two generators with 80% and 20% split
        """

        self.batch_size = (64,)
        self.img_size = (64, 64)
        self.x_data = x_data
        self.y_data = y_data
        self.images = np.array(os.listdir(self.x_data))
        np.random.shuffle(self.images)

        # split > 0 => train split (0.8)
        # split < 0 => validation split (-0.2 means 20% of the data only)
        np.random.seed(42)
        if split > 0.0:
            self.images = self.images[: int(split * len(self.images))]
        else:
            self.images = self.images[int(split * len(self.images)) :]

    def __len__(self):
        return len(self.images) // self.batch_size[0]

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        # get paths
        images = self.images[idx * self.batch_size[0] : (idx + 1) * self.batch_size[0]]

        # create arrays
        x = np.zeros(self.batch_size + self.img_size + (2,), dtype="float32")
        y = np.zeros(self.batch_size + self.img_size + (5,), dtype="float32")

        # read data
        for i in range(self.batch_size[0]):
            x[i] = tifffile.imread(self.x_data + images[i])
            y_r = tifffile.imread(self.y_data + images[i])
            y_r.shape = self.img_size
            y[i, y_r == 0] = [1, 0, 0, 0, 0]
            y[i, y_r == 1] = [0, 1, 0, 0, 0]
            y[i, y_r == 2] = [0, 0, 1, 0, 0]
            y[i, y_r == 3] = [0, 0, 0, 1, 0]
            y[i, y_r == 4] = [0, 0, 0, 0, 1]

        return x, y

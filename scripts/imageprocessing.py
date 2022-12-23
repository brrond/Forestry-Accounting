import numpy as np


class ImageProcessing:
    """
    A class that helps to work with large images.
    Each method of the class is static so can be used without class initialization.
    """

    @staticmethod
    def add_blank(img_arr, width, height):
        """
        Adds blank spaces (zeros) so image can match sizes. All zeros are added along right and bottom sides.

        :param img_arr: input image as numpy array.
        :param width: preferred width of the image.
        :param height: preferred height of the image.

        :return: numpy array that represent same image with additional blank pixels.
        """
        
        img_h, img_w = img_arr.shape[:2]
        h_shape = list(img_arr.shape)
        h_shape[0] = height - img_h
        h_arr = np.zeros(h_shape)
        img_arr = np.append(img_arr, h_arr, 0)
       
        img_h, img_w = img_arr.shape[:2]
        v_shape = list(img_arr.shape)
        v_shape[1] = width - img_w
        v_arr = np.zeros(v_shape)
        img_arr = np.append(img_arr, v_arr, 1)

        return img_arr

    @staticmethod
    def split_img(img_arr, size):
        """
        Allows image cropping.
        Split large image according to grid of size.
        Width and height of each tile is size x size.

        :param img_arr: input image.
        :param size: width AND height of the tile.

        :return: numpy array with shape (NUMBER_OF_TILES, NUMBER_OF_TILES, SIZE, SIZE, CHANNELS)
        """

        h = int(img_arr.shape[0] / size)
        w = int(img_arr.shape[1] / size)
        res = np.zeros((h, w, size, size, img_arr.shape[-1]), dtype=img_arr.dtype)
        
        for i in range(h):
            for j in range(w):
                res[i][j] = img_arr[i * size: i * size + size,
                                    j * size: j * size + size]
        return res

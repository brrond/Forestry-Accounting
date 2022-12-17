import numpy as np


class ImageProcessing:
    @staticmethod
    def add_blank(img_arr, width, height):
        '''
        Parameters
        ----------
        img_arr : numpy array
            Img to add blank spaces
        width : int
            width, that should be
        height : int
            height, that should be

        Returns
        -------
        New img of sizes width x height
        '''
        
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
        h = int(img_arr.shape[0] / size)
        w = int(img_arr.shape[1] / size)
        res = np.zeros((h, w, size, size, img_arr.shape[-1]), dtype=img_arr.dtype)
        
        for i in range(h):
            for j in range(w):
                res[i][j] = img_arr[i * size : i * size + size, 
                                    j * size : j * size + size]
        return res

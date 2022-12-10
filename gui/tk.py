import tkinter as tk
from tkinter.messagebox import showerror, showinfo
from tkinter.filedialog import askdirectory

from pathlib import Path

ASSETS_DIR = Path('assets')
IMAGES_DIR = ASSETS_DIR / 'images'
TMP_DIR = ASSETS_DIR / 'tmp'


class ListVariable(tk.Variable):
    """
    A class used to extend possibilities of default tk.Variable to work as list with on_update callback
    """

    def __init__(self, master=None, value=None, name=None, on_update=None):
        """
        :param master: used to attach variable to widget
        :param value: initial value of the variable
        :param name: see tk.Variable
        :param on_update: callback that will be called each time variable updates

        :raises TypeError: if value isn't list type
        """

        # check if value is list
        if type(value) == list.__class__:
            raise TypeError()

        # init super class
        super().__init__(master, value, name)
        self.values = value if value is not None else []  # set default value
        self.on_update = on_update  # set callback

    def set(self, value: list):
        """
        :param value: new list-value for variable

        :raises TypeError: if value isn't list type
        """

        # check if value is list
        if type(value) == list.__class__:
            raise TypeError()

        self.values = value  # set value
        super().set(value)  # and set with super call
        if self.on_update is not None:
            self.on_update(self.values)  # callback

    def get(self) -> list:
        """
        :return: list of values
        """

        return self.values

    def append(self, obj):
        """
        Works the same way as list.append

        :param obj: object to append
        """

        self.values.append(obj)
        self.set(self.values)  # set updated list

    def pop(self, index):
        """
        Works the same way as list.pop

        :param index: element to remove
        """

        self.values.pop(index)
        self.set(self.values)  # set updated list

    def clear(self):
        """
        Clear list
        """

        self.values = []
        self.set(self.values)


class Application(tk.Tk):
    """
    Main Application class for GUI using tkinter as backend
    """

    def __init__(self, main_controller):
        super().__init__()

        self.mc = self.main_controller = main_controller

        # init default window settings
        self.wm_title('Forestry')
        self.wm_geometry('400x600+100+100')
        self.wm_iconphoto(True, tk.PhotoImage(file=IMAGES_DIR / 'ico.png'))
        self.wm_resizable(False, False)

        # init grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # init ui
        # init input fields
        input_frame = tk.Frame(self)
        tk.Label(input_frame, text='Landsat 8 first: ').grid(column=0, row=0)
        tk.Label(input_frame, text='Landsat 8 second: ').grid(column=0, row=1)

        self.entry_1_var = tk.StringVar(input_frame, value='Path: Row:')
        self.entry_2_var = tk.StringVar(input_frame, value='Path: Row:')
        tk.Entry(input_frame, state='readonly', textvariable=self.entry_1_var, width=30).grid(column=1, row=0)
        tk.Entry(input_frame, state='readonly', textvariable=self.entry_2_var, width=30).grid(column=1, row=1)

        tk.Button(input_frame, text='Select', command=self.select_first_path).grid(column=2, row=0)
        tk.Button(input_frame, text='Select', command=self.select_second_path).grid(column=2, row=1)

        tk.Button(input_frame, text='Clear', command=self.clear_path).grid(column=0, row=2, columnspan=3, sticky='EW',
                                                                           pady=5)
        tk.Button(input_frame, text='Exit', command=self.destroy).grid(column=0, row=3, columnspan=3, sticky='EW',
                                                                       pady=5)
        input_frame.grid(column=0, row=0)

        # init control panel
        def update_mc_coordinates(vals):
            self.mc.coordinates = vals

        control_frame = tk.Frame(self)
        self.coords_var = ListVariable(self, self.mc.coordinates, on_update=update_mc_coordinates)
        tk.Listbox(control_frame, height=4, listvariable=self.coords_var).grid(column=0, row=0, rowspan=3, sticky='NS')
        tk.Button(control_frame, text='Set coordinates', command=lambda: CoordinatesDialog(self, self.coords_var)).grid(
            column=1, row=0, pady=5, sticky='EW')
        tk.Button(control_frame, text='Clear coordinates', command=self.coords_var.clear).grid(column=1, row=1, pady=5,
                                                                                               sticky='EW')
        tk.Button(control_frame, text='Visualize', command=self.mc.visualize_coordinates).grid(column=1, row=2, pady=5,
                                                                                               sticky='EW')
        control_frame.grid(column=0, row=1)

        # init output frame
        output_frame = tk.Frame(self)
        tk.Label(output_frame, text='RGB 1').grid(column=0, row=0)
        tk.Label(output_frame, text='NDVI 1').grid(column=1, row=0)
        tk.Label(output_frame, text='NDVI 2').grid(column=2, row=0)
        tk.Label(output_frame, text='RGB 2').grid(column=3, row=0)
        self.pi_ndvi = tk.PhotoImage(file=IMAGES_DIR / 'ndvi.png')
        self.pi_rgb = tk.PhotoImage(file=IMAGES_DIR / 'rgb.png')
        tk.Button(output_frame, image=self.pi_rgb, command=self.mc.rgb1).grid(column=0, row=1)
        tk.Button(output_frame, image=self.pi_ndvi, command=self.mc.ndvi1).grid(column=1, row=1)
        tk.Button(output_frame, image=self.pi_ndvi, command=self.mc.ndvi2).grid(column=2, row=1)
        tk.Button(output_frame, image=self.pi_rgb, command=self.mc.rgb2).grid(column=3, row=1)

        tk.Label(output_frame, text='RGB stretched').grid(column=0, row=2)
        tk.Label(output_frame, text='GNDVI 1').grid(column=1, row=2)
        tk.Label(output_frame, text='GNDVI 2').grid(column=2, row=2)
        tk.Label(output_frame, text='RGB stretched').grid(column=3, row=2)
        self.pi_gndvi = tk.PhotoImage(file=IMAGES_DIR / 'gndvi.png')
        self.pi_rgb_s = tk.PhotoImage(file=IMAGES_DIR / 'rgb_stretching.png')
        tk.Button(output_frame, image=self.pi_rgb_s, command=self.mc.rgb1_s).grid(column=0, row=3)
        tk.Button(output_frame, image=self.pi_gndvi).grid(column=1, row=3)
        tk.Button(output_frame, image=self.pi_gndvi).grid(column=2, row=3)
        tk.Button(output_frame, image=self.pi_rgb_s, command=self.mc.rgb2_s).grid(column=3, row=3)

        tk.Label(output_frame, text='NDWI 1').grid(column=1, row=4)
        tk.Label(output_frame, text='NDWI 2').grid(column=2, row=4)
        self.pi_ndwi = tk.PhotoImage(file=IMAGES_DIR / 'ndwi.png')
        tk.Button(output_frame, image=self.pi_ndwi).grid(column=1, row=5)
        tk.Button(output_frame, image=self.pi_ndwi).grid(column=2, row=5)

        tk.Label(output_frame, text='De-forestation map').grid(column=1, row=6, columnspan=2)
        self.pi_def = tk.PhotoImage(file=IMAGES_DIR / 'def.png')
        tk.Button(output_frame, image=self.pi_def).grid(column=1, row=7, columnspan=2)
        output_frame.grid(column=0, row=2)

    def select_path(self):
        """
        Open directory dialog and get path. Calls select_path from main controller

        :return: directory, path (landsat 8), row (landsat 8), dt - datetime of acquiring snapshot
        """

        try:
            directory = askdirectory(title='Select LANDSAT 8 directory', mustexist=True)
            directory = Path(directory)
            result = self.mc.select_path(directory)
            if result is None:
                showerror('Error',
                          'Something went wrong during LANDSAT 8 directory loading. Make sure the directory exists and try again.')
                return
        except:
            showerror('Error',
                      'Something went wrong during LANDSAT 8 directory loading. Make sure the directory exists and try again.')
            return

        path, row, dt = result
        return directory, path, row, dt

    def select_first_path(self):
        dir_, path, row, dt = self.select_path()
        self.entry_1_var.set('Path: ' + path + ', Row: ' + row + ' ' + str(dt.date()))
        self.mc.first_path = dir_

    def select_second_path(self):
        dir_, path, row, dt = self.select_path()
        self.entry_2_var.set('Path: ' + path + ', Row: ' + row + ' ' + str(dt.date()))
        self.mc.second_path = dir_

    def clear_path(self):
        self.entry_1_var.set('Path: Row:')
        self.entry_2_var.set('Path: Row:')
        self.mc.first_path = self.mc.second_path = None

    @staticmethod
    def error(msg):
        """
        Shows error to user

        :param msg: message to display
        """
        showerror('Error', msg)


class CoordinatesDialog(tk.Toplevel):
    """
    A class that allows user to set polygon coordinates
    """

    def __init__(self, master, coords_var: ListVariable):
        """
        :param master: top level window
        :param coords_var: ListVariable to change coordinates
        """

        super().__init__()

        self.master = self.root = master
        self.coords_var = coords_var

        # init ui
        self.wm_title('Coordinates selection')
        self.wm_geometry('400x200+100+100')
        self.wm_iconphoto(True, tk.PhotoImage(file=IMAGES_DIR / 'ico.png'))
        self.wm_resizable(False, False)

        # init grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(self, height=4, listvariable=self.coords_var)
        self.listbox.grid(column=0, row=0, columnspan=3, sticky='EW')

        tk.Button(self, command=self.del_selected, text='Delete').grid(column=0, columnspan=3, row=1, ipady=5, sticky='EW')
        tk.Label(self, text='Coord (lat, long): ').grid(column=0, row=2)

        self.coord_var = tk.StringVar(self, value='')  # coordinate var for entry field
        tk.Entry(self, textvariable=self.coord_var).grid(column=1, pady=10, row=2, padx=5, sticky='EW')

        tk.Button(self, command=self.add, text='Add').grid(column=2, padx=5, pady=10, row=2, sticky='EW')
        tk.Button(self, command=self.destroy, text='Close').grid(column=0, columnspan=3, row=5, ipady=5, sticky='EW')

    def del_selected(self):
        """
        Deletes selected field
        """

        if self.listbox.curselection() != ():
            index = self.listbox.curselection()[0]
            self.coords_var.pop(index)
        else:
            showinfo('Info', 'Select some element first to delete.')

    def add(self):
        """
        Checks coordinates and adds new coordinate to list
        """

        s = self.coord_var.get().replace(' ', '')  # get coordinate
        if ',' in s:  # if format isn't correct
            coords = s.split(',')
            if len(coords) != 2:  # if user entered some bullshit
                showerror('Error',
                          'Wrong coordinates input. Must be entered in form: "lat, lon". Use "." as float separator')
            else:
                lat = coords[0]  # get latitude
                lon = coords[1]  # get longitude
                try:
                    lat = float(lat)  # try cast to float
                    lon = float(lon)  # -//-

                    point = str(lat) + ', ' + str(lon)
                    self.coords_var.append(point)

                    self.coord_var.set('')  # clear Entry field
                except:
                    showerror('Error', 'Wrong coordinates input. Must be entered two numbers in form: "lat, lon".')
        else:
            showerror('Error', 'Wrong coordinates input. Should be entered in form: "lat, lon".')
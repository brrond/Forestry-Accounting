import tkinter as tk
from tkinter.messagebox import showerror, showinfo
from tkinter.filedialog import askdirectory
from tkinter.ttk import Progressbar

from pathlib import Path
import random
import os

ASSETS_DIR = Path("assets")
IMAGES_DIR = ASSETS_DIR / "images"
TMP_DIR = ASSETS_DIR / "tmp"
MODELS_DIR = ASSETS_DIR / "models"
DE_FORESTATION_MODELS = os.listdir(MODELS_DIR / "de_forestation")


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

    self = None  # aka Singleton

    def __init__(self, main_controller):
        super().__init__()

        if Application.self is None:
            Application.self = self

        self.mc = self.main_controller = main_controller
        self.pb = None  # ProgressBar will be created when necessary

        # init default window settings
        self.wm_title("Forestry")
        self.wm_geometry("400x600+100+100")
        self.wm_iconphoto(True, tk.PhotoImage(file=IMAGES_DIR / "ico.png"))
        self.wm_resizable(False, False)

        # init grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # init ui
        # init input fields
        input_frame = tk.Frame(self)
        tk.Label(input_frame, text="Landsat 8 first: ").grid(column=0, row=0)
        tk.Label(input_frame, text="Landsat 8 second: ").grid(column=0, row=1)

        self.entry_1_var = tk.StringVar(input_frame, value="Path: Row:")
        self.entry_2_var = tk.StringVar(input_frame, value="Path: Row:")
        tk.Entry(
            input_frame, state="readonly", textvariable=self.entry_1_var, width=30
        ).grid(column=1, row=0)
        tk.Entry(
            input_frame, state="readonly", textvariable=self.entry_2_var, width=30
        ).grid(column=1, row=1)

        tk.Button(input_frame, text="Select", command=self.select_first_path).grid(
            column=2, row=0
        )
        tk.Button(input_frame, text="Select", command=self.select_second_path).grid(
            column=2, row=1
        )

        self.button_info1 = tk.Button(
            input_frame, text="Info", command=self.info_first_path
        )
        self.button_info1.grid(column=3, row=0)
        self.button_info1["state"] = "disabled"
        self.button_info2 = tk.Button(
            input_frame, text="Info", command=self.info_second_path
        )
        self.button_info2.grid(column=3, row=1)
        self.button_info2["state"] = "disabled"

        tk.Button(input_frame, text="Clear", command=self.clear_path).grid(
            column=0, row=2, columnspan=3, sticky="EW", pady=5
        )
        tk.Button(input_frame, text="Exit", command=self.destroy).grid(
            column=0, row=3, columnspan=3, sticky="EW", pady=5
        )
        input_frame.grid(column=0, row=0)

        # init control panel
        def update_mc_coordinates(vals):
            self.mc.coordinates = vals

        def select_coordinates():
            json1, json2 = self.mc.get_first_json(), self.mc.get_second_json()
            if len(json1) == 0 and len(json2) == 0:
                return
            elif len(json1) == 0:
                x1, y1, x2, y2 = json2["bbox"]
            elif len(json2) == 0:
                x1, y1, x2, y2 = json1["bbox"]
            else:
                x1_, y1_, x2_, y2_ = json1["bbox"]
                x1, y1, x2, y2 = json2["bbox"]

                if x1_ > x2 or x1 > x2_ or y1_ > y2 or y1 > y2_:
                    showerror("Error", "Different paths selected")
                    return
                x1 = max(x1, x1_)
                y1 = max(y1, y1_)
                x2 = min(x2, x2_)
                y2 = min(y2, y2_)

            self.coords_var.clear()
            self.coords_var.append(str(y1) + ", " + str(x1))
            self.coords_var.append(str(y1) + ", " + str(x2))
            self.coords_var.append(str(y2) + ", " + str(x2))
            self.coords_var.append(str(y2) + ", " + str(x1))

        def select_random_coordinates():
            json1, json2 = self.mc.get_first_json(), self.mc.get_second_json()
            if len(json1) == 0 and len(json2) == 0:
                return
            elif len(json1) == 0:
                x1, y1, x2, y2 = json2["bbox"]
            elif len(json2) == 0:
                x1, y1, x2, y2 = json1["bbox"]
            else:
                x1_, y1_, x2_, y2_ = json1["bbox"]
                x1, y1, x2, y2 = json2["bbox"]
                if x1_ > x2 or x1 > x2_ or y1_ > y2 or y1 > y2_:
                    showerror("Error", "Different paths selected")
                    return
                x1 = max(x1, x1_)
                y1 = max(y1, y1_)
                x2 = min(x2, x2_)
                y2 = min(y2, y2_)

            x1 = random.randint(int(x1 * 10000), int(x2 * 10000)) / 10000.0
            x2 = random.randint(int(x1 * 10000), int(x2 * 10000)) / 10000.0
            y1 = random.randint(int(y1 * 10000), int(y2 * 10000)) / 10000.0
            y2 = random.randint(int(y1 * 10000), int(y2 * 10000)) / 10000.0

            self.coords_var.clear()
            self.coords_var.append(str(y1) + ", " + str(x1))
            self.coords_var.append(str(y1) + ", " + str(x2))
            self.coords_var.append(str(y2) + ", " + str(x2))
            self.coords_var.append(str(y2) + ", " + str(x1))

        control_frame = tk.Frame(self)
        self.coords_var = ListVariable(
            self, self.mc.coordinates, on_update=update_mc_coordinates
        )
        tk.Listbox(control_frame, height=4, listvariable=self.coords_var).grid(
            column=0, row=0, rowspan=5, sticky="NS"
        )
        tk.Button(
            control_frame,
            text="Set coordinates",
            command=lambda: CoordinatesDialog(self, self.coords_var),
        ).grid(column=1, row=0, pady=5, sticky="EW")
        tk.Button(
            control_frame, text="Select from folder", command=select_coordinates
        ).grid(column=1, row=1, pady=5, sticky="EW")
        tk.Button(
            control_frame, text="Select random area", command=select_random_coordinates
        ).grid(column=1, row=2, pady=5, sticky="EW")
        tk.Button(
            control_frame, text="Clear coordinates", command=self.coords_var.clear
        ).grid(column=1, row=3, pady=5, sticky="EW")
        tk.Button(
            control_frame, text="Visualize", command=self.mc.visualize_coordinates
        ).grid(column=1, row=4, pady=5, sticky="EW")
        control_frame.grid(column=0, row=1)

        # init output frame
        output_frame = tk.Frame(self)
        tk.Label(output_frame, text="RGB 1").grid(column=0, row=0)
        tk.Label(output_frame, text="RGB 2").grid(column=1, row=0)
        self.pi_rgb = tk.PhotoImage(file=IMAGES_DIR / "rgb.png")
        tk.Button(output_frame, image=self.pi_rgb, command=self.mc(1, "rgb")).grid(
            column=0, row=1
        )
        tk.Button(output_frame, image=self.pi_rgb, command=self.mc(2, "rgb")).grid(
            column=1, row=1
        )

        tk.Label(output_frame, text="RGB stretched").grid(column=0, row=2)
        tk.Label(output_frame, text="RGB stretched").grid(column=1, row=2)
        self.pi_rgb_s = tk.PhotoImage(file=IMAGES_DIR / "rgb_stretching.png")
        tk.Button(output_frame, image=self.pi_rgb_s, command=self.mc(1, "rgb_s")).grid(
            column=0, row=3
        )
        tk.Button(output_frame, image=self.pi_rgb_s, command=self.mc(2, "rgb_s")).grid(
            column=1, row=3
        )

        tk.Label(output_frame, text="De-forestation map").grid(
            column=0, row=4, columnspan=2
        )
        self.pi_def = tk.PhotoImage(file=IMAGES_DIR / "def.png")
        tk.Button(
            output_frame,
            image=self.pi_def,
            command=lambda: self.mc.deforestation(self.deforestation_model_var.get()),
        ).grid(column=0, row=5, columnspan=2)
        output_frame.grid(column=0, row=2)

        # init menu
        menubar = tk.Menu(self, tearoff=0)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=self.destroy)

        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(
            label="Select first Landsat folder", command=self.select_first_path
        )
        editmenu.add_command(
            label="Select second Landsat folder", command=self.select_second_path
        )
        editmenu.add_separator()
        editmenu.add_command(
            label="Set coordinates",
            command=lambda: CoordinatesDialog(self, self.coords_var),
        )
        editmenu.add_command(label="Clear coordinates", command=self.coords_var.clear)

        modelsmenu = tk.Menu(menubar, tearoff=0)
        self.deforestation_model_var = tk.StringVar()
        self.deforestation_model_var.set(DE_FORESTATION_MODELS[-1])
        deforestation_modelmenu = tk.Menu(modelsmenu, tearoff=0)
        for i, model in enumerate(DE_FORESTATION_MODELS):
            deforestation_modelmenu.add_radiobutton(
                label=model, variable=self.deforestation_model_var, value=model
            )
        modelsmenu.add_cascade(label="De-forestation", menu=deforestation_modelmenu)

        plotmenu = tk.Menu(menubar, tearoff=0)
        # TODO: Add indices
        indices = ["NDVI", "GNDVI", "GRVI", "NDWI"]
        commands1 = [
            self.mc(1, "ndvi"),
            self.mc(1, "gndvi"),
            self.mc(1, "grvi"),
            self.mc(1, "ndwi"),
        ]
        commands2 = [
            self.mc(2, "ndvi"),
            self.mc(2, "gndvi"),
            self.mc(2, "grvi"),
            self.mc(2, "ndwi"),
        ]
        indices1menu = tk.Menu(plotmenu, tearoff=0)
        indices2menu = tk.Menu(plotmenu, tearoff=0)
        for index, command1, command2 in zip(indices, commands1, commands2):
            indices1menu.add_command(label=index, command=command1)
            indices2menu.add_command(label=index, command=command2)
        approximatedvisualizations1menu = tk.Menu(plotmenu, tearoff=0)
        approximatedvisualizations2menu = tk.Menu(plotmenu, tearoff=0)
        approximatedvisualizations = ["Land classification (NDVI based)"]
        commands1 = [self.mc(1, "ndvi_classes")]
        commands2 = [self.mc(2, "ndvi_classes")]
        for title, command1, command2 in zip(
            approximatedvisualizations, commands1, commands2
        ):
            approximatedvisualizations1menu.add_command(label=title, command=command1)
            approximatedvisualizations2menu.add_command(label=title, command=command2)
        plotmenu.add_cascade(label="Spectral Indices I", menu=indices1menu)
        plotmenu.add_cascade(label="Spectral Indices II", menu=indices2menu)
        plotmenu.add_cascade(
            label="Approximated Visualizations I", menu=approximatedvisualizations1menu
        )
        plotmenu.add_cascade(
            label="Approximated Visualizations II", menu=approximatedvisualizations2menu
        )

        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Edit", menu=editmenu)
        menubar.add_cascade(label="Models", menu=modelsmenu)
        menubar.add_cascade(label="Plots", menu=plotmenu)
        self.config(menu=menubar)

    def start(self):
        self.mainloop()

    def select_path(self):
        """
        Open directory dialog and get path. Calls select_path from main controller

        :return: directory, path (landsat 8), row (landsat 8), dt - datetime of acquiring snapshot
        """

        try:
            directory = askdirectory(title="Select LANDSAT 8 directory", mustexist=True)
            directory = Path(directory)
            result = self.mc.select_path(directory)
            if result is None:
                showerror(
                    "Error",
                    "Something went wrong during LANDSAT 8 directory loading. Make sure the directory exists and try again.",
                )
                return
        except:
            showerror(
                "Error",
                "Something went wrong during LANDSAT 8 directory loading. Make sure the directory exists and try again.",
            )
            return

        path, row, dt = result
        return directory, path, row, dt

    def select_first_path(self):
        dir_, path, row, dt = self.select_path()
        self.entry_1_var.set("Path: " + path + ", Row: " + row + " " + str(dt.date()))
        self.button_info1["state"] = "normal"
        self.mc.set_first_path(dir_)

    def select_second_path(self):
        dir_, path, row, dt = self.select_path()
        self.entry_2_var.set("Path: " + path + ", Row: " + row + " " + str(dt.date()))
        self.button_info2["state"] = "normal"
        self.mc.set_second_path(dir_)

    def info_first_path(self):
        json = self.mc.get_first_json()
        JSONVisualizer(self, json)

    def info_second_path(self):
        json = self.mc.get_second_json()
        JSONVisualizer(self, json)

    def clear_path(self):
        self.entry_1_var.set("Path: Row:")
        self.entry_2_var.set("Path: Row:")
        self.button_info1["state"] = "disabled"
        self.button_info2["state"] = "disabled"
        self.mc.clear_paths()

    @staticmethod
    def error(msg):
        """
        Shows error to user

        :param msg: message to display
        """
        showerror("Error", msg)

    """
    start_progressbar is invoked from within of the class -> progress bar is initialized (via init_progressbar)
        and sets update_progressbar function (works as interval in JS)
    update_progressbar is called every n ms, if percentage (on main controller) is 1 (100%) PB will be destroyed.
    destroy_progressbar is called to destroy progressbar (only way). 
    """

    def init_progressbar(self):
        self.pb = ProgressbarDialog(self)

    def update_progressbar(self):
        p = self.mc.percentage
        if p == 1:
            self.destroy_progressbar()
            return
        self.pb.update_pb(p)
        self.after(20, self.update_progressbar)

    def destroy_progressbar(self):
        self.pb.destroy()
        self.pb = None

    @staticmethod
    def start_progressbar():
        Application.self.after(1, Application.self.init_progressbar)
        Application.self.after(20, Application.self.update_progressbar)


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
        self.wm_title("Coordinates selection")
        self.wm_geometry("400x200+100+100")
        self.wm_iconphoto(True, tk.PhotoImage(file=IMAGES_DIR / "ico.png"))
        self.wm_resizable(False, False)

        # init grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(self, height=4, listvariable=self.coords_var)
        self.listbox.grid(column=0, row=0, columnspan=3, sticky="EW")

        tk.Button(self, command=self.del_selected, text="Delete").grid(
            column=0, columnspan=3, row=1, ipady=5, sticky="EW"
        )
        tk.Label(self, text="Coord (lat, lon): ").grid(column=0, row=2)

        self.coord_var = tk.StringVar(self, value="")  # coordinate var for entry field
        tk.Entry(self, textvariable=self.coord_var).grid(
            column=1, pady=10, row=2, padx=5, sticky="EW"
        )

        tk.Button(self, command=self.add, text="Add").grid(
            column=2, padx=5, pady=10, row=2, sticky="EW"
        )
        tk.Button(self, command=self.destroy, text="Close").grid(
            column=0, columnspan=3, row=5, ipady=5, sticky="EW"
        )

    def del_selected(self):
        """
        Deletes selected field
        """

        if self.listbox.curselection() != ():
            index = self.listbox.curselection()[0]
            self.coords_var.pop(index)
        else:
            showinfo("Info", "Select some element first to delete.")

    def add(self):
        """
        Checks coordinates and adds new coordinate to list
        """

        s = self.coord_var.get().replace(" ", "")  # get coordinate
        if "," in s:  # if format isn't correct
            coords = s.split(",")
            if len(coords) != 2:  # if user entered some bullshit
                showerror(
                    "Error",
                    'Wrong coordinates input. Must be entered in form: "lat, lon". Use "." as float separator',
                )
            else:
                lat = coords[0]  # get latitude
                lon = coords[1]  # get longitude
                try:
                    lat = float(lat)  # try cast to float
                    lon = float(lon)  # -//-

                    point = str(lat) + ", " + str(lon)
                    self.coords_var.append(point)

                    self.coord_var.set("")  # clear Entry field
                except:
                    showerror(
                        "Error",
                        'Wrong coordinates input. Must be entered two numbers in form: "lat, lon".',
                    )
        else:
            showerror(
                "Error",
                'Wrong coordinates input. Should be entered in form: "lat, lon".',
            )


class ProgressbarDialog(tk.Toplevel):
    """
    A class that shows user progressbar in new window.
    This window is impossible to close
    """

    def __init__(self, master):
        super().__init__(master)

        self.master = self.root = master
        self.percentage = 0  # progress in percent (from 0 to 1)
        self.bar_size = 100

        # init ui
        self.wm_title("Operation in progress")
        self.wm_geometry("300x200+100+100")
        self.wm_resizable(False, False)

        # init grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # init gui elements
        tk.Label(self, text="Current operation takes some time").grid(column=0, row=0)
        self.pb = Progressbar(
            self, orient="horizontal", length=self.bar_size, mode="determinate"
        )
        self.pb.grid(column=0, row=1)
        self.pb_text = tk.Label(
            self, text="% left:"
        )  # Label will be changed every update
        self.pb_text.grid(column=0, row=2)

        # make exit not possible
        def empty():
            pass

        self.protocol("WM_DELETE_WINDOW", empty)

    def update_pb(self, percentage):
        """
        Method updates PB

        :param percentage: of progress (from 0 to 1)
        """

        delta = percentage - self.percentage  # calculate delta of progress
        self.pb.step(delta * self.bar_size)  # calculate and apply delta
        self.percentage = percentage  # save new previous percentage
        self.pb_text.configure(
            text=str(round(self.percentage * 100_00) / 100.0) + "%"
        )  # update text
        self.update()  # update widget


class JSONVisualizer(tk.Toplevel):
    def __init__(self, master, json):
        super().__init__(master)

        self.master = self.root = master

        # init ui
        self.wm_title("JSON Viewer")
        self.wm_geometry("500x500+100+100")
        self.wm_resizable(False, False)

        # init grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # init gui elements
        tree = tk.ttk.Treeview(self)
        tree.column("# 0", width=2000)

        # attach a Horizontal (x) scrollbar to the frame
        treeXScroll = tk.ttk.Scrollbar(self, orient="horizontal")
        treeXScroll.configure(command=tree.xview)
        tree.configure(xscrollcommand=treeXScroll.set)

        def add_json_array_to_tree(arr, parent=""):
            for el in arr:
                if type(el).__name__ == "dict":
                    id_of_inserted_element = tree.insert(parent, "end")
                    add_json_object_to_tree(el, id_of_inserted_element)
                    continue
                elif type(el).__name__ == "list":
                    id_of_inserted_element = tree.insert(parent, "end")
                    add_json_array_to_tree(el, id_of_inserted_element)
                    continue
                tree.insert(parent, "end", text=el)

        def add_json_object_to_tree(obj, parent=""):
            keys = list(obj.keys())
            for key in keys:
                data = obj[key]
                id_of_inserted_element = tree.insert(parent, "end", text=key)
                if type(data).__name__ == "list":
                    add_json_array_to_tree(data, id_of_inserted_element)
                elif type(data).__name__ == "dict":
                    add_json_object_to_tree(data, id_of_inserted_element)
                else:
                    tree.insert(id_of_inserted_element, "end", text=str(data))

        add_json_object_to_tree(json)
        tree.grid(column=0, row=0, sticky="NSEW")
        treeXScroll.grid(column=0, row=1, sticky="WE")

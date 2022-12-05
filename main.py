import tkinter as tk


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        # init default window settings
        self.wm_title('Forestry')
        self.wm_geometry('400x600+100+100')
        self.wm_iconphoto(True, tk.PhotoImage(file='./assets/images/ico.png'))
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.wm_resizable(False, False)

        # init ui
        # init input fields
        input_frame = tk.Frame(self)
        tk.Label(input_frame, text='Landsat 8 first: ').grid(column=0, row=0)
        tk.Label(input_frame, text='Landsat 8 second: ').grid(column=0, row=1)

        entry_1_var = tk.StringVar(input_frame, value='Path: Row:')
        entry_2_var = tk.StringVar(input_frame, value='Path: Row:')
        tk.Entry(input_frame, state='readonly', textvariable=entry_1_var).grid(column=1, row=0)
        tk.Entry(input_frame, state='readonly', textvariable=entry_2_var).grid(column=1, row=1)

        tk.Button(input_frame, text='Select').grid(column=2, row=0)
        tk.Button(input_frame, text='Select').grid(column=2, row=1)

        tk.Button(input_frame, text='Clear').grid(column=0, row=2, columnspan=3, sticky='EW', pady=5)
        tk.Button(input_frame, text='Exit', command=self.destroy).grid(column=0, row=3, columnspan=3, sticky='EW', pady=5)
        input_frame.grid(column=0, row=0)

        # init control panel
        control_frame = tk.Frame(self)
        self.coords_var = tk.Variable(self)
        tk.Listbox(control_frame, height=4, listvariable=self.coords_var).grid(column=0, row=0, rowspan=3, sticky='NS')
        tk.Button(control_frame, text='Set coordinates').grid(column=1, row=0, pady=5, sticky='EW')
        tk.Button(control_frame, text='Clear coordinates').grid(column=1, row=1, pady=5, sticky='EW')
        tk.Button(control_frame, text='Visualize').grid(column=1, row=2, pady=5, sticky='EW')
        control_frame.grid(column=0, row=1)

        # init output frame
        output_frame = tk.Frame(self)
        tk.Label(output_frame, text='NDVI 1').grid(column=0, row=0)
        tk.Label(output_frame, text='NDVI 2').grid(column=1, row=0)
        self.pi_ndvi = tk.PhotoImage(file='./assets/images/ndvi.png')
        tk.Button(output_frame, image=self.pi_ndvi).grid(column=0, row=1)
        tk.Button(output_frame, image=self.pi_ndvi).grid(column=1, row=1)

        tk.Label(output_frame, text='GNDVI 1').grid(column=0, row=2)
        tk.Label(output_frame, text='GNDVI 2').grid(column=1, row=2)
        self.pi_gndvi = tk.PhotoImage(file='./assets/images/gndvi.png')
        tk.Button(output_frame, image=self.pi_gndvi).grid(column=0, row=3)
        tk.Button(output_frame, image=self.pi_gndvi).grid(column=1, row=3)

        tk.Label(output_frame, text='NDWI 1').grid(column=0, row=4)
        tk.Label(output_frame, text='NDWI 2').grid(column=1, row=4)
        self.pi_ndwi = tk.PhotoImage(file='./assets/images/ndwi.png')
        tk.Button(output_frame, image=self.pi_ndwi).grid(column=0, row=5)
        tk.Button(output_frame, image=self.pi_ndwi).grid(column=1, row=5)

        tk.Label(output_frame, text='De-forestation map').grid(column=0, row=6, columnspan=2)
        self.pi_def = tk.PhotoImage(file='./assets/images/def.png')
        tk.Button(output_frame, image=self.pi_def).grid(column=0, row=7, columnspan=2)
        output_frame.grid(column=0, row=2)


if __name__ == '__main__':
    app = Application()
    app.mainloop()

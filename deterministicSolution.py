import tkinter as tk
from tkinter import ttk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from functions import *

# Sample data (replace with your actual data generation logic)
class MyApp:
    def __init__(self,root):
        # Select case problem Lines, Area, Volume
        self.Solving_Case = "Lines"
        # Load data
        self.load_data()
        self.root = root
        self.root.title("Environment Simulation")
        # Create a frame for the plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(padx=10, pady=10)
        # DEFINE CONTROLS
        # Create a slider for input
        self.sdSegment = 0
        self.slider_label = tk.Label(self.plot_frame, text="Select Geometry Segment:")
        self.slider_label.pack()
        self.slider = ttk.Scale(self.plot_frame, from_=0, to=np.size(self.coor_WK,1)-1, orient=tk.HORIZONTAL, command=self.update_plot)
        self.slider.pack()
        # Create a button to trigger calculations (can be linked to a specific function)
        self.calculate_button = tk.Button(self.plot_frame, text="Calculate", command=self.calculate_angle)
        self.calculate_button.pack()
        # Create a textbox to display results (can be modified to show specific values)
        self.textbox = tk.Entry(self.plot_frame)
        self.textbox.pack()
        # SETUP PLOT
        # Matplotlib 3D figure
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.view_init(0, -90, 0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax.grid(False)
        self.ax.set_facecolor('white')
        self.ax.axis('off')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_title("Environment Simulation")
        # Plot data
        self.translation = np.array([0,0,0])
        self.rotation = np.array([0,0,0])
        sdSegment = int(self.slider.get())
        self.rotation = [0,0,0]
        coor_WK, coor_Laser = self.get_data()
        if self.Solving_Case.upper() == "LINES":
            x, y, z = [np.array(coor_WK[0][0, :0]), np.array(coor_WK[0][1, :0]), np.array(coor_WK[0][2, :0])]
            self.scWorkpiece = self.ax.scatter(x, y, z, c='black', label='Workpiece')
            self.pltWorkpiece, = self.ax.plot(x, y, z, c='black', label='Workpiece')
            x, y, z = [np.array(coor_Laser[0, :]), np.array(coor_Laser[1, :]), np.array(coor_Laser[2, :])]
            self.scLaser = self.ax.scatter(x, y, z, c='red', label='Laser')
            self.pltLaser, = self.ax.plot(x, y, z, c='red', label='Workpiece')
        if self.Solving_Case.upper() == "AREA":
            x, y, z = [np.array(coor_WK[0][0, :0]), np.array(coor_WK[0][1, :0]), np.array(coor_WK[0][2, :0])]
            self.scWorkpiece = self.ax.scatter(x, y, z, c='black', label='Workpiece')
            self.pltWorkpiece, = self.ax.plot(x, y, z, c='black', label='Workpiece', linestyle='dashed')
            x, y, z = [np.array(coor_WK[1][0, :0]), np.array(coor_WK[1][1, :0]), np.array(coor_WK[1][2, :0])]
            self.pltWorkpiece_L, = self.ax.plot(x, y, z, c='black', label='Workpiece')
            x, y, z = [np.array(coor_Laser[0, :]), np.array(coor_Laser[1, :]), np.array(coor_Laser[2, :])]
            self.scLaser = self.ax.scatter(x, y, z, c='red', label='Laser')
            self.pltLaser, = self.ax.plot(x, y, z, c='red', label='Workpiece', linestyle='dashed')
        self.ax.set_aspect('equal')
        self.canvas.draw()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    def load_data(self):
        # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Triangle_C.txt"
        # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Triangle4_C.txt"
        file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/HourGlass.txt"
        # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Star.txt"
        # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Square.txt"
        # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/spiral_coordinates.txt"
        # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Truss_HalfBCC.txt"
        # file_path =  "C:/Users/salin/Documents/Doctorado/LaserOcclusion/UnitLattices_HalfBCC.txt"
        points = parse_file(file_path, 'gcode')
        points[:, 2] = points[:, 2] * -1
        self.coor_WK = (HT(points.transpose(), [0,0,0], [0,0,0], [0,0,0], 1)[0]).transpose()#np.array([[0, 0, 0, 0, 0, 0, 0], [0, 5, 10, 15, 20, 25, 30], [0, 5, 0, 5, 0, 5, 0]])
        self.coor_Laser = np.array([[0,0,0],[-10,0,3.63970234266202]]).transpose() #np.array([[0, 0], [-10, 0], [3.63970234266202, 0]])
        if self.Solving_Case.upper() == "AREA":
            pass
            # Make offset better
            # for c,coor in enumerate(self.coor_WK):
            # self.coor_WK_L = (HT(self.coor_WK , [0,0,0], [0,0,0], [0,0,0], 2)[0]).transpose()
            # self.coor_WK_R = (HT(self.coor_WK , [0,0,0], [0,0,0], [0,0,0], 0.5)[0]).transpose()
        return
    def get_data(self):
        translation = self.translation
        rotation = self.rotation
        coor_WK = (HT(self.coor_WK,[0,0,0],-1*translation,[0,0,0],1)[0]).transpose()
        coor_WK = (HT(coor_WK, rotation, [0,0,0], [0, 0, 0], 1)[0]).transpose()
        if self.Solving_Case.upper() == "AREA":
            coor_WK_L = (HT(self.coor_WK_L, [0, 0, 0], -1 * translation, [0, 0, 0], 1)[0]).transpose()
            coor_WK_L = (HT(coor_WK_L, rotation, [0, 0, 0], [0, 0, 0], 1)[0]).transpose()
            coor_WK_R = (HT(self.coor_WK_R, [0, 0, 0], -1 * translation, [0, 0, 0], 1)[0]).transpose()
            coor_WK_R = (HT(coor_WK_R, rotation, [0, 0, 0], [0, 0, 0], 1)[0]).transpose()
            return [coor_WK,coor_WK_L,coor_WK_R], self.coor_Laser
        else:
            return [coor_WK,0,0], self.coor_Laser

    # Function to update plot based on slider value
    def update_plot(self,event):
        self.sdSegment = int(self.slider.get())
        self.translation = self.coor_WK[:, self.sdSegment] - self.coor_WK[:, 0]
        self.mod_coor_WK = self.get_data()[0]
        x,y,z = [np.array(self.mod_coor_WK[0][0,:self.sdSegment+1]),
               np.array(self.mod_coor_WK[0][1,:self.sdSegment+1]),
               np.array(self.mod_coor_WK[0][2,:self.sdSegment+1])]
        self.scWorkpiece._offsets3d = (x, y, z)
        self.pltWorkpiece.set_data_3d(x, y, z)
        if self.Solving_Case.upper() == "AREA":
            x, y, z = [np.array(self.mod_coor_WK[1][0, :self.sdSegment + 1]),
                       np.array(self.mod_coor_WK[1][1, :self.sdSegment + 1]),
                       np.array(self.mod_coor_WK[1][2, :self.sdSegment + 1])]
            self.pltWorkpiece_L.set_data_3d(x, y, z)
        # self.ax.set_aspect('equal')
        self.canvas.draw()

    def calculate_angle(self):
        if self.sdSegment > 0:
            self.d_coor_WK = self.mod_coor_WK[0][:,1:]-self.mod_coor_WK[0][:,:np.size(self.coor_WK,1) - 1]
            self.d_coor_Laser = self.coor_Laser[:, 1:]-self.coor_Laser[:, :np.size(self.coor_Laser, 1) - 1]
            self.theta = []
            for i in range(0,np.size(self.coor_WK,1)-1):
                t = np.arctan2(self.d_coor_WK[2,i],self.d_coor_WK[0,i])
                self.theta.append(np.pi+t if t <= 0 else t)
            print(f"thetas: {np.rad2deg(self.theta[self.sdSegment-1])}")
            self.ALPHA = []
            for i in range(0,np.size(self.coor_Laser,1)-1):
                self.ALPHA.append(np.arctan2(self.d_coor_Laser[2,i],self.d_coor_Laser[0,i]))
            print(f"ALPHAS: {np.rad2deg(self.ALPHA)}")
            self.alpha = np.full_like(self.theta,self.ALPHA[0])-self.theta
            print(f"alpha: {np.rad2deg(self.alpha)}")
            if self.alpha[self.sdSegment-1] > 0 and self.sdSegment > 1:
                if self.alpha[self.sdSegment-2]>0:
                    self.rotation = [0,-1*np.rad2deg(self.alpha[self.sdSegment-1]),0]
                    self.update_plot(root)
                    print(f"Segment: {self.sdSegment} "
                          f"rotation: {self.rotation}")
            # else:
                # self.rotation = [0,0,0]
                # print([0, , 0])
        return

    def on_close(self):
        # This function is called when the window is closed
        print("Closing the application.")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    # test_performance(20,10,500)  # Call every 500 milliseconds (adjust as needed)
    # update_viewAngles(500)
    root.mainloop()
# Create the main window

# # Create a frame for the plot
# plot_frame = tk.Frame(root)
# plot_frame.pack(padx=10, pady=10)
# # Create a figure for the 3D plot
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection='3d')
# # Create a canvas to embed the plot in the GUI
# canvas = FigureCanvasTkAgg(fig, master=plot_frame)
# canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
# # Create a slider for input
# slider_label = tk.Label(plot_frame, text="X-Value:")
# slider_label.pack()
# slider = ttk.Scale(plot_frame, from_=0, to=10, orient=tk.HORIZONTAL, command=update_plot)
# slider.pack()
# # Create a button to trigger calculations (can be linked to a specific function)
# calculate_button = tk.Button(plot_frame, text="Calculate", command=update_plot)
# calculate_button.pack()
# # Create a textbox to display results (can be modified to show specific values)
# textbox = tk.Entry(plot_frame)
# textbox.pack()
#
# # Run the main loop
# root.mainloop()

# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.pyplot as plt
# import numpy as np
# import time
#
# # Sample data (replace with your data loading logic)
# coor_WKx = [0,0,0,0,0,0,0]
# coor_WKy = [0, 5, 10, 15, 20, 25, 30]
# coor_WKz = [0, 5, 0, 5, 0, 5, 0]
#
# coor_Ly = [-10, 0]
# coor_Lz = [3.63970234266202, 0]
#
# # Matplotlib 3D figure
# figure = plt.figure()
# ax = figure.add_subplot(111, projection='3d')
# # self.ax.view_init(45, 45, -45)
# # Setup plot
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.set_title("Environment Simulation")
#
# x,y,z = [np.array(coor_WKy),np.array(coor_WKy),np.array(coor_WKy)]
# pltWorkpiece = ax.scatter(x, y, z, c='red', label='Laser')
# ax.set_aspect('equal')
# canvas = FigureCanvasTkAgg(figure)
# canvas.draw()
#
# time.sleep(10)

# dWKy
# dWKz


# Function to generate 3D plots
# def generate_plots(data):
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#   x, y, z = zip(*data)
#
#   # Use ax.scatter for both plots
#   ax1.scatter(x, y, z, c=z, cmap='viridis')
#   # ax1.set_xlabel('X')
#   # ax1.set_ylabel('Y')
#   # ax1.set_zlabel('Z')
#   # ax1.set_title('Plot 1')
#
#   ax2.scatter(x, y, c=z, cmap='viridis')
#   # ax2.set_xlabel('X')
#   # ax2.set_ylabel('Y')
#   # ax2.set_title('Plot 2')
#   # fig.tight_layout()
#   return fig
#
# # Function to update tables
# def update_tables():
#   # Clear existing table entries (if any)
#   for row in range(10):  # Adjust for your table size
#     for col in range(3):
#       table1.delete(tk.CELL, row, col)
#       table2.delete(tk.CELL, row, col)
#
#   # Fill tables with data (replace with your data loading logic)
#   for row, (r1, r2) in enumerate(zip(data1, data2)):
#     for col, value in enumerate(r1):
#       table1.insert(tk.END, value, bd=1)
#     for col, value in enumerate(r2):
#       table2.insert(tk.END, value, bd=1)
#
# # Create the main window
# root = tk.Tk()
# root.title("3D Plots and Data Tables")
#
# # Create a notebook for tabs
# notebook = ttk.Notebook(root)
# notebook.pack(padx=10, pady=10, fill=tk.BOTH)
#
# # Create frames (containers) for each tab
# plot_frame = tk.Frame(notebook)
# table_frame = tk.Frame(notebook)
#
# # Add frames to notebook tabs
# notebook.add(plot_frame, text="3D Plots")
# notebook.add(table_frame, text="Data Tables")
#
# # Create buttons (can be placed anywhere in the GUI)
# load_button = tk.Button(root, text="LOAD", command=update_tables)
# load_button.pack(padx=5, pady=5)
# update_button = tk.Button(root, text="UPDATE", command=update_tables)
# update_button.pack(padx=5, pady=5)
#
# # Generate initial plots (replace with your data)
# fig1 = generate_plots(data1)
# fig2 = generate_plots(data2)
#
# # Embed plots in canvas
# canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
# canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame)
#
# canvas1.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)
# canvas2.get_tk_widget().pack(side=tk.RIGHT, padx=10, pady=10)
#
# # Create tables
# table1 = ttk.Treeview(table_frame, columns=("Col1", "Col2", "Col3"), show="headings")
# table1.heading("#0", text="Row")
# table1.heading("Col1", text="Data 1")
# table1.heading("Col2", text="Data 2")
# table1.heading("Col3", text="Data 3")
# table1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
#
# table2 = ttk.Treeview(table_frame, columns=("Col1", "Col2", "Col3"), show="headings")
# table2.heading("#0", text="Row")
# table2.heading("Col1", text="Data 1")
# table2.heading("Col2", text="Data 2")
# table2.heading("Col3", text="Data 3")
# table2.pack(side=tk.RIGHT, expand=True, fill=tk.X)  # Place table2 on the right side
#
# root.mainloop()

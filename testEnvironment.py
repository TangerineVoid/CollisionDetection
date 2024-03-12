import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from init import *

# Errores pendientes:
# Mostrar en Action los 3 angulos de movimiento, y agregarlo al estado.
# El campo de collision no se está mostrando bien
# Cuidado que puede formatear tu computador si no tienes las credenciales
class MyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Environment Simulation")

        # Matplotlib 3D figure
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # Text label
        self.label_EnvData = tk.StringVar()
        self.label_EnvData.set(f"Environment Data:\n"
                               f"Collision: {bool(env.state[0])}\n"
                               f"Collision with Laser: {bool(env.check_Collision()[1])}\n"
                               f"Collision with Filament: {bool(env.check_Collision()[2])}\n"
                               f"Action: {float((env.state[1]))}\n")
        self.label = tk.Label(root, textvariable=self.label_EnvData, justify=tk.LEFT)
        self.label.pack(side=tk.RIGHT, padx=10, pady=10)
        # Setup plot
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_title("Environment Simulation")
        self.show_trajectory = False
        # Plot system: Laser, Filament, Piece
        # If it is desired to downsamble
        ds = 20
        x,y,z = [np.array(lcd.points)[::ds, 0],np.array(lcd.points)[::ds, 1],np.array(lcd.points)[::ds, 2]]
        self.pltLaser = self.ax.scatter(x, y, z, c='red', label='Laser')
        x, y, z = [np.array(fcd.points)[::ds, 0], np.array(fcd.points)[::ds, 1], np.array(fcd.points)[::ds, 2]]
        self.pltFilament = self.ax.scatter(x, y, z, c='blue', label='Filament')
        x, y, z = [np.array(env.mod_dcgeometry.points)[::ds, 0], np.array(env.mod_dcgeometry.points)[::ds, 1], np.array(env.mod_dcgeometry.points)[::ds, 2]]
        self.pltPiece = self.ax.scatter(x, y, z, c='black', label='Piece')
        if self.show_trajectory:
            x, y, z = [np.array(env.mod_trajectory)[::ds, 0], np.array(env.mod_trajectory)[::ds, 1],
                       np.array(env.mod_trajectory)[::ds, 2]]
            self.pltTrajectory = self.ax.scatter(x, y, z, c='black', label='Trajectory')
        x, y, z = [np.array(env.mod_TCP)[0,3],np.array(env.mod_TCP)[1,3],np.array(env.mod_TCP)[2,3]]
        u, v, w = [np.dot(np.array(env.mod_TCP)[:3,:3],[0,0,1])[0],
                   np.dot(np.array(env.mod_TCP)[:3,:3],[0,0,1])[1],
                   np.dot(np.array(env.mod_TCP)[:3,:3],[0,0,1])[2]]
        self.TCP = self.ax.quiver(x, y, z, 3*u, 3*v, 3*w,  color='red')
        # # Autoscale to adjust the axis range to fit the shown plot
        self.ax.set_aspect('equal')
        self.canvas.draw()
        # Bind keys
        self.root.bind('q', lambda event: self.handle_key('Q'))
        self.root.bind('w', lambda event: self.handle_key('W'))
        self.root.bind('e', lambda event: self.handle_key('E'))
        self.root.bind('a', lambda event: self.handle_key('A'))
        self.root.bind('s', lambda event: self.handle_key('S'))
        self.root.bind('d', lambda event: self.handle_key('D'))
        self.root.bind('<Left>', lambda event: self.handle_key('Left'))
        self.root.bind('<Right>', lambda event: self.handle_key('Right'))
        self.root.bind('<Up>', lambda event: self.handle_key('Up'))
        self.root.bind('<Down>', lambda event: self.handle_key('Down'))

        # Buttons
        button_commands = ['E', 'Q', 'W', 'S', 'D', 'A','Up','Down','Reset']
        button_text = ['+Z','-Z','+Y','-Y','+X','-X', '+Step', '-Step','Reset']
        for i,command in enumerate(button_commands):
            button = ttk.Button(root, text=button_text[i], command=lambda c=command: self.handle_key(c))
            button.pack(side=tk.LEFT)
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)


    def handle_key(self, key):
        # This function is called when a key or button is pressed
        # You can modify this function based on your specific needs
        print(f'Key pressed: {key}')
        if key == 'Up':
            env.continue_process(20)
        elif key == 'Down':
            env.continue_process(-20)
        elif key == 'A':
            env.step(-10,'X')
        elif key == 'D':
            env.step(10, 'X')
        elif key == 'S':
            env.step(-10,'Y')
        elif key == 'W':
            env.step(10, 'Y')
        elif key == 'Q':
            env.step(-10,'Z')
        elif key == 'E':
            env.step(10, 'Z')
        elif key == 'Reset':
            env.reset()
        # Update 3D plot based on the pressed key
        self.update_EnvData()
        self.ax.set_title("Environment Simulation")
        # Plot system: Laser, Filament, Piece
        # If it is desired to downsample
        ds = 20
        x, y, z = [np.array(env.mod_dcgeometry.points)[::ds, 0], np.array(env.mod_dcgeometry.points)[::ds, 1],
                   np.array(env.mod_dcgeometry.points)[::ds, 2]]
        self.pltPiece._offsets3d = (x,y,z)  # Update only x and y positions
        if self.show_trajectory:
            x, y, z = [np.array(env.mod_trajectory)[::ds, 0], np.array(env.mod_trajectory)[::ds, 1],
                       np.array(env.mod_trajectory)[::ds, 2]]
            self.pltTrajectory._offsets3d = (x,y,z)
        x, y, z = [np.array(env.mod_TCP)[0, 3], np.array(env.mod_TCP)[1, 3], np.array(env.mod_TCP)[2, 3]]
        u, v, w = [np.dot(np.array(env.mod_TCP)[:3, :3], [0, 0, 1])[0],
                   np.dot(np.array(env.mod_TCP)[:3, :3], [0, 0, 1])[1],
                   np.dot(np.array(env.mod_TCP)[:3, :3], [0, 0, 1])[2]]
        self.TCP.remove()
        self.TCP = self.ax.quiver(x, y, z, 3*u, 3*v, 3*w,  color='red')
        self.ax.set_aspect('equal')
        self.canvas.draw()

    def update_EnvData(self):
        # Update the text label based on the pressed key or button
        # You can modify this function to update the label text dynamically
        string = f"Environment Data:\n" \
                f"Collision: {bool(env.state[0])}\n" \
                f"Collision with Laser: {bool(env.check_Collision()[1])}\n" \
                f"Collision with Filament: {bool(env.check_Collision()[2])}\n" \
                f"Action: {float((env.state[1]))}\n"
        self.label_EnvData.set(string)

    def on_close(self):
        # This function is called when the window is closed
        print("Closing the application.")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
from scipy.interpolate import griddata
import polynomial
from polynomial import polyfit2d, polyval2d
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Resampler and Fitter")
        
        # Initialize data attributes
        self.data = None
        self.shifted_data = None
        self.resampled = None
        self.fitted = None

        # Control frame
        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Load button
        self.load_button = tk.Button(ctrl_frame, text="Load Point Cloud", command=self.load_data)
        self.load_button.pack(fill=tk.X, pady=2)

        # Resample step input
        tk.Label(ctrl_frame, text="Resample Steps:").pack(anchor='w')
        self.step_var = tk.StringVar(value="100")
        self.step_entry = tk.Entry(ctrl_frame, textvariable=self.step_var)
        self.step_entry.pack(fill=tk.X, pady=2)

        # Degree input
        tk.Label(ctrl_frame, text="Polynomial Degree:").pack(anchor='w')
        self.degree_var = tk.StringVar(value="4")
        self.degree_entry = tk.Entry(ctrl_frame, textvariable=self.degree_var)
        self.degree_entry.pack(fill=tk.X, pady=2)

        # Resample & Fit button
        self.resample_fit_button = tk.Button(ctrl_frame, text="Resample & Fit", state=tk.DISABLED, command=self.resample_and_fit)
        self.resample_fit_button.pack(fill=tk.X, pady=2)

        # Output button
        self.output_button = tk.Button(ctrl_frame, text="Export Results", state=tk.DISABLED, command=self.export_results)
        self.output_button.pack(fill=tk.X, pady=2)

        # Plot frame
        plot_frame = tk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar for interaction (pan/zoom/rotate)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def load_data(self):
        # Reset state
        self.data = None
        self.shifted_data = None
        self.resampled = None
        self.fitted = None
        self.resample_fit_button.config(state=tk.DISABLED)
        self.output_button.config(state=tk.DISABLED)
        self.ax.clear()
        self.canvas.draw()

        file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.txt *.csv"), ("All Files", "*")])
        if not file_path:
            return
        try:
            data = np.loadtxt(file_path, delimiter=',')
        except Exception:
            data = np.loadtxt(file_path)
        if data.shape[1] < 3:
            messagebox.showerror("Error", "File must have at least three columns for x, y, z.")
            return
        self.data = data[:, :3]
        # Shift center to origin
        x, y, z = self.data[:,0], self.data[:,1], self.data[:,2]
        cx = (x.min() + x.max()) / 2
        cy = (y.min() + y.max()) / 2
        shifted = self.data.copy()
        shifted[:,0] -= cx
        shifted[:,1] -= cy
        self.shifted_data = shifted
        # Plot original shifted data with smaller, transparent markers
        self.ax.clear()
        self.ax.scatter(shifted[:,0], shifted[:,1], shifted[:,2], s=2, alpha=0.6)
        self.ax.set_title("Shifted Point Cloud")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_box_aspect([1,1,1])
        self.canvas.draw()
        # Enable resample button
        self.resample_fit_button.config(state=tk.NORMAL)

    def resample_and_fit(self):
        if self.shifted_data is None:
            return
        try:
            steps = int(self.step_var.get())
            deg = int(self.degree_var.get())
        except ValueError:
            messagebox.showerror("Error", "Steps and Degree must be integers.")
            return
        x = self.shifted_data[:,0]
        y = self.shifted_data[:,1]
        z = self.shifted_data[:,2]
        # Determine grid resolution for square cells
        range_x = x.max() - x.min()
        range_y = y.max() - y.min()
        if range_x <= range_y:
            steps_x = steps
            # scale steps for y to maintain square spacing
            steps_y = int(round(steps * (range_y / range_x))) if range_x > 0 else steps
        else:
            steps_y = steps
            steps_x = int(round(steps * (range_x / range_y))) if range_y > 0 else steps
        # Create grid
        xi = np.linspace(x.min(), x.max(), steps_x)
        yi = np.linspace(y.min(), y.max(), steps_y)
        grid_x, grid_y = np.meshgrid(xi, yi)
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
        mask = ~np.isnan(grid_z)
        xf = grid_x[mask]
        yf = grid_y[mask]
        zf = grid_z[mask]
        self.resampled = (xf, yf, zf)
        # Fit polynomial
        coeff = polyfit2d(xf, yf, zf, deg)
        self.coeff = coeff
        # Compute fitted values
        z_fit = polyval2d(xf, yf, coeff)
        self.fitted = (xf, yf, z_fit)
        # Plot resampled and fitted with smaller, transparent markers
        self.ax.clear()
        self.ax.scatter(xf, yf, zf, s=2, alpha=0.6, label='Resampled')
        self.ax.scatter(xf, yf, z_fit, s=2, alpha=0.6, label='Fitted')
        self.ax.set_title(f"Resampled vs Fitted ({steps_x}x{steps_y}, Degree {deg})")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_box_aspect([1,1,1])
        self.ax.legend()
        self.canvas.draw()
        # Enable export button
        self.output_button.config(state=tk.NORMAL)

    def export_results(self):
        if self.coeff is None or self.resampled is None or self.fitted is None:
            return
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return
        # Save coefficients
        coeff_path = os.path.join(dir_path, 'coeff.csv')
        np.savetxt(coeff_path, self.coeff, delimiter=',')
        # Save resampled point cloud
        xf, yf, zf = self.resampled
        resampled_path = os.path.join(dir_path, 'resampled.csv')
        np.savetxt(resampled_path, np.vstack([xf, yf, zf]).T, delimiter=',', comments='')
        # Save fitted point cloud
        xf2, yf2, zf2 = self.fitted
        fitted_path = os.path.join(dir_path, 'fitted.csv')
        np.savetxt(fitted_path, np.vstack([xf2, yf2, zf2]).T, delimiter=',', comments='')
        messagebox.showinfo("Export", f"Files saved in {dir_path}")

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

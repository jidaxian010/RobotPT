import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from scipy.spatial.transform import Rotation

class ConfigurablePlotter:
    """
    A flexible plotting class that allows easy configuration of subplots and their content.
    
    Usage:
        plotter = ConfigurablePlotter(figsize=(15, 10))
        plotter.add_subplot("acc_x", row=0, col=0, title="Acceleration X", ylabel="m/s²")
        plotter.add_subplot("acc_y", row=0, col=1, title="Acceleration Y", ylabel="m/s²")
        plotter.plot_data("acc_x", time_data, acc_x_data, label="Acc X", color="red")
        plotter.show()
    """
    
    def __init__(self, rows: int = 2, cols: int = 2, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the plotter.
        
        Args:
            rows: Number of subplot rows
            cols: Number of subplot columns
            figsize: Figure size (width, height)
        """
        self.rows = rows
        self.cols = cols
        self.figsize = figsize
        self.fig = None
        self.axes = {}
        self.subplot_configs = {}
        self.data_storage = {}
        
    def add_subplot(self, 
                   name: str, 
                   row: int, 
                   col: int, 
                   title: str = "", 
                   xlabel: str = "", 
                   ylabel: str = "",
                   grid: bool = True,
                   legend: bool = True):
        """
        Add a subplot configuration.
        
        Args:
            name: Unique identifier for this subplot
            row: Row position (0-indexed)
            col: Column position (0-indexed)
            title: Subplot title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
            legend: Whether to show legend
        """
        if row >= self.rows or col >= self.cols:
            raise ValueError(f"Position ({row}, {col}) exceeds grid size ({self.rows}, {self.cols})")
        
        self.subplot_configs[name] = {
            'position': (row, col),
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'grid': grid,
            'legend': legend
        }
        self.data_storage[name] = []
        
    def plot_data(self, 
                  subplot_name: str, 
                  x_data: np.ndarray, 
                  y_data: np.ndarray, 
                  label: str = "", 
                  color: str = None,
                  linestyle: str = '-',
                  linewidth: float = 1.0,
                  alpha: float = 1.0):
        """
        Add data to be plotted on a specific subplot.
        
        Args:
            subplot_name: Name of the subplot to plot on
            x_data: X-axis data
            y_data: Y-axis data
            label: Data series label
            color: Line color
            linestyle: Line style ('-', '--', ':', etc.)
            linewidth: Line width
            alpha: Transparency
        """
        if subplot_name not in self.subplot_configs:
            raise ValueError(f"Subplot '{subplot_name}' not configured. Add it first with add_subplot()")
        
        plot_params = {
            'x_data': x_data,
            'y_data': y_data,
            'label': label,
            'color': color,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'alpha': alpha
        }
        
        self.data_storage[subplot_name].append(plot_params)
        
    def create_figure(self):
        """Create the matplotlib figure and axes."""
        self.fig, axes_array = plt.subplots(self.rows, self.cols, figsize=self.figsize)
        
        # Store axes by name
        for name, config in self.subplot_configs.items():
            row, col = config['position']
            
            # Handle different subplot configurations
            if self.rows == 1 and self.cols == 1:
                # Single subplot case
                self.axes[name] = axes_array
            elif self.rows == 1 or self.cols == 1:
                # Single row or single column
                if self.rows == 1:
                    self.axes[name] = axes_array[col]
                else:  # self.cols == 1
                    self.axes[name] = axes_array[row]
            else:
                # Multiple rows and columns
                self.axes[name] = axes_array[row, col]
            
    def plot_all(self):
        """Plot all configured data on their respective subplots."""
        if self.fig is None:
            self.create_figure()
            
        for subplot_name, config in self.subplot_configs.items():
            ax = self.axes[subplot_name]
            
            # Plot all data series for this subplot
            for plot_params in self.data_storage[subplot_name]:
                # Build plot kwargs, excluding None values
                plot_kwargs = {}
                if plot_params['label']:
                    plot_kwargs['label'] = plot_params['label']
                if plot_params['color']:
                    plot_kwargs['color'] = plot_params['color']
                if plot_params['linestyle']:
                    plot_kwargs['linestyle'] = plot_params['linestyle']
                if plot_params['linewidth']:
                    plot_kwargs['linewidth'] = plot_params['linewidth']
                if plot_params['alpha']:
                    plot_kwargs['alpha'] = plot_params['alpha']
                
                ax.plot(plot_params['x_data'], 
                       plot_params['y_data'],
                       **plot_kwargs)
            
            # Apply subplot configuration
            if config['title']:
                ax.set_title(config['title'])
            if config['xlabel']:
                ax.set_xlabel(config['xlabel'])
            if config['ylabel']:
                ax.set_ylabel(config['ylabel'])
            if config['grid']:
                ax.grid(True, alpha=0.3)
            if config['legend'] and any(p['label'] for p in self.data_storage[subplot_name]):
                ax.legend()
                
    def show(self, save_path: Optional[str] = None):
        """
        Display the plot and optionally save it.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        self.plot_all()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
            
        plt.show()
        
    def clear(self):
        """Clear all data and reset the plotter."""
        self.data_storage = {name: [] for name in self.subplot_configs.keys()}
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axes = {}


#
def quaternion_to_rotation_matrix(quaternions):
    """Convert quaternion(s) [w, x, y, z] to rotation matrix(es)."""
    if quaternions.ndim == 1:
        # Single quaternion
        w, x, y, z = quaternions
        rotation_matrix = np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
        ])
        return rotation_matrix
    else:
        # Array of quaternions
        rotation_matrices = []
        for i in range(len(quaternions)):
            w, x, y, z = quaternions[i]
            rotation_matrix = np.array([
                [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
                [2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
            ])
            rotation_matrices.append(rotation_matrix)
        
        return np.array(rotation_matrices)


def quaternion_to_euler(quaternions):
    """Convert quaternion array to euler angles with unwrapping."""
    if quaternions.ndim == 1:
        # Single quaternion
        rotation = Rotation.from_quat(quaternions[[1, 2, 3, 0]])  # Convert [w,x,y,z] to [x,y,z,w]
        euler = rotation.as_euler('zyx', degrees=True)
        return euler
    else:
        # Array of quaternions
        euler_angles = []
        for i in range(len(quaternions)):
            # Convert [w,x,y,z] to [x,y,z,w] for scipy
            quat_scipy = quaternions[i, [1, 2, 3, 0]]
            rotation = Rotation.from_quat(quat_scipy)
            euler = rotation.as_euler('zyx', degrees=True)
            euler_angles.append(euler)
        
        euler_angles = np.array(euler_angles)
        
        # Apply unwrapping to prevent jumps
        # Convert to radians for unwrapping, then back to degrees
        euler_rad = np.radians(euler_angles)
        euler_unwrapped_rad = np.unwrap(euler_rad, axis=0)
        euler_unwrapped = np.degrees(euler_unwrapped_rad)
        
        return euler_unwrapped
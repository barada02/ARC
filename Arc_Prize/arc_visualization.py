import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any, Union, Optional
import io
from PIL import Image

# Define the color map based on the provided colors
ARC_COLORS = {
    0: "#010100",  # Black
    1: "#1e93fe",  # Blue
    2: "#f83d31",  # Red
    3: "#4fcc31",  # Green
    4: "#fedc00",  # Yellow
    5: "#999898",  # Gray
    6: "#e53ba2",  # Pink
    7: "#fe841b",  # Orange
    8: "#87d8f1",  # Light Blue
    9: "#921330"   # Burgundy
}

# Create a ListedColormap for matplotlib - convert hex to RGB
def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple (0-1 range for matplotlib)"""
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)

# Create colormap with RGB values (0-1 range)
ARC_CMAP = ListedColormap([hex_to_rgb(ARC_COLORS[i]) for i in range(10)])


def plot_grid(grid: np.ndarray, title: str = None) -> Image.Image:
    """
    Plot a single grid using the ARC color map
    
    Args:
        grid: 2D numpy array with values 0-9
        title: Optional title for the grid
    
    Returns:
        PIL Image of the plotted grid
    """
    try:
        # Convert to numpy array if it's not already
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
        
        # Ensure grid is 2D
        if len(grid.shape) != 2:
            raise ValueError(f"Grid must be 2D, got shape {grid.shape}")
        
        # Convert to float for imshow
        grid_float = grid.astype(float)
        
        # Create figure
        height, width = grid.shape
        figsize = (max(4, width * 0.5), max(4, height * 0.5))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the grid
        ax.imshow(grid_float, cmap=ARC_CMAP, vmin=0, vmax=9)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        
        # Remove tick marks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title if provided
        if title:
            ax.set_title(title)
        
        # Add cell values as text
        for i in range(height):
            for j in range(width):
                cell_value = int(grid[i, j])
                # Use white text for dark colors, black for light colors
                text_colors = {
                    0: 'white', 1: 'white', 2: 'white',
                    3: 'black', 4: 'black', 5: 'black',
                    6: 'white', 7: 'black', 8: 'black', 9: 'white'
                }
                text_color = text_colors.get(cell_value, 'black')
                ax.text(j, i, str(cell_value), 
                        ha='center', va='center', color=text_color)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        return image
        
    except Exception as e:
        # Create an error image
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"Error plotting grid: {str(e)}", 
                ha='center', va='center', color='red')
        ax.axis('off')
        
        # Convert error message to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        return image


def visualize_task(task: Dict[str, Any]) -> Dict[str, List[Image.Image]]:
    """
    Visualize all examples in an ARC task (input/output pairs)
    
    Args:
        task: ARC task dictionary containing 'train' and 'test' examples
    
    Returns:
        Dictionary with 'train' and 'test' keys, each containing lists of 
        tuples (input_image, output_image)
    """
    result = {'train': [], 'test': []}
    
    # Process training examples
    for i, example in enumerate(task.get('train', [])):
        input_grid = np.array(example.get('input', []))
        input_image = plot_grid(input_grid, f"Train {i+1} - Input")
        
        # Check if output exists
        if 'output' in example:
            output_grid = np.array(example.get('output', []))
            output_image = plot_grid(output_grid, f"Train {i+1} - Output")
        else:
            # Create a placeholder image for missing output
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.text(0.5, 0.5, "No output data available", 
                    ha='center', va='center', color='red')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            output_image = Image.open(buf)
            plt.close(fig)
        
        result['train'].append((input_image, output_image))
    
    # Process test examples
    for i, example in enumerate(task.get('test', [])):
        input_grid = np.array(example.get('input', []))
        input_image = plot_grid(input_grid, f"Test {i+1} - Input")
        
        # Check if output exists
        if 'output' in example:
            output_grid = np.array(example.get('output', []))
            output_image = plot_grid(output_grid, f"Test {i+1} - Output")
        else:
            # Create a placeholder image for missing output
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.text(0.5, 0.5, "No output data available", 
                    ha='center', va='center', color='red')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            output_image = Image.open(buf)
            plt.close(fig)
        
        result['test'].append((input_image, output_image))
    
    return result


def get_color_swatch():
    """
    Generate a color swatch showing all ARC colors with their indices
    
    Returns:
        PIL Image of the color swatch
    """
    fig, ax = plt.subplots(figsize=(10, 1))
    
    # Create a simple array with values 0-9
    swatch = np.array([list(range(10))])
    
    # Plot the swatch
    ax.imshow(swatch, cmap=ARC_CMAP, aspect='auto')
    
    # Add color indices as text
    for i in range(10):
        # Use same text colors as in plot_grid
        text_colors = {
            0: 'white', 1: 'white', 2: 'white',
            3: 'black', 4: 'black', 5: 'black',
            6: 'white', 7: 'black', 8: 'black', 9: 'white'
        }
        text_color = text_colors.get(i, 'black')
        ax.text(i, 0, str(i), ha='center', va='center', 
                color=text_color, fontsize=12, fontweight='bold')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("ARC Color Palette")
    
    # Convert to PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    
    return image
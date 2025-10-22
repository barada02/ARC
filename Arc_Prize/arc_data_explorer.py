import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from typing import Dict, List, Any, Tuple
import random
from collections import defaultdict

# Define paths to data files
DATA_DIR = os.path.join(os.path.dirname(__file__), 'arc-prize-2025')
TRAINING_CHALLENGES_PATH = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')
TRAINING_SOLUTIONS_PATH = os.path.join(DATA_DIR, 'arc-agi_training_solutions.json')
EVAL_CHALLENGES_PATH = os.path.join(DATA_DIR, 'arc-agi_evaluation_challenges.json')
EVAL_SOLUTIONS_PATH = os.path.join(DATA_DIR, 'arc-agi_evaluation_solutions.json')
TEST_CHALLENGES_PATH = os.path.join(DATA_DIR, 'arc-agi_test_challenges.json')

# Define a colormap for visualization
# ARC uses 10 colors (0-9)
CMAP = ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', 
    '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', 
    '#7FDBFF', '#870C25'
])

class ARCDataExplorer:
    def __init__(self):
        """Initialize the ARC data explorer."""
        self.training_challenges = self._load_json(TRAINING_CHALLENGES_PATH)
        self.training_solutions = self._load_json(TRAINING_SOLUTIONS_PATH)
        self.eval_challenges = self._load_json(EVAL_CHALLENGES_PATH)
        self.eval_solutions = self._load_json(EVAL_SOLUTIONS_PATH)
        self.test_challenges = self._load_json(TEST_CHALLENGES_PATH)
        
        print(f"Loaded {len(self.training_challenges)} training tasks")
        print(f"Loaded {len(self.eval_challenges)} evaluation tasks")
        print(f"Loaded {len(self.test_challenges)} test tasks")
    
    def _load_json(self, filepath: str) -> Dict:
        """Load a JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def visualize_task(self, task_id: str, dataset: str = 'training'):
        """Visualize a task with its train pairs and test input."""
        # Select the appropriate dataset
        if dataset == 'training':
            challenges = self.training_challenges
            solutions = self.training_solutions
        elif dataset == 'eval':
            challenges = self.eval_challenges
            solutions = self.eval_solutions
        elif dataset == 'test':
            challenges = self.test_challenges
            solutions = None  # No solutions for test set
        else:
            print(f"Unknown dataset: {dataset}")
            return
        
        # Check if the task exists
        if task_id not in challenges:
            print(f"Task {task_id} not found in {dataset} dataset")
            return
        
        task = challenges[task_id]
        
        # Calculate the number of rows and columns needed for the plot
        train_pairs = task['train']
        num_train_pairs = len(train_pairs)
        
        # Create a figure with subplots for train pairs and test input
        fig = plt.figure(figsize=(15, 5 * (num_train_pairs + 1)))
        fig.suptitle(f"Task {task_id} ({dataset} dataset)", fontsize=16)
        
        # Plot train pairs
        for i, pair in enumerate(train_pairs):
            # Input grid
            ax_input = fig.add_subplot(num_train_pairs + 1, 2, i * 2 + 1)
            self._plot_grid(pair['input'], ax_input)
            ax_input.set_title(f"Train Pair {i+1} - Input")
            
            # Output grid
            ax_output = fig.add_subplot(num_train_pairs + 1, 2, i * 2 + 2)
            self._plot_grid(pair['output'], ax_output)
            ax_output.set_title(f"Train Pair {i+1} - Output")
        
        # Plot test input
        ax_test = fig.add_subplot(num_train_pairs + 1, 2, num_train_pairs * 2 + 1)
        self._plot_grid(task['test'], ax_test)
        ax_test.set_title("Test Input")
        
        # Plot test output (solution) if available
        if solutions and task_id in solutions:
            test_output = solutions[task_id][0]  # Assuming first solution
            ax_test_output = fig.add_subplot(num_train_pairs + 1, 2, num_train_pairs * 2 + 2)
            self._plot_grid(test_output, ax_test_output)
            ax_test_output.set_title("Test Output (Solution)")
        
        plt.tight_layout()
        plt.show()
    
    def _plot_grid(self, grid: List[List[int]], ax=None):
        """Plot a grid using the ARC color scheme."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        
        grid_array = np.array(grid)
        ax.imshow(grid_array, cmap=CMAP, vmin=0, vmax=9)
        
        # Add grid lines
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
        
        # Show grid values
        for i in range(grid_array.shape[0]):
            for j in range(grid_array.shape[1]):
                ax.text(j, i, str(grid_array[i, j]), 
                       ha="center", va="center", color="w",
                       fontsize=12, fontweight='bold')
        
        # Adjust axes
        ax.set_xticks(np.arange(0, grid_array.shape[1], 1))
        ax.set_yticks(np.arange(0, grid_array.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    def analyze_dataset_stats(self, dataset: str = 'training'):
        """Analyze basic statistics of a dataset."""
        # Select the appropriate dataset
        if dataset == 'training':
            challenges = self.training_challenges
        elif dataset == 'eval':
            challenges = self.eval_challenges
        elif dataset == 'test':
            challenges = self.test_challenges
        else:
            print(f"Unknown dataset: {dataset}")
            return
        
        # Initialize statistics
        num_tasks = len(challenges)
        train_examples_counts = []
        input_grid_sizes = []
        output_grid_sizes = []
        symbols_used = defaultdict(int)
        
        # Collect statistics
        for task_id, task in challenges.items():
            train_pairs = task['train']
            train_examples_counts.append(len(train_pairs))
            
            # Analyze train pairs
            for pair in train_pairs:
                input_grid = pair['input']
                output_grid = pair['output']
                
                # Grid sizes
                input_grid_sizes.append((len(input_grid), len(input_grid[0])))
                output_grid_sizes.append((len(output_grid), len(output_grid[0])))
                
                # Symbols used
                for row in input_grid:
                    for cell in row:
                        symbols_used[cell] += 1
                for row in output_grid:
                    for cell in row:
                        symbols_used[cell] += 1
        
        # Print statistics
        print(f"\n=== {dataset.upper()} DATASET STATISTICS ===")
        print(f"Number of tasks: {num_tasks}")
        print(f"Average train examples per task: {np.mean(train_examples_counts):.2f}")
        print(f"Min train examples: {min(train_examples_counts)}, Max: {max(train_examples_counts)}")
        
        # Grid size statistics
        input_heights, input_widths = zip(*input_grid_sizes)
        output_heights, output_widths = zip(*output_grid_sizes)
        
        print(f"\nInput grid height - Avg: {np.mean(input_heights):.2f}, Min: {min(input_heights)}, Max: {max(input_heights)}")
        print(f"Input grid width - Avg: {np.mean(input_widths):.2f}, Min: {min(input_widths)}, Max: {max(input_widths)}")
        print(f"Output grid height - Avg: {np.mean(output_heights):.2f}, Min: {min(output_heights)}, Max: {max(output_heights)}")
        print(f"Output grid width - Avg: {np.mean(output_widths):.2f}, Min: {min(output_widths)}, Max: {max(output_widths)}")
        
        # Symbols used
        print("\nSymbols frequency:")
        for symbol, count in sorted(symbols_used.items()):
            print(f"Symbol {symbol}: {count} occurrences ({count/sum(symbols_used.values())*100:.2f}%)")
    
    def get_random_task_id(self, dataset: str = 'training'):
        """Return a random task ID from the specified dataset."""
        if dataset == 'training':
            return random.choice(list(self.training_challenges.keys()))
        elif dataset == 'eval':
            return random.choice(list(self.eval_challenges.keys()))
        elif dataset == 'test':
            return random.choice(list(self.test_challenges.keys()))
        else:
            print(f"Unknown dataset: {dataset}")
            return None
    
    def analyze_transformation_types(self, dataset: str = 'training', num_samples: int = 10):
        """Analyze basic transformation types between inputs and outputs."""
        # Select the appropriate dataset
        if dataset == 'training':
            challenges = self.training_challenges
        elif dataset == 'eval':
            challenges = self.eval_challenges
        else:
            print(f"Unknown dataset: {dataset}")
            return
        
        task_ids = list(challenges.keys())
        if num_samples > len(task_ids):
            num_samples = len(task_ids)
        
        sampled_tasks = random.sample(task_ids, num_samples)
        
        transformations = {
            'same_size': 0,
            'larger_output': 0,
            'smaller_output': 0,
            'color_change': 0,
            'same_colors': 0,
            'fewer_colors': 0,
            'more_colors': 0
        }
        
        for task_id in sampled_tasks:
            task = challenges[task_id]
            for pair in task['train']:
                input_grid = pair['input']
                output_grid = pair['output']
                
                # Compare sizes
                input_size = (len(input_grid), len(input_grid[0]))
                output_size = (len(output_grid), len(output_grid[0]))
                
                if input_size == output_size:
                    transformations['same_size'] += 1
                elif input_size[0] * input_size[1] < output_size[0] * output_size[1]:
                    transformations['larger_output'] += 1
                else:
                    transformations['smaller_output'] += 1
                
                # Compare colors used
                input_colors = set([cell for row in input_grid for cell in row])
                output_colors = set([cell for row in output_grid for cell in row])
                
                if input_colors == output_colors:
                    transformations['same_colors'] += 1
                elif len(input_colors) > len(output_colors):
                    transformations['fewer_colors'] += 1
                else:
                    transformations['more_colors'] += 1
                
                # Check if colors are mapped differently
                if any(c in output_colors for c in input_colors) and input_colors != output_colors:
                    transformations['color_change'] += 1
        
        print(f"\n=== TRANSFORMATION ANALYSIS ({num_samples} sampled tasks from {dataset}) ===")
        for transform_type, count in transformations.items():
            print(f"{transform_type}: {count} occurrences ({count/sum(transformations.values())*100:.2f}%)")


# Example usage
if __name__ == "__main__":
    explorer = ARCDataExplorer()
    
    # Analyze dataset statistics
    explorer.analyze_dataset_stats('training')
    explorer.analyze_dataset_stats('eval')
    
    # Analyze transformation types
    explorer.analyze_transformation_types('training', num_samples=20)
    
    # Visualize random tasks
    for dataset in ['training', 'eval']:
        task_id = explorer.get_random_task_id(dataset)
        print(f"\nVisualizing random task {task_id} from {dataset} dataset")
        explorer.visualize_task(task_id, dataset)
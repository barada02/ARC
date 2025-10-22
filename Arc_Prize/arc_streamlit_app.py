import streamlit as st
import json
import os
import numpy as np
from PIL import Image
import pandas as pd
import io
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import the visualization module
from arc_visualization import plot_grid, visualize_task, get_color_swatch

# Set page config
st.set_page_config(
    page_title="ARC Task Visualizer",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define file paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arc-prize-2025')
DATA_FILES = {
    "Training Challenges": os.path.join(DATA_DIR, 'arc-agi_training_challenges.json'),
    "Training Solutions": os.path.join(DATA_DIR, 'arc-agi_training_solutions.json'),
    "Evaluation Challenges": os.path.join(DATA_DIR, 'arc-agi_evaluation_challenges.json'),
    "Evaluation Solutions": os.path.join(DATA_DIR, 'arc-agi_evaluation_solutions.json'),
    "Test Challenges": os.path.join(DATA_DIR, 'arc-agi_test_challenges.json')
}

# Function to load data from JSON file
@st.cache_data
def load_data(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Invalid JSON file: {file_path}")
        return {}
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return {}

# Function to show grid as raw matrix
def display_raw_matrix(grid):
    df = pd.DataFrame(grid)
    
    # Style the dataframe
    def color_cells(val):
        # Set text color based on value
        colors = {
            0: 'white',
            1: 'white', 
            2: 'white',
            3: 'black',
            4: 'black',
            5: 'black',
            6: 'white',
            7: 'black',
            8: 'black',
            9: 'white'
        }
        
        # Set background color based on ARC color palette
        bg_colors = {
            0: "#010100", 
            1: "#1e93fe",
            2: "#f83d31",
            3: "#4fcc31",
            4: "#fedc00",
            5: "#999898",
            6: "#e53ba2",
            7: "#fe841b",
            8: "#87d8f1",
            9: "#921330"
        }
        
        return f'color: {colors[val]}; background-color: {bg_colors[val]}'
    
    # Apply the styling
    styled_df = df.style.applymap(color_cells)
    
    return styled_df

def main():
    # Add title
    st.title("🧩 ARC Task Visualizer")
    
    # Display color swatch
    st.sidebar.header("ARC Color Palette")
    color_swatch = get_color_swatch()
    st.sidebar.image(color_swatch, use_column_width=True)
    
    # File selector
    st.sidebar.header("Data Selection")
    selected_file = st.sidebar.selectbox(
        "Select a data file:",
        list(DATA_FILES.keys())
    )
    
    # Load selected data
    file_path = DATA_FILES[selected_file]
    data = load_data(file_path)
    
    if not data:
        st.warning("No data loaded. Please check the file path.")
        return
    
    # Task selector
    task_ids = list(data.keys())
    
    # Add search box for task IDs
    search_term = st.sidebar.text_input("Search for task ID:")
    if search_term:
        filtered_ids = [tid for tid in task_ids if search_term.lower() in tid.lower()]
        if filtered_ids:
            task_ids = filtered_ids
        else:
            st.sidebar.warning(f"No tasks matching '{search_term}'")
    
    st.sidebar.text(f"Total tasks: {len(task_ids)}")
    
    selected_task_id = st.sidebar.selectbox(
        "Select a task:",
        task_ids
    )
    
    # Get the selected task
    selected_task = data.get(selected_task_id, {})
    
    if not selected_task:
        st.warning(f"Task '{selected_task_id}' not found in the data.")
        return
    
    # Display task info
    st.header(f"Task ID: {selected_task_id}")
    
    # Visualization options
    view_options = st.radio(
        "Select View:",
        ["Grid Visualization", "Raw Matrix", "Both"],
        horizontal=True
    )
    
    # Visualize the task
    task_images = visualize_task(selected_task)
    
    # Show train examples
    st.subheader("Training Examples")
    train_examples = selected_task.get('train', [])
    if not train_examples:
        st.info("No training examples available.")
    else:
        for i, (example, images) in enumerate(zip(train_examples, task_images['train'])):
            st.markdown(f"##### Example {i+1}")
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input**")
                input_grid = np.array(example['input'])
                
                if view_options in ["Grid Visualization", "Both"]:
                    st.image(images[0], use_column_width=True)
                
                if view_options in ["Raw Matrix", "Both"]:
                    st.dataframe(display_raw_matrix(input_grid), use_container_width=True)
            
            with col2:
                st.markdown("**Output**")
                output_grid = np.array(example['output'])
                
                if view_options in ["Grid Visualization", "Both"]:
                    st.image(images[1], use_column_width=True)
                
                if view_options in ["Raw Matrix", "Both"]:
                    st.dataframe(display_raw_matrix(output_grid), use_container_width=True)
    
    # Show test examples
    st.subheader("Test Examples")
    test_examples = selected_task.get('test', [])
    if not test_examples:
        st.info("No test examples available.")
    else:
        for i, (example, images) in enumerate(zip(test_examples, task_images['test'])):
            st.markdown(f"##### Example {i+1}")
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input**")
                input_grid = np.array(example['input'])
                
                if view_options in ["Grid Visualization", "Both"]:
                    st.image(images[0], use_column_width=True)
                
                if view_options in ["Raw Matrix", "Both"]:
                    st.dataframe(display_raw_matrix(input_grid), use_container_width=True)
            
            with col2:
                st.markdown("**Output**")
                output_grid = np.array(example['output'])
                
                if view_options in ["Grid Visualization", "Both"]:
                    st.image(images[1], use_column_width=True)
                
                if view_options in ["Raw Matrix", "Both"]:
                    st.dataframe(display_raw_matrix(output_grid), use_container_width=True)
    
    # Add some information about the task structure
    with st.expander("Task JSON Structure"):
        st.json(selected_task)

if __name__ == "__main__":
    main()
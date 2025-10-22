# ARC Task Visualizer

A Streamlit application for visualizing tasks from the Abstraction and Reasoning Corpus (ARC) Prize 2025 competition.

## Features

- View ARC tasks in both grid visualization and raw matrix formats
- Browse through training and test examples
- Search for specific task IDs
- View the color palette used in ARC tasks
- Display full JSON structure of tasks

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the directory containing the application
2. Run the Streamlit app:

```bash
streamlit run arc_streamlit_app.py
```

3. The application will open in your default web browser
4. Select a data file from the sidebar
5. Choose a task ID to visualize
6. Toggle between grid visualization and raw matrix views

## Files

- `arc_streamlit_app.py`: The main Streamlit application
- `arc_visualization.py`: Module containing visualization functions
- `requirements.txt`: List of required Python packages

## ARC Color Palette

The application uses the following color palette for visualizing ARC tasks:

| Index | Color (Hex) | Description |
|-------|-------------|-------------|
| 0     | #010100     | Black       |
| 1     | #1e93fe     | Blue        |
| 2     | #f83d31     | Red         |
| 3     | #4fcc31     | Green       |
| 4     | #fedc00     | Yellow      |
| 5     | #999898     | Gray        |
| 6     | #e53ba2     | Pink        |
| 7     | #fe841b     | Orange      |
| 8     | #87d8f1     | Light Blue  |
| 9     | #921330     | Burgundy    |
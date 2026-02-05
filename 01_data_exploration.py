"""
Phase 1: Data Exploration

This script performs initial exploratory data analysis on the NASA C-MAPSS dataset.
It loads the raw training data, examines the structure, and visualizes sensor
readings over time to understand engine degradation patterns.

Usage:
    python 01_data_exploration.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_training_data():
    """
    Load NASA C-MAPSS training data and add column headers.
    
    The raw data file has no headers and contains space-separated values.
    Each row represents a single operational cycle (measurement snapshot) for an engine.
    
    Returns:
        pandas.DataFrame: Training data with proper column names
    """
    # Define column names based on dataset documentation
    # Columns: unit_id, time_cycles, 3 operational settings, 21 sensor measurements
    columns = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
    columns += [f'sensor_{i}' for i in range(1, 22)]
    
    # Load data from text file
    train_df = pd.read_csv('data/train_FD001.txt', sep=' ', header=None, 
                            names=columns, index_col=False)
    
    # Remove extra NaN columns created by trailing spaces in the file
    train_df = train_df.dropna(axis=1)
    
    return train_df


def display_basic_info(train_df):
    """
    Display basic information about the dataset.
    
    Args:
        train_df (pandas.DataFrame): Training data
    """
    print("Dataset Shape:", train_df.shape)
    print("\nFirst 5 rows:")
    print(train_df.head())
    
    print("\nDataset Info:")
    print(train_df.info())
    
    print("\nData Structure Summary:")
    print(f"Number of unique engines: {train_df['unit_id'].nunique()}")
    print(f"Total number of measurements: {len(train_df)}")
    
    # Example: Check one engine's complete lifecycle
    engine_1 = train_df[train_df['unit_id'] == 1]
    print(f"\nEngine 1 lifecycle: {engine_1['time_cycles'].max()} cycles before failure")


def visualize_sensor_degradation(train_df):
    """
    Create visualization showing how sensor readings change as engines degrade.
    
    Plots sensor readings over time for the first 4 engines to identify
    patterns and trends related to engine degradation.
    
    Args:
        train_df (pandas.DataFrame): Training data
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot sensor readings for first 4 engines
    for i, engine_id in enumerate([1, 2, 3, 4]):
        ax = axes[i//2, i%2]
        engine_data = train_df[train_df['unit_id'] == engine_id]
        
        # Plot selected sensors to show degradation patterns
        ax.plot(engine_data['time_cycles'], engine_data['sensor_2'], label='Sensor 2')
        ax.plot(engine_data['time_cycles'], engine_data['sensor_3'], label='Sensor 3')
        ax.plot(engine_data['time_cycles'], engine_data['sensor_4'], label='Sensor 4')
        ax.set_title(f'Engine {engine_id} - Sensor Readings Over Time')
        ax.set_xlabel('Time Cycles')
        ax.set_ylabel('Sensor Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/engine_degradation.png')
    plt.show()


if __name__ == "__main__":
    print("Phase 1: Data Exploration")
    print("-" * 50)
    
    # Load the training data
    train_df = load_training_data()
    
    # Display basic information
    display_basic_info(train_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_sensor_degradation(train_df)
    
    print("\nExploration complete! Check 'results/engine_degradation.png'")
    print("Next step: Run 02_data_preprocessing.py")
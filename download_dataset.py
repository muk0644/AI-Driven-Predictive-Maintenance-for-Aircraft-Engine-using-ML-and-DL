"""
NASA C-MAPSS Dataset Download Script

This script downloads the NASA Turbofan Engine Degradation Simulation dataset
from Kaggle and copies it to the local project data directory.

Dataset Source: NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
Contains sensor measurements from aircraft turbofan engines running until failure.
"""

import kagglehub
import shutil
import os


def download_and_setup_data():
    """
    Download NASA C-MAPSS dataset from Kaggle and copy to project directory.
    
    The function performs the following steps:
    1. Downloads dataset to kagglehub cache directory
    2. Creates local data folder if it doesn't exist
    3. Copies all dataset files to the project data folder
    """
    # Download dataset to kagglehub cache
    print("Downloading NASA C-MAPSS dataset from Kaggle...")
    cache_path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
    print("Downloaded to cache:", cache_path)
    
    # Set up target directory for project data
    project_data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(project_data_dir, exist_ok=True)
    
    # Copy all dataset files from cache to project directory
    print("Copying files to project data folder...")
    for item in os.listdir(cache_path):
        src = os.path.join(cache_path, item)
        dst = os.path.join(project_data_dir, item)
        
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    
    print("Dataset successfully copied to:", project_data_dir)
    print("You can now proceed with data exploration and preprocessing.")


if __name__ == "__main__":
    download_and_setup_data()

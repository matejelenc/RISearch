#!/bin/bash

# List of packages to download
packages=(
    "numpy"
    "pandas"
    "torch"
    "opencv-python"
    "scikit-learn"
    "matplotlib"
    "python-csv"
    "torchvision"
    "SimpleITK"
    "cuda-python"
)

# Loop through the list of packages and install them using pip
for package in "${packages[@]}"; do
    echo "Downloading and installing $package..."
    pip install "$package"
    echo "-----------------------------------------"
done

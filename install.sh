#!/bin/bash

# Create a Python virtual environment
python3 -m venv venv

# Check if virtual environment creation was successful
if [ $? -ne 0 ]; then
    echo "Error: Virtual environment creation failed."
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Install the library at the root
pip install .

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Veagle Library installation failed."
    exit 1
fi

# Navigate to the mPlug-owl-2 directory
cd mPLUG-Owl2

# Install the setup.py in the mPlug-owl-2 directory
pip install .

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Installation of setup.py in mPlug-owl-2 directory failed."
    exit 1
fi

# get back to the root directory 
cd ..

# install the 4.35.0 version of transformers
pip install transformers==4.35.0

# download all the models
python download_hf.py

# deactivate the environment
deactivate
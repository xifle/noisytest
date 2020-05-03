#!/usr/bin/env bash
# Setup a virtual environment for noisytest

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools

# Install requirements
pip install -r requirements.txt

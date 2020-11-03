#!/usr/bin/env bash
# Setup a virtual environment for development / test
# of noisytest

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools

#pip install -r requirements.txt

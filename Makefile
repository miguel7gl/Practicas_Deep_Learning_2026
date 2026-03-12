# Makefile
SHELL = /bin/bash

# VARIABLES
VENV_NAME:=dl_lab
PYTHON_VERSION:=3.12

# ---------------------------------------------------
# COMMANDS
# ---------------------------------------------------
# create local python environment
venv:
	pip install uv && \
	uv venv --python=${PYTHON_VERSION} ${VENV_NAME} && \
    source ${VENV_NAME}/bin/activate && \
	uv pip install --upgrade pip && \
	pip install -r requirements.txt

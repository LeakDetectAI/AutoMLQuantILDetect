# Project Variables
NAME := AutoMLQuantILDetect
PACKAGE_NAME := autoqild

# Directories
DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DOCDIR := docs
EXAMPLES_DIR := examples
# Output files
INDEX_HTML := "file://${DIR}/docs/build/html/index.html"

# Command Aliases
PYTHON ?= python
PYTEST ?= python -m pytest
PIP ?= python -m pip
MAKE ?= make

.PHONY: help install clean docs examples

help:
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements"
	@echo "* clean            to clean any doc or build files"
	@echo "* docs             to generate and view the html files"
# Installation and Setup
install:
	@export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
	@sudo apt-get install pandoc
	$(PIP) install -e "./"

# Documentation
clean-doc:
	$(MAKE) -C ${DOCDIR} clean

docs:
	$(MAKE) -C ${DOCDIR} html
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

# Clean up any builds
clean: clean-doc
	rm -rf dist build *.egg-info

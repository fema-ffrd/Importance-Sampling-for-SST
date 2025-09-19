# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

import SSTImportanceSampling

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SSTImportanceSampling'
author = ""
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx.ext.doctest",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_logo = "_static/sst_importance_sampling_logo.png"

# Substitutions
release = str(SSTImportanceSampling.__version__)
version = release

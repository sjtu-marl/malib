# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import inspect
import shutil


__location__ = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

sys.path.insert(0, __location__)


# -- Project information -----------------------------------------------------

project = "MALib"
copyright = "2021, SJTU-MARL"
author = "SJTU-MARL"

# The full version, including alpha/beta/rc tags
release = "v0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # add napoleon to extension list to support numpy and google-style docstring
    "sphinx_rtd_theme",  # theme
    "sphinx.ext.autodoc",  # automatically extract docs from docstrings
    "sphinx.ext.coverage",  # make coverage generates documentation coverage reports
    "sphinx.ext.viewcode",  # link to sourcecode from docs
    # "sphinx.ext.grahviz", # graphviz
    "sphinxcontrib.apidoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    # "sphinxcontrib.bibtex",
]

apidoc_module_dir = os.path.join(__location__, "malib")
# apidoc_exclude_paths = ["setup.py"]
apidoc_module_first = True
# apidoc_extra_args = [
#     "--force",
#     "--separate",
#     "--ext-viewcode",
#     "--doc-project=MALib",
#     "--maxdepth=2",
# ]

exclude_patterns = ["tests/", "test*.py"]

# ensure __init__ is always documented
autoclass_content = "both"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

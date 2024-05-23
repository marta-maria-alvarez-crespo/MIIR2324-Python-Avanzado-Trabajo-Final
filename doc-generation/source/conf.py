# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

relative_path = "scripts"
absolute_path = os.path.abspath(relative_path)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Entrenamiento de RN para la identificacion de galaxias"
copyright = "2024, Marta M. Álvarez Crespo"
author = "Marta M. Álvarez Crespo"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ["sphinx.ext.autodoc", "sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autosummary"]

todo_include_todos = True  # Show TODOs in the documentation

templates_path = ["_templates"]
exclude_patterns = []

language = "[es]"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "press"
html_static_path = ["_static"]

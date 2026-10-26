"""Sphinx configuration for the public neural_ode API."""

from importlib.metadata import version as package_version
import os


# Autodoc imports plotting modules; use a backend that needs no display.
os.environ["MPLBACKEND"] = "Agg"

project = "neural_ode"
release = package_version("neural-ode")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build", ".DS_Store"]

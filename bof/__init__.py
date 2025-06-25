"""Top-level package for Bag of Factors."""

from importlib.metadata import metadata
import numpy as np

np.set_printoptions(legacy="1.25")

infos = metadata(__name__)
__version__ = infos["Version"]
__author__ = """Fabien Mathieu"""
__email__ = "loufab@gmail.com"

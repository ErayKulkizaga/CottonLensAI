"""Colab-only training package. Never install this package in the runtime image."""

import os

# Set before importing Keras, TensorFlow, SHAP or pyplot, including console scripts.
os.environ["MPLBACKEND"] = "Agg"
os.environ["KERAS_BACKEND"] = "tensorflow"

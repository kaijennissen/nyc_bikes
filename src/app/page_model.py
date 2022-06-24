import glob
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay


def from_pickle(filename: str):
    with open(f"images/{filename}.pickle", "rb") as f:
        fig = pickle.load(f)
    return fig


def app():
    labels = ["Customer", "Subscriber"]
    with st.expander("LogisticRegression"):
        image = Image.open("images/LogisticRegression.png")
        st.image(image, caption="Enter any caption here")

        st.markdown("Accuracy: 0.94")
        st.markdown("F1-Score: 0.9667405764966741")

    with st.expander("XGBoost"):
        image = Image.open("images/XGBoost.png")
        st.image(image, caption="Enter any caption here")
        st.markdown("Accuracy: 0.94")
        st.markdown("F1-Score: 0.9667405764966741")

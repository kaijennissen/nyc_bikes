import glob
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def from_pickle(filename: str):
    with open(f"images/{filename}.pickle", "rb") as f:
        fig = pickle.load(f)
    return fig


def app():

    st.markdown("# Features")
    st.markdown(
        """
    - **Features basierend auf der Zeit:**
        - Wochentag
        - Uhrzeit (Stunde)
        - Monat
        - Wochenende
        - Feiertag
        - bezahlter Feiertag
        - *Ferien*
    - **Features basierend auf der Wetter:**
        - Temperatur
        - Luftfeuchtigkeit
        - Windgeschwindigkeit
        - Windrichtung
        - Regenmenge
        - Schneemenge
        - Sonnenscheinmenge
        - *Sonnenaufgang*
        - *Sonnenuntergang*
        - Wetterkennung (Regen, Schnee, Sonne, Wolken, Nebel, Sturm)
    - **Features basierend auf Start bzw. Zielstation:**
        - Stadtbezirke (Manhattan, Queens, Brooklyn)
            - OneHot oder TargetEncoder
        - Viertel (Soho, Hellmannsdorf, West Village, ...)
            - OneHot oder TargetEncoder
        - PLZ
        - *Touristenattraktion in der Nähe*
        -
    - **Features basierend auf der Strecke (Google bzw OpenMaps API):**
        - Distanz mit dem Fahrrad zwischen Start- und Zielstation
        - Fahrtzeit mit dem Fahrrad zwischen Start- und Zielstation
        - Rundtrip (Start=Zielstation)
    """
    )

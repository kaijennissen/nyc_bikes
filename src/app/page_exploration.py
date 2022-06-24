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
    st.markdown("# Data Exploration")
    with st.expander("Plot 1"):
        fig1 = from_pickle("plot1")
        fig1.update_layout(height=1000)
        st.plotly_chart(fig1, use_container_width=True, height=1000)

    with st.expander("Plot 2"):
        fig2 = from_pickle("plot2")
        fig2.update_layout(height=1000)
        st.plotly_chart(fig2, use_container_width=True, height=1000)

    with st.expander("Plot 3"):
        fig3 = from_pickle("plot3")
        fig3.update_layout(height=1000)
        st.plotly_chart(fig3, use_container_width=True, height=1000)

    with st.expander("Plot 4"):
        fig4 = from_pickle("plot4")
        fig4.update_layout(height=1000)
        st.plotly_chart(fig4, use_container_width=True, height=1000)
    with st.expander("Plot 5"):
        fig5 = from_pickle("plot5")
        fig5.update_layout(height=1000)
        st.plotly_chart(fig5, use_container_width=True, height=1000)

    with st.expander("Plot 6"):
        fig6 = from_pickle("plot6")
        fig6.update_layout(height=1000)
        st.plotly_chart(fig6, use_container_width=True, height=1000)

    with st.expander("Plot 7"):
        fig7 = from_pickle("plot7")
        fig7.update_layout(height=1000)
        st.plotly_chart(fig7, use_container_width=True, height=1000)

    with st.expander("Plot 8"):
        fig8 = from_pickle("plot8")
        fig8.update_layout(height=1000)
        st.plotly_chart(fig8, use_container_width=True, height=1000)

    with st.expander("Plot 9"):
        fig9 = from_pickle("plot9")
        fig9.update_layout(height=1000)
        st.plotly_chart(fig9, use_container_width=True, height=1000)

    with st.expander("Plot 10"):
        fig10 = from_pickle("plot10")
        fig10.update_layout(height=1000)
        st.plotly_chart(fig10, use_container_width=True, height=1000)

    with st.expander("Plot 11"):
        fig11 = from_pickle("plot11")
        fig11.update_layout(height=1000)
        st.plotly_chart(fig11, use_container_width=True, height=1000)

    with st.expander("Plot 12"):
        fig12 = from_pickle("plot12")
        fig12.update_layout(height=1000)
        st.plotly_chart(fig12, use_container_width=True, height=1000)

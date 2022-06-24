import page_exploration
import page_features
import page_model
import streamlit as st

# Custom imports
from multipage import MultiPage

st.set_page_config(layout="wide")
# Create an instance of the app
app = MultiPage()


# Add all your applications (pages) here
app.add_page("Model", page_model.app)
app.add_page("Features", page_features.app)
app.add_page("Data Exploration", page_exploration.app)

# The main app
app.run()

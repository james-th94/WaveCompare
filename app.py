# %% Code description
"""Streamlit app to show wave data. Deployed via GitHub repository: https://github.com/james-th94/WaveCompare/

Created 26 June 2025, JT.
"""

# %% Python setup
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# %% User inputs
wavedata_file = "data/Grassy_-40.1250_144.0625_19870101_20231231_WHACS.csv"
# date parser for the wavedata file "datetime" format
custom_date_parser = lambda x: pd.to_datetime(
    x, format="%Y-%m-%d %H:%M:%S", errors="coerce"
)

datetime_col = "datetime"
direction_col = "dp"
direction_resolution = 22.5  # degrees
wave_col = "cge"
wave_bins = np.arange(0, 10, 1)  # Wave "height/power" bins for waverose


# %% Load data
@st.cache_data
def load_data(filename, datetime_column=datetime_col):
    df = pd.read_csv(
        wavedata_file,
        index_col=datetime_col,
        parse_dates=[datetime_col],
        date_parser=custom_date_parser,
    )
    df["year"] = df.index.year
    return df


df = load_data(wavedata_file)

# %% Process data and create waverose
# Sidebar - Year selection
years = sorted(df["year"].unique())
selected_year = st.slider(
    "Select year",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=int(min(years)),
)

# Filter data
df_year = df[df["year"] == selected_year]

# Compute wave rose bins
dir_bins = np.arange(0, 361, direction_resolution)

df_year["dir_bin"] = pd.cut(
    df_year[direction_col], bins=dir_bins, right=False, labels=dir_bins[:-1]
)
df_year["wave_bin"] = pd.cut(df_year[wave_col], bins=wave_bins, right=False)

# Count occurrences
rose_data = df_year.groupby(["dir_bin", "wave_bin"]).size().reset_index(name="counts")
rose_data["dir_bin"] = rose_data["dir_bin"].astype(float)
rose_data["wave_bin"] = rose_data["wave_bin"].astype(str)

# Plot polar wave rose
fig = px.bar_polar(
    rose_data,
    r="counts",
    theta="dir_bin",
    color="wave_bin",
    color_discrete_sequence=px.colors.sequential.Plasma_r,
    title=f"Wave Rose - {selected_year}",
)

# %% Deploy chart with streamlit
st.plotly_chart(fig, use_container_width=True)

# %%

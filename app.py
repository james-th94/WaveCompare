# %% Code description
"""Streamlit app to show wave data. Deployed via GitHub repository: https://github.com/james-th94/WaveCompare/

Created 26 June 2025, JT.
"""

# %% Python setup
"""Make sure these packages are in the requirements.txt file for Streamlit to work."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# %% User inputs
wavedata_file = "data/Grassy_-40.1250_144.0625_19870101_20231231_WHACS.csv"
# date parser for the wavedata file "datetime" format
custom_date_parser = lambda x: pd.to_datetime(
    x, format="%Y-%m-%d %H:%M:%S", errors="coerce"
)

# Set variables to plot
datetime_col = "datetime"
direction_col = "dp"
direction_resolution = 22.5  # degrees
direction_name = "Directional bin (\u00b0)"
wave_col = "cge"
wave_bins = np.arange(0, 120, 10)  # Wave "height/power" bins for waverose
wave_name = "Wave energy flux (kW/m)"
wave_colours = px.colors.sequential.Plasma_r
wave2_col = "hs"
wave2_bins = np.arange(0, 8, 0.5)  # Wave "height/power" bins for waverose
wave2_name = "Significant wave height (m)"
wave2_colours = px.colors.sequential.Plasma_r
wave3_col = "Tp"
wave3_name = "Peak wave period (Tp)"
# For wave roses:
freq_name = "Relative frequency (%)"


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

# %% Timeseries multi-plot
# Create 3 stacked subplots
fig_ts = go.Figure()

# Subplot 1
fig_ts.add_trace(go.Scatter(x=df.index, y=df[wave_col], name=wave_name, yaxis="y1"))

# Subplot 2
fig_ts.add_trace(go.Scatter(x=df.index, y=df[wave2_col], name=wave2_name, yaxis="y2"))

# Subplot 3
fig_ts.add_trace(go.Scatter(x=df.index, y=df[wave3_col], name=wave3_name, yaxis="y3"))

# Subplot 4 - Direction
fig_ts.add_trace(
    go.Scatter(x=df.index, y=df[direction_col], name=direction_name, yaxis="y4")
)

fig_ts.update_layout(
    height=800,
    margin=dict(t=30, b=50),
    xaxis=dict(domain=[0.05, 0.95]),
    yaxis=dict(title=wave_name, anchor="x", domain=[0.77, 0.98]),
    yaxis2=dict(title=wave2_name, anchor="x", domain=[0.52, 0.73]),
    yaxis3=dict(title=wave3_name, anchor="x", domain=[0.27, 0.48]),
    yaxis4=dict(title=direction_name, anchor="x", domain=[0.05, 0.23]),
    showlegend=False,
)
st.plotly_chart(fig_ts, use_container_width=True)

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
df_year[direction_name] = pd.cut(
    df_year[direction_col], bins=dir_bins, right=False, labels=dir_bins[:-1]
)

df_year[wave_name] = pd.cut(df_year[wave_col], bins=wave_bins, right=False)
df_year[wave2_name] = pd.cut(df_year[wave2_col], bins=wave2_bins, right=False)

# %% Wave rose 1
rose_data = (
    df_year.groupby([direction_name, wave_name]).size().reset_index(name="counts")
)
rose_data[freq_name] = round(100 * (rose_data["counts"] / rose_data["counts"].sum()), 2)
rose_data[direction_name] = rose_data[direction_name].astype(float)
rose_data[wave_name] = rose_data[wave_name].astype(str)

# Plot polar wave rose
fig = px.bar_polar(
    rose_data,
    r=freq_name,
    theta=direction_name,
    color=wave_name,
    color_discrete_sequence=wave_colours,
    title=f"Wave Rose - {selected_year}",
)

# %% Wave rose 2
rose2_data = (
    df_year.groupby([direction_name, wave2_name]).size().reset_index(name="counts")
)
rose2_data[freq_name] = round(
    100 * (rose2_data["counts"] / rose2_data["counts"].sum()), 2
)
rose2_data[direction_name] = rose2_data[direction_name].astype(float)
rose2_data[wave2_name] = rose2_data[wave2_name].astype(str)

# Plot polar wave rose
fig2 = px.bar_polar(
    rose2_data,
    r=freq_name,
    theta=direction_name,
    color=wave2_name,
    color_discrete_sequence=wave2_colours,
    title=f"Wave Rose - {selected_year}",
)

# %% Deploy chart with streamlit
tab1, tab2 = st.tabs([wave_name, wave2_name])
with tab1:
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.plotly_chart(fig2, use_container_width=True)

# %%

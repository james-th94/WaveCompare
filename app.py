# %% Code description
"""Streamlit app to show wave data.
GitHub repository: https://github.com/james-th94/WaveCompare/
App: https://wavecompare.streamlit.app/
Created 26 June 2025, JT.
"""

# %% Python setup
# """Make sure these packages are in the requirements.txt file for Streamlit to work."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def parse_two_formats(datetime_series):
    # Try fast parsing with format 1
    parsed = pd.to_datetime(
        datetime_series, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    # Fill NaT values using second format
    mask = parsed.isna()
    parsed[mask] = pd.to_datetime(
        datetime_series[mask], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    ).dt.round("s")

    return parsed


# %% User inputs
# Input file
wavedata_file = "data/wave_data.csv"

# Set variables to plot
datetime_col = "datetime"
direction_col = "dp"
direction_resolution = 22.5  # degrees
direction_name = "Peak wave direction (\u00b0)"
direction_rosename = "Directional bin (\u00b0)"
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

# Event dates
event_dates_start = "10/01/2021"
event_dates_end = "28/03/2023"


# %% Load data & cache to streamlit app
@st.cache_data
def load_data(filename, datetime_column=datetime_col):
    # Load data from csv
    df = pd.read_csv(
        filename,
    )
    # Fix datetimes and set as index
    df[datetime_column] = parse_two_formats(datetime_series=df[datetime_column])
    df.index = df[datetime_column]
    df = df.drop(columns=[datetime_column])
    # Add Year column
    df["year"] = df.index.year
    return df


# Load data to dataframe (df) for data analysis
df = load_data(filename=wavedata_file)

# %% Add header to Streamlit webpage
st.header("Wave Data", divider="grey")

# %% Timeseries multi-plot - 3 stacked subplots
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
    xaxis4=dict(domain=[0.05, 0.95]),
    yaxis=dict(title=wave_name, anchor="x", domain=[0.77, 0.98]),
    yaxis2=dict(title=wave2_name, anchor="x", domain=[0.52, 0.73]),
    yaxis3=dict(title=wave3_name, anchor="x", domain=[0.27, 0.48]),
    yaxis4=dict(title=direction_name, anchor="x", domain=[0.05, 0.23]),
    showlegend=False,
    title="Wave Timeseries (all data)",
)
st.plotly_chart(fig_ts, use_container_width=True)

# %% Process data and create waverose
# Sidebar - Year selection
years = sorted(df["year"].unique())
st.subheader("Yearly wave roses")
selected_year = st.slider(
    "Select year",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=int(min(years)),
)

# Filter data
df_year = df[df["year"] == selected_year].copy()

# Compute wave rose bins
dir_bins = np.arange(0, 361, direction_resolution)
df_year.loc[:, direction_rosename] = pd.cut(
    df_year[direction_col], bins=dir_bins, right=False, labels=dir_bins[:-1]
)

df_year.loc[:, wave_name] = pd.cut(df_year[wave_col], bins=wave_bins, right=False)
df_year.loc[:, wave2_name] = pd.cut(df_year[wave2_col], bins=wave2_bins, right=False)

# %% Wave rose 1
rose_data = (
    df_year.groupby([direction_rosename, wave_name], observed=False)
    .size()
    .reset_index(name="counts")
)
rose_data[freq_name] = round(100 * (rose_data["counts"] / rose_data["counts"].sum()), 2)
rose_data[direction_rosename] = rose_data[direction_rosename].astype(float)
rose_data[wave_name] = rose_data[wave_name].astype(str)

# Plot polar wave rose
fig = px.bar_polar(
    rose_data,
    r=freq_name,
    theta=direction_rosename,
    color=wave_name,
    color_discrete_sequence=wave_colours,
)

# %% Wave rose 2
rose2_data = (
    df_year.groupby([direction_rosename, wave2_name], observed=False)
    .size()
    .reset_index(name="counts")
)
rose2_data[freq_name] = round(
    100 * (rose2_data["counts"] / rose2_data["counts"].sum()), 2
)
rose2_data[direction_rosename] = rose2_data[direction_rosename].astype(float)
rose2_data[wave2_name] = rose2_data[wave2_name].astype(str)

# Plot polar wave rose
fig2 = px.bar_polar(
    rose2_data,
    r=freq_name,
    theta=direction_rosename,
    color=wave2_name,
    color_discrete_sequence=wave2_colours,
)

# %% Wave rose 3
df_during = df[(df.index >= event_dates_start) & (df.index <= event_dates_end)].copy()
df_pre = df[df.index < event_dates_start].copy()

figs = {}
for idx, df in enumerate([df_pre, df_during]):
    # Filter data
    df_year = df[df["year"] == selected_year].copy()

    # Compute wave rose bins
    dir_bins = np.arange(0, 361, direction_resolution)
    df_year.loc[:, direction_rosename] = pd.cut(
        df_year[direction_col], bins=dir_bins, right=False, labels=dir_bins[:-1]
    )

    df_year.loc[:, wave_name] = pd.cut(df_year[wave_col], bins=wave_bins, right=False)

    rose_data = (
        df_year.groupby([direction_rosename, wave_name], observed=False)
        .size()
        .reset_index(name="counts")
    )
    rose_data[freq_name] = round(
        100 * (rose_data["counts"] / rose_data["counts"].sum()), 2
    )
    rose_data[direction_rosename] = rose_data[direction_rosename].astype(float)
    rose_data[wave_name] = rose_data[wave_name].astype(str)

    # Plot polar wave rose
    figs[idx] = px.bar_polar(
        rose_data,
        r=freq_name,
        theta=direction_rosename,
        color=wave_name,
        color_discrete_sequence=wave_colours,
    )

# %% Deploy chart with streamlit
tab1, tab2, tab3, tab4 = st.tabs(
    [wave_name, wave2_name, f"Pre-event: {wave_name}", f"During-event: {wave_name}"]
)
with tab1:
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.plotly_chart(fig2, use_container_width=True)
with tab3:
    st.plotly_chart(figs[0], use_container_width=True)
with tab4:
    st.plotly_chart(figs[1], use_container_width=True)

# %% THE END

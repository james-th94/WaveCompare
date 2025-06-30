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

# %% User inputs
# User inputs - set these in the app.py file
# Input file - observation (obs) data
obs_wavefile = "data/brissy_data.csv"
# Columns of interest - observation data
obs_datetime_col = "Date/Time (AEST)"
obs_timezone = "Australia/Brisbane"
obs_direction_col = "Peak Direction (degrees)"
obs_wave_col = "Hs (m)"
obs_wave2_col = "Tp (s)"
obs_na_value = -99.9000

# Input file - model data
model_wavefile = "data/brissy_model_data.csv"
# Columns of interest - model data
model_datetime_col = "datetime"
model_direction_col = "dp"
model_wave_col = "hs"
model_wave2_col = "Tp"
model_na_value = -9999

# Set common variables
datetime_col = "Datetime (UTC)"
timestep = "h"  # Choose minimum step for resampling data: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
slider_timestep = "Month"

# Set variables to plot
# Wave direction
direction_resolution = 22.5  # degrees
direction_name = "Peak wave direction (\u00b0)"
direction_rosename = "Directional bin (\u00b0)"
# Wave "height"
wave_col = "hs"
wave_bins = np.arange(0, 6, 0.5)
wave_name = "Significant wave height (m)"
wave_colours = px.colors.sequential.Plasma_r
# Wave period
wave2_col = "Tp"
wave2_name = "Peak wave period (Tp)"
# For wave roses:
freq_name = "Relative frequency (%)"

# %% Create functions
# Create functions


def parse_obs_dates(datetime_series):
    # Try fast parsing with format 1
    parsed = (
        pd.to_datetime(datetime_series, format="%Y-%m-%dT%H:%M", errors="coerce")
        .dt.tz_localize(obs_timezone)
        .dt.tz_convert("UTC")
    )

    return parsed


def parse_model_dates(datetime_series):
    # Try fast parsing with format 1
    parsed = pd.to_datetime(
        datetime_series, format="%Y-%m-%d %H:%M:%S", utc="True", errors="coerce"
    )

    # Fill NaT values using second format
    mask = parsed.isna()
    parsed[mask] = pd.to_datetime(
        datetime_series[mask],
        format="%Y-%m-%d %H:%M:%S.%f",
        utc="True",
        errors="coerce",
    ).dt.round("s")

    return parsed


def load_obs_data(filename, datetime_column=obs_datetime_col):
    # Load data from csv
    df = pd.read_csv(
        filename,
    )
    # Fix datetimes and set as index
    df[datetime_col] = parse_obs_dates(datetime_series=df[datetime_column])
    df.index = df[datetime_col]
    df = df.drop(columns=[datetime_col, datetime_column])  # Drop datetime columns
    df = df.resample(timestep).mean()  # Resample to timestep of interest
    df.replace(obs_na_value, np.nan, inplace=True)  # Replace na values
    df.loc[df[obs_wave_col] < 0] = np.nan  # Replace rows with negative wave data
    # Rename columns for plotting
    df[wave_name] = df[obs_wave_col]
    df[wave2_name] = df[obs_wave2_col]
    df[direction_name] = df[obs_direction_col]
    # Save only the columns of interest
    df = df[[wave_name, wave2_name, direction_name]].copy(deep=True)
    return df


def load_model_data(filename, datetime_column=model_datetime_col):
    # Load data from csv
    df = pd.read_csv(
        filename,
    )
    # Fix datetimes and set as index
    df[datetime_col] = parse_model_dates(datetime_series=df[datetime_column])
    df.index = df[datetime_col]
    df = df.drop(columns=[datetime_col, datetime_column])  # Drop datetime columns
    df = df.resample(timestep).mean()  # Resample to timestep of interest
    df.replace(model_na_value, np.nan, inplace=True)  # Replace na values
    df.loc[df[model_wave_col] < 0] = np.nan  # Replace rows with negative wave data
    # Rename columns for plotting
    df[wave_name] = df[model_wave_col]
    df[wave2_name] = df[model_wave2_col]
    df[direction_name] = df[model_direction_col]
    # Save only the columns of interest
    df = df[[wave_name, wave2_name, direction_name]].copy(deep=True)
    return df


# %% Load data & cache to streamlit app
# Load data
# @st.cache_data
def merge_wave_data(df1, df2):
    # Merge dataframes
    df = df1.join(df2, how="inner", lsuffix="_obs", rsuffix="_model")
    # Add options for slider timestep
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    return df


# Load data to dataframe (df) for data analysis
df_obs = load_obs_data(filename=obs_wavefile)
df_model = load_model_data(filename=model_wavefile)
df = merge_wave_data(df_obs, df_model)

# %% Add header to Streamlit webpage
st.header("Wave Data", divider="grey")

# %% Timeseries multi-plot - 3 stacked subplots
fig_ts = go.Figure()
for idx, column in enumerate([wave_name, wave2_name, direction_name]):
    axis_id = (
        f"y{idx+1}" if idx > 0 else "y"
    )  # "y" for primary axis, "y2", "y3", etc. for others

    fig_ts.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"{column}_obs"],
            name=f"{column} (obs)",
            yaxis=axis_id,
            line=dict(color="black"),
        )
    )

    fig_ts.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"{column}_model"],
            name=f"{column} (model)",
            yaxis=axis_id,
            line=dict(color="blue"),
        )
    )

# # Subplot 1 - Wave "height"
# fig_ts.add_trace(go.Scatter(
#     x=df.index,
#     y=df[f"{wave_name}_obs"],
#     name=f"{wave_name} (obs)",
#     yaxis="y1",
#     line=dict(color="black")
# ))

# fig_ts.add_trace(go.Scatter(
#     x=df.index,
#     y=df[f"{wave_name}_model"],
#     name=f"{wave_name} (model)",
#     yaxis="y1",
#     line=dict(color="blue")
# ))

# # Subplot 2 - Wave period
# fig_ts.add_trace(go.Scatter(
#     x=df.index,
#     y=df[f"{wave2_name}_obs"],
#     name=f"{wave2_name} (obs)",
#     yaxis="y2",
#     line=dict(color="black")
# ))

# fig_ts.add_trace(go.Scatter(
#     x=df.index,
#     y=df[f"{wave2_name}_model"],
#     name=f"{wave2_name} (model)",
#     yaxis="y2",
#     line=dict(color="blue")
# ))

# # Subplot 3 - Direction
# fig_ts.add_trace(go.Scatter(
#     x=df.index,
#     y=df[f"{direction_name}_obs"],
#     name=f"{direction_name} (obs)",
#     yaxis="y3",
#     line=dict(color="black")
# ))

# fig_ts.add_trace(go.Scatter(
#     x=df.index,
#     y=df[f"{direction_name}_model"],
#     name=f"{direction_name} (model)",
#     yaxis="y3",
#     line=dict(color="blue")
# ))

fig_ts.update_layout(
    height=800,
    margin=dict(t=30, b=50),
    xaxis3=dict(domain=[0.05, 0.95]),
    yaxis=dict(title=wave_name, anchor="x", domain=[0.7, 0.95]),
    yaxis2=dict(title=wave2_name, anchor="x", domain=[0.37, 0.62]),
    yaxis3=dict(title=direction_name, anchor="x", domain=[0.05, 0.3]),
    showlegend=False,
    title="Wave Timeseries (all data)",
)

st.plotly_chart(fig_ts, use_container_width=True)

# # %% Process data and create waverose
# # Sidebar - Year selection
# years = sorted(df["year"].unique())
# st.subheader("Yearly wave roses")
# selected_year = st.slider(
#     "Select year",
#     min_value=int(min(years)),
#     max_value=int(max(years)),
#     value=int(min(years)),
# )

# # Filter data
# df_year = df[df["year"] == selected_year].copy()

# # Compute wave rose bins
# dir_bins = np.arange(0, 361, direction_resolution)
# df_year.loc[:, direction_rosename] = pd.cut(
#     df_year[direction_col], bins=dir_bins, right=False, labels=dir_bins[:-1]
# )

# df_year.loc[:, wave_name] = pd.cut(df_year[wave_col], bins=wave_bins, right=False)
# df_year.loc[:, wave2_name] = pd.cut(df_year[wave2_col], bins=wave2_bins, right=False)

# # %% Wave rose 1
# rose_data = (
#     df_year.groupby([direction_rosename, wave_name], observed=False)
#     .size()
#     .reset_index(name="counts")
# )
# rose_data[freq_name] = round(100 * (rose_data["counts"] / rose_data["counts"].sum()), 2)
# rose_data[direction_rosename] = rose_data[direction_rosename].astype(float)
# rose_data[wave_name] = rose_data[wave_name].astype(str)

# # Plot polar wave rose
# fig = px.bar_polar(
#     rose_data,
#     r=freq_name,
#     theta=direction_rosename,
#     color=wave_name,
#     color_discrete_sequence=wave_colours,
# )

# # %% Wave rose 2
# rose2_data = (
#     df_year.groupby([direction_rosename, wave2_name], observed=False)
#     .size()
#     .reset_index(name="counts")
# )
# rose2_data[freq_name] = round(
#     100 * (rose2_data["counts"] / rose2_data["counts"].sum()), 2
# )
# rose2_data[direction_rosename] = rose2_data[direction_rosename].astype(float)
# rose2_data[wave2_name] = rose2_data[wave2_name].astype(str)

# # Plot polar wave rose
# fig2 = px.bar_polar(
#     rose2_data,
#     r=freq_name,
#     theta=direction_rosename,
#     color=wave2_name,
#     color_discrete_sequence=wave2_colours,
# )

# # %% Deploy charts with streamlit
# tab1, tab2 = st.tabs([wave_name, wave2_name])
# with tab1:
#     st.plotly_chart(fig, use_container_width=True)
# with tab2:
#     st.plotly_chart(fig2, use_container_width=True)


# # %% THE END

# %%

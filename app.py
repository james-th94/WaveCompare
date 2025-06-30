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

st.cache_data.clear()

# %% User inputs
# User inputs - set these in the app.py file
page_title = "Wave data comparison"
suffixes = ["obs", "model"]
# Input file - observation (obs) data
obs_wavefile = "data/brissy_data.csv"
sitename = "Brisbane"
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

# Set variables to plot
# Wave direction
direction_resolution = 22.5  # degrees
direction_name = "Peak wave direction (\u00b0)"
direction_rosename = "Directional bin (\u00b0)"
# Wave "height"
wave_col = "hs"
wave_bins = np.arange(0, 5.5, 0.5)
wave_name = "Significant wave height (m)"
wave_colours = px.colors.sequential.Plasma_r
# Wave period
wave2_col = "Tp"
wave2_name = "Peak wave period (Tp)"
# For timeseries
timeseries_title = f"{sitename}: Model (blue) vs observed (black)"
# For wave roses:
freq_name = "Relative frequency (%)"
waverose_names = ["Observations", "Modelled"]


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


def month2season(datetime_series):
    months = datetime_series.month
    seasons = {
        12: "Summer",
        1: "Summer",
        2: "Summer",
        3: "Autumn",
        4: "Autumn",
        5: "Autumn",
        6: "Winter",
        7: "Winter",
        8: "Winter",
        9: "Spring",
        10: "Spring",
        11: "Spring",
    }
    season = months.map(seasons)
    return season


season2seasonValue = {"Summer": 1, "Autumn": 2, "Winter": 3, "Spring": 4}
seasonValue2season = {v: k for k, v in season2seasonValue.items()}


# Wave statistics based on Wave Information Studies Model Evaluation Statistics (Bryant et al. 2016)
def waveStats(predictions, targets):
    if len(predictions) != len(targets):
        return print("Failed: Length of predictions and targets variables differ.")
    else:
        residuals = predictions - targets
        SSresiduals = np.sum(np.square(residuals))
        SStotal = np.sum(np.square(np.mean(targets) - targets))
        R_2 = 1 - np.divide(
            SSresiduals, SStotal
        )  # Coefficient of determination (-inf to 1)
        N = len(targets)  # number of samples
        bias = np.divide(np.sum(residuals), N)  # bias in predictions
        temp = np.sum(np.square(residuals - bias))
        rmse_demeaned = np.sqrt(
            np.divide(temp, N - 1)
        )  # a measure of the residuals between the predictions and observations with bias removed
        rmse = np.sqrt(((residuals) ** 2).mean())
        SI = np.divide(
            rmse_demeaned, np.mean(targets)
        )  # scatter index like Bryant et al. (2016)
        SI_2 = np.divide(rmse, np.mean(targets))  # scatter index like Ris et al. (1999)
        temp = np.sum([np.abs(predictions[i] - targets[i]) for i in range(N)])
        MAE = np.divide(temp, N)
        temp = np.sum(
            [np.abs((predictions[i] - targets[i]) / targets[i]) for i in range(N)]
        )
        MAPE = np.divide(temp, N)
        data = {
            "Bias": np.round(bias, 4),
            "Root Mean Sqare Error (RMSE)": np.round(rmse, 4),
            "Scatter Index or Normalised RMSE": np.round(SI_2, 4),
            "Coefficient of Determination, R_2": np.round(R_2, 4),
            "Mean Average Error": np.round(MAE, 4),
            "Mean Absolute Percentage Error": np.round(MAPE, 4),
            "RMSE (Bryant et al. 2016)": np.round(rmse_demeaned, 4),
            "Scatter Index (Bryant et al. 2016)": np.round(SI, 4),
        }
        return data


# %% Load data & cache to streamlit app
# Load data
@st.cache_data
def merge_wave_data(df1, df2):
    # Merge dataframes
    df = df1.join(
        df2, how="inner", lsuffix=f"_{suffixes[0]}", rsuffix=f"_{suffixes[1]}"
    )
    # Add options for slider timestep
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Season"] = month2season(df.index)
    df["Season_val"] = df["Season"].map(season2seasonValue)
    return df


# Load data to dataframe (df) for data analysis
df_obs = load_obs_data(filename=obs_wavefile)
df_model = load_model_data(filename=model_wavefile)
df = merge_wave_data(df_obs, df_model)

# %% Add header to Streamlit webpage
st.header(page_title, divider="grey")

# %% Timeseries multi-plot - stacked subplots
fig_ts = go.Figure()
for idx, column in enumerate([wave_name, wave2_name, direction_name]):
    yaxis_id = (
        f"y{idx+1}" if idx > 0 else "y"
    )  # "y" for primary axis, "y2", "y3", etc. for others
    xaxis_id = f"x{idx+1}" if idx > 0 else "x"
    # Observation data
    fig_ts.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"{column}_obs"],
            name=f"{column} (obs)",
            xaxis=xaxis_id,
            yaxis=yaxis_id,
            line=dict(color="black", width=1),
        )
    )
    # Model data
    fig_ts.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"{column}_model"],
            name=f"{column} (model)",
            xaxis=xaxis_id,
            yaxis=yaxis_id,
            line=dict(color=px.colors.qualitative.G10[0], width=1),
        )
    )

fig_ts.update_layout(
    height=800,
    margin=dict(t=30, b=50),
    # Share same x-axis domain across all subplots
    xaxis=dict(domain=[0.05, 0.95], anchor="y", showticklabels=False),
    xaxis2=dict(domain=[0.05, 0.95], anchor="y2", showticklabels=False),
    xaxis3=dict(domain=[0.05, 0.95], anchor="y3", showticklabels=True),
    # Y-axis deatils for each subplot
    yaxis=dict(title=wave_name, anchor="x", domain=[0.7, 0.95]),
    yaxis2=dict(title=wave2_name, anchor="x2", domain=[0.37, 0.62]),
    yaxis3=dict(title=direction_name, anchor="x3", domain=[0.05, 0.3]),
    showlegend=False,
    title=timeseries_title,
)

st.plotly_chart(fig_ts, use_container_width=True, key="Timeseries")
st.divider()

# %% Statistics
st.subheader("Comparison statistics")
stats_data = waveStats(
    predictions=df[f"{wave_name}_model"], targets=df[f"{wave_name}_obs"]
)
st.table(stats_data)
st.divider()

# %% Create waverose
st.subheader("Wave roses")
# Sliderbar selection
slider_timestep = st.select_slider(
    label="Choose time period for Waverose plotting",
    options=["All", "Season", "Month", "Year"],
    value="All",
)
# Compute wave rose bins
dir_bins = np.arange(0, 361, direction_resolution)
if slider_timestep != "All":
    if slider_timestep == "Season":
        selected = st.select_slider(
            label=f"Select Austral season (e.g., Summer is DJF and Winter is JJA)",
            options=["Summer", "Autumn", "Winter", "Spring"],
        )
    else:
        times = sorted(df[slider_timestep].unique())
        if max(times) == min(times):
            selected = times[0]
            st.text(
                f"{slider_timestep} is set to {selected} (no other options available)."
            )
        else:
            selected = st.slider(
                label=f"Select {slider_timestep}",
                min_value=int(min(times)),
                max_value=int(max(times)),
                value=int(min(times)),
            )
    # Filter data
    df_selected = df[df[slider_timestep] == selected].copy()

    figs = {}
    tabs = {}
    for idx, data_source in enumerate(suffixes):

        # Name the tab:
        tabs[idx] = waverose_names[idx]
        # Select data
        # Set direction data
        df_selected.loc[:, direction_rosename] = pd.cut(
            df_selected[f"{direction_name}_{data_source}"],
            bins=dir_bins,
            right=False,
            labels=dir_bins[:-1],
        )
        # Set wave "height" data
        df_selected.loc[:, wave_name] = pd.cut(
            df_selected[f"{wave_name}_{data_source}"], bins=wave_bins, right=False
        )

        rose_data = (
            df_selected.groupby([direction_rosename, wave_name], observed=False)
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

    # Plot wave rose charts with Streamlit + Plotly
    tab1, tab2 = st.tabs([tabs[0], tabs[1]])
    with tab1:
        st.plotly_chart(figs[0], use_container_width=True, key=tabs[0])
    with tab2:
        st.plotly_chart(figs[1], use_container_width=True, key=tabs[1])
else:

    figs = {}
    tabs = {}
    for idx, data_source in enumerate(suffixes):

        # Name the tab:
        tabs[idx] = waverose_names[idx]
        # Select data
        # Set direction data
        df.loc[:, direction_rosename] = pd.cut(
            df[f"{direction_name}_{data_source}"],
            bins=dir_bins,
            right=False,
            labels=dir_bins[:-1],
        )
        # Set wave "height" data
        df.loc[:, wave_name] = pd.cut(
            df[f"{wave_name}_{data_source}"], bins=wave_bins, right=False
        )

        rose_data = (
            df.groupby([direction_rosename, wave_name], observed=False)
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

    # Plot wave rose charts with Streamlit + Plotly
    tab1, tab2 = st.tabs([tabs[0], tabs[1]])
    with tab1:
        st.plotly_chart(figs[0], use_container_width=True, key=tabs[0])
    with tab2:
        st.plotly_chart(figs[1], use_container_width=True, key=tabs[1])

# %% THE END

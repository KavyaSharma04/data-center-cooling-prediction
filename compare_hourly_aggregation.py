
import pandas as pd

print("Loading original power dataset...")

power_df = pd.read_csv(
    "data/raw/power_file.csv",
    low_memory=False
)

weather_df = pd.read_csv(
    "data/raw/weather_file.csv",
    low_memory=False
)


def parse_timestamps(series):
    cleaned = series.str.replace(
        r"\s*\([^)]*\)\s*$",
        "",
        regex=True
    )

    return pd.to_datetime(
        cleaned,
        format="%a %b %d %Y %H:%M:%S GMT%z",
        errors="coerce",
        utc=True
    )


# Prepare timestamps
power_df["ts"] = parse_timestamps(power_df["ts"])
weather_df["ts"] = parse_timestamps(weather_df["ts"])


# Prepare power data
power_df = power_df.dropna(subset=["ts"]).copy()

power_df = power_df[
    ["ts", "cooling_kw", "it_power_kw"]
].copy()

power_df["hour"] = power_df["ts"].dt.floor("h")


# Aggregate using hourly mean
power_mean = (
    power_df.groupby("hour", as_index=False)
    .agg({
        "cooling_kw": "mean",
        "it_power_kw": "mean"
    })
)


# Aggregate using hourly median
power_median = (
    power_df.groupby("hour", as_index=False)
    .agg({
        "cooling_kw": "median",
        "it_power_kw": "median"
    })
)


# Prepare weather data
weather_df = weather_df.dropna(subset=["ts"]).copy()

weather_df = weather_df[
    weather_df["ts"] >= "2016-01-01"
].copy()

weather_df = weather_df[
    ["ts", "outdoor_air_temp", "outdoor_air_humidity"]
].copy()

weather_df = (
    weather_df.groupby("ts", as_index=False)
    .agg({
        "outdoor_air_temp": "mean",
        "outdoor_air_humidity": "mean"
    })
)

weather_df["hour"] = weather_df["ts"].dt.floor("h")

weather_hourly = (
    weather_df.groupby("hour", as_index=False)
    .agg({
        "outdoor_air_temp": "mean",
        "outdoor_air_humidity": "mean"
    })
)


# Merge each power dataset with weather
mean_df = pd.merge(
    power_mean,
    weather_hourly,
    on="hour",
    how="inner"
)

median_df = pd.merge(
    power_median,
    weather_hourly,
    on="hour",
    how="inner"
)


# Compare cooling-power distributions
print("\n========== HOURLY MEAN DATASET ==========")

print(mean_df["cooling_kw"].describe())

print(
    "Records above 100 kW:",
    (mean_df["cooling_kw"] > 100).sum()
)

print("\n========== HOURLY MEDIAN DATASET ==========")

print(median_df["cooling_kw"].describe())

print(
    "Records above 100 kW:",
    (median_df["cooling_kw"] > 100).sum()
)


# Compare the two extreme hours
extreme_hours = [
    "2019-12-12 17:00:00+00:00",
    "2019-12-12 20:00:00+00:00"
]

print("\n========== EXTREME HOUR COMPARISON ==========")

for timestamp in extreme_hours:
    timestamp = pd.Timestamp(timestamp)

    mean_value = mean_df.loc[
        mean_df["hour"] == timestamp, "cooling_kw"
    ]

    median_value = median_df.loc[
        median_df["hour"] == timestamp, "cooling_kw"
    ]

    print(f"\nTimestamp: {timestamp}")

    print(
        "Hourly mean:",
        mean_value.iloc[0] if not mean_value.empty else "Not found"
    )

    print(
        "Hourly median:",
        median_value.iloc[0] if not median_value.empty else "Not found"
    )


print("\nAggregation comparison completed!")
# Save the hourly median dataset separately
median_df.to_csv(
    "results/data/hourly_median_merged_data.csv",
    index=False
)

print("\nMedian dataset saved successfully!")
print("File: results/data/hourly_median_merged_data.csv")
print("Dataset shape:", median_df.shape)
print("Columns:", median_df.columns.tolist())
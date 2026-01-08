from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(page_title="Vehicles US - EDA", layout="wide")
st.title("Exploratory Data Analysis: Vehicles US dataset")


# ---------------------------------------------------------
# Load + Clean data (this is the core of your EDA in an app)
# ---------------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, int, int]:
    # NOTE: Adjust this path if your CSV is not inside /data
    data_path = Path(__file__).parent / "vehicles_us" / "vehicles_us.csv"

    # Load dataset
    df = pd.read_csv(data_path)

    # Save original number of rows BEFORE cleaning
    original_rows = len(df)

    # ---------------------------------------------------------
    # DATA CLEANING SECTION
    # ---------------------------------------------------------

    # 1) Normalize text columns
    text_cols = ["model", "condition", "fuel", "type", "paint_color", "transmission"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    # 2) Convert date column to datetime
    if "date_posted" in df.columns:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")

    # 3) Convert numeric columns safely
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "model_year" in df.columns:
        df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce").astype("Int64")

    if "cylinders" in df.columns:
        df["cylinders"] = pd.to_numeric(df["cylinders"], errors="coerce").astype("Int64")

    if "odometer" in df.columns:
        df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")

    if "days_listed" in df.columns:
        df["days_listed"] = pd.to_numeric(df["days_listed"], errors="coerce")

    # ---------------------------------------------------------
    # Remove records where odometer equals 0 (with justification)
    #
    # During the review of the 'odometer' variable, we identified records with a value equal to 0.
    # Since these vehicles are from years prior to 2015, we conclude those values do not represent
    # new cars, but rather missing or incorrectly captured data. Due to the low proportion and to
    # keep quantitative variables consistent, we decided to remove these records before continuing
    # the exploratory analysis.
    # ---------------------------------------------------------
    odometer_zero_rows = 0
    if "odometer" in df.columns:
        odometer_zero_rows = int((df["odometer"] == 0).sum())
        df = df[df["odometer"] > 0].reset_index(drop=True)

    # 4) Convert is_4wd to boolean
    if "is_4wd" in df.columns:
        df["is_4wd"] = df["is_4wd"].fillna(0)
        df["is_4wd"] = df["is_4wd"].astype(int).astype(bool)

    # 5) Handle missing values
    if "paint_color" in df.columns:
        df["paint_color"] = df["paint_color"].fillna("unknown")

    # cylinders: fill with median by model, then global median
    if "cylinders" in df.columns and "model" in df.columns:
        cyl_median_by_model = (
            df.groupby("model")["cylinders"]
            .transform("median")
            .round()
            .astype("Int64")
        )
        df["cylinders"] = df["cylinders"].fillna(cyl_median_by_model)

        global_cyl_median = int(round(df["cylinders"].median()))
        df["cylinders"] = df["cylinders"].fillna(global_cyl_median).astype("Int64")

    # model_year: fill with median by model, then global median
    if "model_year" in df.columns and "model" in df.columns:
        model_year_median_by_model = (
            df.groupby("model")["model_year"]
            .transform("median")
            .round()
            .astype("Int64")
        )
        df["model_year"] = df["model_year"].fillna(model_year_median_by_model)

        global_year_median = int(round(df["model_year"].median()))
        df["model_year"] = df["model_year"].fillna(global_year_median).astype("Int64")

    # odometer: hierarchical median strategy
    if "odometer" in df.columns:
        if "model" in df.columns and "model_year" in df.columns:
            df["odometer"] = df["odometer"].fillna(
                df.groupby(["model", "model_year"])["odometer"].transform("median")
            )

        if "model" in df.columns:
            df["odometer"] = df["odometer"].fillna(
                df.groupby("model")["odometer"].transform("median")
            )

        df["odometer"] = df["odometer"].fillna(df["odometer"].median())

    # 6) Remove impossible values
    if "price" in df.columns:
        df.loc[df["price"] <= 0, "price"] = np.nan

    # We already removed odometer == 0 records above; here we only handle negative mileage
    if "odometer" in df.columns:
        df.loc[df["odometer"] < 0, "odometer"] = np.nan

    # 7) Feature engineering
    if "date_posted" in df.columns:
        df["post_year"] = df["date_posted"].dt.year.astype("Int64")
        df["post_month"] = df["date_posted"].dt.month.astype("Int64")

    if "model_year" in df.columns and "date_posted" in df.columns:
        df["car_age"] = (df["date_posted"].dt.year - df["model_year"]).astype("Float64")
        df.loc[df["car_age"] < 0, "car_age"] = np.nan

    # 8) Remove duplicates
    df = df.drop_duplicates()

    # 9) Optional: trim extreme outliers (for cleaner plots)
    if "price" in df.columns and df["price"].notna().any():
        p01, p99 = df["price"].quantile([0.01, 0.99])
        df = df[(df["price"].isna()) | ((df["price"] >= p01) & (df["price"] <= p99))]

    if "odometer" in df.columns and df["odometer"].notna().any():
        o01, o99 = df["odometer"].quantile([0.01, 0.99])
        df = df[(df["odometer"].isna()) | ((df["odometer"] >= o01) & (df["odometer"] <= o99))]

    return df, original_rows, odometer_zero_rows


# Load cleaned dataset + original row count + odometer zero removed count
df, original_rows, odometer_zero_rows = load_data()


# -----------------------------
# Sidebar filters (example)
# -----------------------------
st.sidebar.header("Filters")
df_filtered = df.copy()

if "price" in df_filtered.columns and df_filtered["price"].notna().any():
    min_price = int(df_filtered["price"].min())
    max_price = int(df_filtered["price"].max())
    price_range = st.sidebar.slider(
        "Price range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
    )
    df_filtered = df_filtered[
        (df_filtered["price"].isna()) |
        ((df_filtered["price"] >= price_range[0]) & (df_filtered["price"] <= price_range[1]))
    ]


# -----------------------------
# Dataset Overview (UPDATED)
# -----------------------------
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Original number of rows", f"{original_rows:,}")

with col2:
    st.metric("Rows after data cleaning", f"{len(df):,}")

with col3:
    st.metric("Rows after filters", f"{len(df_filtered):,}")

st.markdown(
    f"""
### Data Cleaning Summary and Justification

The following data cleaning steps were applied to the dataset, each one based on
exploratory findings and analytical reasoning:

- **Normalization of categorical text fields**  
  All categorical variables were converted to lowercase and trimmed to remove extra spaces.
  This step prevents the same category from being represented as multiple distinct values
  (e.g., `"SUV"` vs `"suv"`), ensuring consistency and reliable grouping during analysis.

- **Date column conversion to datetime format**  
  Date-related fields were converted to proper datetime objects to enable time-based
  calculations, comparisons, and feature engineering. Keeping dates as strings would
  limit analytical flexibility and increase the risk of interpretation errors.

- **Conversion of numerical columns to appropriate numeric types**  
  Numerical variables were explicitly converted to numeric data types. Invalid or malformed
  values were coerced to missing values to avoid calculation errors and ensure mathematical
  operations behaved as expected.

- **Removal of records where `odometer == 0` (removed: {odometer_zero_rows:,} rows)**  
  During the review of the odometer variable, records with a mileage value equal to zero
  were identified. Since these vehicles mostly belong to years prior to 2015, a zero mileage
  value does not represent new vehicles, but rather missing or incorrectly captured data.
  Due to their low proportion and in order to maintain the coherence of quantitative variables,
  these records were removed before continuing the exploratory data analysis.

- **Handling missing values using median imputation by model**  
  Missing values in key numerical variables such as `model_year`, `cylinders`, and `odometer`
  were imputed using the median value within each vehicle model. This approach preserves
  model-specific characteristics and reduces bias compared to using a global average.
  When model-level medians were not available, a global median was used as a fallback.

- **Replacement of missing categorical values with meaningful placeholders**  
  Missing categorical values (e.g., `paint_color`) were replaced with a descriptive placeholder
  such as *"unknown"*. This preserves the record while explicitly indicating the absence of
  information, instead of silently dropping data.

- **Removal of impossible numerical values**  
  Records containing logically impossible values, such as negative prices or negative odometer
  readings, were treated as invalid. These values cannot represent real-world vehicle data and
  would distort statistical summaries and visualizations if left unaddressed.

- **Feature engineering for analytical insight**  
  Additional features, such as vehicle age, were created using existing variables. These derived
  features provide more meaningful representations of vehicle condition and help explain
  relationships with price and mileage.

- **Removal of duplicate records**  
  Duplicate rows were removed to avoid double-counting vehicles and to ensure that all
  observations represent unique listings.

- **Trimming of extreme outliers for visualization purposes**  
  Extreme values in variables such as price and odometer were trimmed using percentile-based
  thresholds. This step improves visualization readability and prevents extreme cases from
  dominating graphical analysis, while still preserving the majority of the data distribution.

Overall, these cleaning steps improve data consistency, reduce noise, and ensure that the
dataset is suitable for reliable exploratory analysis and interpretation.
"""
)

if st.checkbox("Show cleaned data table"):
    st.dataframe(df_filtered, use_container_width=True)


# -----------------------------
# Price Distribution (Histogram)
# -----------------------------
st.subheader("Price Distribution (Histogram)")

st.markdown(
    """
**What it shows:**
- Whether prices are skewed (right-skew is common in car markets)
- Typical price ranges
- Presence of extreme outliers
"""
)

show_price_hist = st.checkbox("Show price histogram", value=True)

if show_price_hist:
    if "price" not in df_filtered.columns:
        st.warning("Column 'price' not found in dataset.")
    else:
        price_series = df_filtered["price"].dropna()

        if price_series.empty:
            st.warning("No valid price data available after filtering.")
        else:
            # Optional controls for bin size + range
            bins = st.slider("Number of bins", min_value=10, max_value=100, value=50, step=5)

            # Optional: allow trimming for visualization only
            trim_for_plot = st.checkbox("Trim extreme values for this plot (1st–99th percentile)", value=True)
            plot_data = df_filtered.copy()

            if trim_for_plot:
                p01, p99 = price_series.quantile([0.01, 0.99])
                plot_data = plot_data[(plot_data["price"].isna()) | ((plot_data["price"] >= p01) & (plot_data["price"] <= p99))]

            fig_price = px.histogram(
                plot_data.dropna(subset=["price"]),
                x="price",
                nbins=bins,
                title="Vehicle Price Distribution (USD)"
            )
            fig_price.update_layout(xaxis_title="Price (USD)", yaxis_title="Frequency")
            st.plotly_chart(fig_price, use_container_width=True)

            # Interpretation text (English version of your paragraph)
            st.markdown(
                """
**Interpretation (EDA):**  
The price distribution typically shows a strong positive skew (right-skew). Most vehicles concentrate in relatively lower
price ranges (often around a few thousand to tens of thousands of USD), while higher prices become increasingly rare.
This pattern is common in automotive listings: many budget vehicles, fewer mid-to-high range vehicles, and very few
luxury or collectible outliers.
"""
            )

# -----------------------------
# Price vs Vehicle Age (Scatter)
# -----------------------------
st.subheader("Price vs Vehicle Age (Scatter)")

st.markdown(
    """
**What it shows:**
- How vehicle age relates to price (depreciation patterns)
- Whether newer vehicles tend to be more expensive
- Whether outliers exist (high price at high age, etc.)
"""
)

show_price_age = st.checkbox("Show price vs car age", value=True)

if show_price_age:
    required_cols = {"price", "car_age"}
    if not required_cols.issubset(df_filtered.columns):
        st.warning("This plot requires 'price' and 'car_age' columns.")
    else:
        scatter_df = df_filtered.dropna(subset=["price", "car_age"]).copy()
        if scatter_df.empty:
            st.warning("No valid data for 'price' and 'car_age' after filtering.")
        else:
            fig_age = px.scatter(
                scatter_df,
                x="car_age",
                y="price",
                title="Price vs Vehicle Age",
                hover_data=[c for c in ["model", "condition", "type", "odometer"] if c in scatter_df.columns]
            )
            fig_age.update_layout(xaxis_title="Car Age (years)", yaxis_title="Price (USD)")
            st.plotly_chart(fig_age, use_container_width=True)

            st.markdown(
                """
**Interpretation (EDA):**  
In most markets, price decreases as vehicle age increases due to depreciation. The scatter plot helps confirm that trend,
and highlights exceptions such as high-priced older vehicles (potentially collector cars or special trims).
"""
            )

# -----------------------------
# Boxplot: Price by Vehicle Type
# -----------------------------
st.subheader("Price by Vehicle Type (Boxplot)")

st.markdown(
    """
**What it shows:**
- How vehicle prices vary across different vehicle types
- Median price differences between categories
- Price dispersion and presence of outliers per vehicle type
"""
)

show_price_type = st.checkbox("Show price by vehicle type", value=True)

if show_price_type:
    required_cols = {"price", "type"}
    if not required_cols.issubset(df_filtered.columns):
        st.warning("This plot requires 'price' and 'type' columns.")
    else:
        box_df = df_filtered.dropna(subset=["price", "type"]).copy()

        if box_df.empty:
            st.warning("No valid data available after filtering.")
        else:
            # Optional: trim extreme values only for visualization
            trim_for_plot = st.checkbox(
                "Trim extreme values for this plot (1st–99th percentile)",
                value=True,
                key="trim_price_type"
            )

            if trim_for_plot:
                p01, p99 = box_df["price"].quantile([0.01, 0.99])
                box_df = box_df[(box_df["price"] >= p01) & (box_df["price"] <= p99)]

            # 🔹 OPTION A: order vehicle types by median price
            order = (
                box_df.groupby("type")["price"]
                .median()
                .sort_values()
                .index
            )

            fig_box = px.box(
                box_df,
                x="type",
                y="price",
                category_orders={"type": order},
                title="Price Distribution by Vehicle Type",
            )

            fig_box.update_layout(
                xaxis_title="Vehicle Type",
                yaxis_title="Price (USD)"
            )

            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown(
                """
**Interpretation (EDA):**  
The boxplot reveals clear price differences across vehicle types. Larger and more specialized vehicles
such as trucks, pickups, off-road vehicles, and buses tend to show higher median prices and greater
price dispersion. Smaller vehicle types such as sedans and hatchbacks exhibit lower median prices and
more compact distributions, reflecting a more standardized segment of the used-car market.
"""
            )

# -----------------------------
# Boxplot: Price by Condition
# -----------------------------
st.subheader("Price by Condition (Boxplot)")

st.markdown(
    """
**What it shows:**
- How vehicle prices vary by condition
- Differences in median price across condition categories
- Price dispersion and outliers within each condition level
"""
)

show_price_condition = st.checkbox("Show price by condition", value=True)

if show_price_condition:
    required_cols = {"price", "condition"}
    if not required_cols.issubset(df_filtered.columns):
        st.warning("This plot requires 'price' and 'condition' columns.")
    else:
        box_df = df_filtered.dropna(subset=["price", "condition"]).copy()

        if box_df.empty:
            st.warning("No valid data available after filtering.")
        else:
            # Optional: trim extreme prices only for visualization
            trim_for_plot = st.checkbox(
                "Trim extreme values for this plot (1st–99th percentile)",
                value=True,
                key="trim_price_condition"
            )

            if trim_for_plot:
                p01, p99 = box_df["price"].quantile([0.01, 0.99])
                box_df = box_df[(box_df["price"] >= p01) & (box_df["price"] <= p99)]

            # Order conditions by median price (helps storytelling)
            order = (
                box_df.groupby("condition")["price"]
                .median()
                .sort_values()
                .index
            )

            fig = px.box(
                box_df,
                x="condition",
                y="price",
                category_orders={"condition": order},
                title="Price Distribution by Condition",
            )
            fig.update_layout(xaxis_title="Condition", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
**Interpretation (EDA):**  
Prices generally increase as vehicle condition improves. Better conditions typically show higher median prices,
while lower conditions tend to concentrate at lower prices and may display wider variability due to inconsistent
maintenance and vehicle history.
"""
            )

# -----------------------------
# Histogram: Odometer Distribution
# -----------------------------
st.subheader("Odometer Distribution (Histogram)")

st.markdown(
    """
**What it shows:**
- Typical mileage ranges in the dataset
- Whether odometer values are skewed
- Presence of extreme mileage outliers
"""
)

show_odo_hist = st.checkbox("Show odometer histogram", value=True)

if show_odo_hist:
    if "odometer" not in df_filtered.columns:
        st.warning("Column 'odometer' not found in dataset.")
    else:
        odo_series = df_filtered["odometer"].dropna()

        if odo_series.empty:
            st.warning("No valid odometer data available after filtering.")
        else:
            bins = st.slider("Number of bins (odometer)", 10, 120, 50, 5, key="odo_bins")

            # Optional: trim only for visualization
            trim_for_plot = st.checkbox(
                "Trim extreme values for this plot (1st–99th percentile)",
                value=True,
                key="trim_odo_hist"
            )

            plot_df = df_filtered.dropna(subset=["odometer"]).copy()

            if trim_for_plot:
                o01, o99 = odo_series.quantile([0.01, 0.99])
                plot_df = plot_df[(plot_df["odometer"] >= o01) & (plot_df["odometer"] <= o99)]

            fig = px.histogram(
                plot_df,
                x="odometer",
                nbins=bins,
                title="Odometer Distribution"
            )
            fig.update_layout(xaxis_title="Odometer", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
**Interpretation (EDA):**  
The odometer distribution often shows a right-skew, where most vehicles cluster around typical mileage ranges
and fewer vehicles appear with extremely high mileage. Outliers can represent heavily used vehicles or data-entry issues.
"""
            )

# -----------------------------
# Scatter: Price vs Odometer
# -----------------------------
st.subheader("Price vs Odometer (Scatter)")

st.markdown(
    """
**What it shows:**
- The relationship between mileage and price
- Whether higher mileage tends to correlate with lower price
- Outliers (e.g., high price with high mileage)
"""
)

show_price_odo = st.checkbox("Show price vs odometer", value=True)

if show_price_odo:
    required_cols = {"price", "odometer"}
    if not required_cols.issubset(df_filtered.columns):
        st.warning("This plot requires 'price' and 'odometer' columns.")
    else:
        scatter_df = df_filtered.dropna(subset=["price", "odometer"]).copy()

        if scatter_df.empty:
            st.warning("No valid data available after filtering.")
        else:
            # Optional: trim extremes only for visualization
            trim_for_plot = st.checkbox(
                "Trim extreme values for this plot (1st–99th percentile for both axes)",
                value=True,
                key="trim_price_odo"
            )

            if trim_for_plot:
                p01, p99 = scatter_df["price"].quantile([0.01, 0.99])
                o01, o99 = scatter_df["odometer"].quantile([0.01, 0.99])
                scatter_df = scatter_df[
                    (scatter_df["price"].between(p01, p99)) &
                    (scatter_df["odometer"].between(o01, o99))
                ]

            fig = px.scatter(
                scatter_df,
                x="odometer",
                y="price",
                title="Price vs Odometer",
                hover_data=[c for c in ["model", "condition", "type", "model_year"] if c in scatter_df.columns]
            )
            fig.update_layout(xaxis_title="Odometer", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
**Interpretation (EDA):**  
A negative relationship is commonly expected: as mileage increases, price tends to decrease.
However, the scatter plot may show substantial dispersion due to factors such as vehicle type,
condition, model year, and brand differences.
"""
            )

# -----------------------------
# Correlation Heatmap (Numerical Variables)
# -----------------------------
st.subheader("Correlation Heatmap (Numerical Variables)")

st.markdown(
    """
**What it shows:**
- Strength and direction of linear relationships between numerical variables
- Which variables are most strongly correlated with price
- Potential multicollinearity between predictors
"""
)

show_corr = st.checkbox("Show correlation heatmap", value=True)

if show_corr:
    # Select only relevant numerical columns
    numeric_cols = [
        col for col in ["price", "odometer", "car_age", "model_year"]
        if col in df_filtered.columns
    ]

    corr_df = df_filtered[numeric_cols].dropna()

    if corr_df.empty or corr_df.shape[1] < 2:
        st.warning("Not enough numerical data available to compute correlations.")
    else:
        corr_matrix = corr_df.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix of Numerical Variables"
        )

        fig_corr.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables"
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(
            """
**Interpretation (Correlation Heatmap):**  

The correlation matrix highlights several strong and expected linear relationships
between numerical variables in the dataset.

Vehicle price shows a **moderate negative correlation** with both odometer
(-0.48) and vehicle age (-0.45), indicating that vehicles with higher mileage
and older vehicles tend to be priced lower. This aligns with standard depreciation
patterns in the used-car market.

Price also presents a **moderate positive correlation** with model year (0.46),
suggesting that newer vehicles generally command higher prices.

As expected, vehicle age and model year exhibit an **almost perfect negative
correlation (-1.00)**, since these variables represent inverse measures of the
same temporal characteristic. This confirms internal consistency within the dataset.

Additionally, odometer is positively correlated with vehicle age (0.49) and
negatively correlated with model year (-0.49), reinforcing the idea that older
vehicles tend to accumulate more mileage over time.

Overall, the correlation structure is coherent and supports the trends observed
in previous visual analyses, such as scatter plots and boxplots.
"""
        )

st.markdown(
    """ **General Conclusion**

The exploratory data analysis reveals clear and consistent pricing patterns
within the used vehicle market represented in this dataset.

Vehicle price is primarily influenced by **age, mileage, condition, and vehicle type**.
As vehicles age and accumulate mileage, their prices tend to decrease, reflecting
expected depreciation behavior. This relationship is supported by both scatter plots
and correlation analysis.

Categorical factors further explain price variability. Vehicles in better conditions
and larger or more specialized categories, such as trucks, SUVs, and off-road vehicles,
generally show higher median prices and greater dispersion, while sedans and hatchbacks
tend to cluster in lower and more standardized price ranges.

The presence of outliers across multiple distributions highlights the heterogeneous
nature of the used-car market, where similar vehicles may differ substantially in price
due to factors such as brand, trim level, usage history, or market demand.

The applied data cleaning process improved data consistency and reliability, allowing
for clearer insights and more accurate interpretation of market dynamics. Overall,
the analysis provides a solid foundation for understanding pricing behavior and can
support informed decision-making for buyers, sellers, and marketplace platforms.

""" )
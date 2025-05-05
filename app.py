import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Marketing Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    """Loads data from CSV and preprocesses it."""
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        if df.empty:
             st.error(f"Error: Loaded file {file_path} is empty.")
             return None
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}.")
        return None
    except pd.errors.EmptyDataError:
         st.error(f"Error: File {file_path} is empty or invalid.")
         return None
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        return None

# --- Metric Calculation ---
def calculate_metrics(df):
    """Calculates derived metrics on the dataframe."""
    df_calc = df.copy()
    if df_calc.empty:
        # Define columns expected after calculation even if input is empty
        expected_cols = df.columns.tolist() + ['ROAS', 'CPC', 'CTR', 'Conversion Rate']
        return pd.DataFrame(columns=expected_cols)

    df_calc['ROAS'] = df_calc.apply(lambda row: row['Revenue'] / row['Spend'] if row['Spend'] > 0 else 0, axis=1)
    df_calc['CPC'] = df_calc.apply(lambda row: row['Spend'] / row['Conversions'] if row['Conversions'] > 0 else pd.NA, axis=1) # Use NA for undefined CPC
    df_calc['CTR'] = df_calc.apply(lambda row: row['Clicks'] / row['Impressions'] if row['Impressions'] > 0 else 0, axis=1)
    df_calc['Conversion Rate'] = df_calc.apply(lambda row: row['Conversions'] / row['Clicks'] if row['Clicks'] > 0 else 0, axis=1)
    return df_calc

# --- Define Metrics ---
PRIMARY_METRICS = ['Spend', 'Impressions', 'Clicks', 'Conversions', 'Revenue']
CALCULATED_METRICS = ['ROAS', 'CPC', 'CTR', 'Conversion Rate']
ALL_METRICS = PRIMARY_METRICS + CALCULATED_METRICS
AGGREGATABLE_METRICS = PRIMARY_METRICS # Metrics suitable for direct sum aggregation

# --- Formatting Helper ---
def format_metric(metric_name, value):
    """Formats metric values for display."""
    if pd.isna(value): return "-"
    try:
        if metric_name in ['Spend', 'Revenue', 'CPC']: return f"${value:,.2f}"
        if metric_name in ['CTR', 'Conversion Rate']: return f"{value:.2%}"
        if metric_name == 'ROAS': return f"{value:.2f}x"
        return f"{value:,.0f}" # Default for counts like Impressions, Clicks, Conversions
    except (ValueError, TypeError): return str(value)


# --- Load Data ---
df_raw = load_data('marketing_campaign_data.csv')

if df_raw is not None:

    # --- Sidebar Filters & Configuration ---
    st.sidebar.header("Filters")
    min_date = df_raw['Date'].min().date()
    max_date = df_raw['Date'].max().date()
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )

    all_channels = sorted(df_raw['Channel'].unique())
    selected_channels = st.sidebar.multiselect(
        "Select Channel(s)", options=all_channels, default=all_channels
    )

    st.sidebar.header("Dashboard Configuration")
    # --- KPI Selection ---
    selected_kpis = st.sidebar.multiselect(
        "Select KPI Metrics", options=ALL_METRICS,
        default=['Revenue', 'Spend', 'ROAS', 'Conversions'],
        help="Choose key metrics to display at the top."
    )

    # --- Channel Comparison Metric Selection ---
    selected_bar_metric = st.sidebar.selectbox(
            "Channel Comparison Metric",
            options=ALL_METRICS,
            index=ALL_METRICS.index('ROAS') if 'ROAS' in ALL_METRICS else 0, # Default to ROAS
            help="Choose the metric to compare across channels in the bar chart."
        )


    # --- Main Dashboard Area ---
    st.title("📊 Interactive Marketing Analysis")

    # --- Apply Filters ---
    start_datetime = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_datetime = datetime.datetime.combine(end_date, datetime.datetime.max.time())
    df_filtered = df_raw[
        (df_raw['Date'] >= start_datetime) &
        (df_raw['Date'] <= end_datetime) &
        (df_raw['Channel'].isin(selected_channels))
    ].copy()

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # --- Calculate Metrics on Filtered Data ---
    df_calculated = calculate_metrics(df_filtered)

    # --- Calculate Overall Aggregates for KPIs ---
    overall_totals = {metric: df_calculated[metric].sum() for metric in AGGREGATABLE_METRICS if metric in df_calculated.columns}
    overall_calculated = {}
    # Calculate derived metrics based on overall totals
    if 'Revenue' in overall_totals and 'Spend' in overall_totals:
        overall_calculated['ROAS'] = overall_totals['Revenue'] / overall_totals['Spend'] if overall_totals['Spend'] > 0 else 0
    if 'Spend' in overall_totals and 'Conversions' in overall_totals:
        overall_calculated['CPC'] = overall_totals['Spend'] / overall_totals['Conversions'] if overall_totals['Conversions'] > 0 else pd.NA
    if 'Clicks' in overall_totals and 'Impressions' in overall_totals:
        overall_calculated['CTR'] = overall_totals['Clicks'] / overall_totals['Impressions'] if overall_totals['Impressions'] > 0 else 0
    if 'Conversions' in overall_totals and 'Clicks' in overall_totals:
        overall_calculated['Conversion Rate'] = overall_totals['Conversions'] / overall_totals['Clicks'] if overall_totals['Clicks'] > 0 else 0

    kpi_values = {**overall_totals, **overall_calculated}

    # --- Display KPIs ---
    st.header("Overall Performance Summary")
    if not selected_kpis:
        st.warning("Please select at least one KPI metric in the sidebar.")
    else:
        num_kpis = len(selected_kpis)
        # Dynamically adjust columns for KPIs, max 5 per row for readability
        cols = st.columns(min(num_kpis, 5))
        col_index = 0
        for kpi in selected_kpis:
            with cols[col_index % 5]:
                st.metric(label=kpi, value=format_metric(kpi, kpi_values.get(kpi, pd.NA)))
            col_index += 1


    # --- Aggregate Data for Charts ---
    # Time Series (Daily)
    agg_dict_time = {metric: 'sum' for metric in AGGREGATABLE_METRICS if metric in df_calculated.columns}
    df_time_series_agg = df_calculated.groupby('Date').agg(agg_dict_time).reset_index()
    df_time_series = calculate_metrics(df_time_series_agg) # Recalculate derived for daily view

    # Channel Summary
    agg_dict_channel = {metric: 'sum' for metric in AGGREGATABLE_METRICS if metric in df_calculated.columns}
    df_channel_summary_agg = df_calculated.groupby('Channel').agg(agg_dict_channel).reset_index()
    df_channel_summary = calculate_metrics(df_channel_summary_agg) # Recalculate derived for channel view

    # Campaign Summary (for Table)
    agg_dict_campaign = {metric: 'sum' for metric in AGGREGATABLE_METRICS if metric in df_calculated.columns}
    df_campaign_summary_agg = df_calculated.groupby(['CampaignName', 'Channel']).agg(agg_dict_campaign).reset_index()
    df_campaign_summary = calculate_metrics(df_campaign_summary_agg) # Recalculate derived for campaign view


    # --- Display Visualizations ---
    st.header("Performance Trends Over Time")
    if df_time_series.empty:
        st.warning("No time series data to display.")
    else:
        # Fixed Trend Chart Metrics
        trend_metrics = ['Revenue', 'Spend', 'ROAS']
        valid_trend_metrics = [m for m in trend_metrics if m in df_time_series.columns]
        if valid_trend_metrics:
            fig_time_trends = px.line(df_time_series, x='Date', y=valid_trend_metrics,
                                     title='Key Metrics Over Time', markers=True,
                                     labels={'value': 'Metric Value', 'variable': 'Metric'})
            fig_time_trends.update_layout(hovermode="x unified")
            st.plotly_chart(fig_time_trends, use_container_width=True)
        else:
            st.warning("Required metrics for trend chart not available.")


    st.header("Channel Performance Comparison")
    # Use the selected_bar_metric from the sidebar
    if not selected_bar_metric:
        st.warning("Please select a metric for Channel Comparison in the sidebar.")
    elif df_channel_summary.empty or selected_bar_metric not in df_channel_summary.columns:
         st.warning(f"Cannot generate Channel Comparison chart. Data empty or metric '{selected_bar_metric}' not available.")
    else:
        # Sort bars based on selected metric (descending for most, ascending for cost metrics)
        ascending_sort = selected_bar_metric in ['CPC', 'Spend'] # Lower is generally better for these
        df_channel_summary_sorted = df_channel_summary.sort_values(selected_bar_metric, ascending=ascending_sort, na_position='last')

        fig_channel_bar = px.bar(df_channel_summary_sorted,
                                  x='Channel', y=selected_bar_metric,
                                  title=f'{selected_bar_metric} by Channel',
                                  color='Channel', labels={selected_bar_metric: selected_bar_metric})
        st.plotly_chart(fig_channel_bar, use_container_width=True)


    st.header("Campaign Details")
    if df_campaign_summary.empty:
        st.warning("No campaign details to display.")
    else:
        # Fixed Columns for Detail Table
        display_cols = ['CampaignName', 'Channel', 'Spend', 'Revenue', 'ROAS',
                        'Conversions', 'CPC', 'Clicks', 'CTR', 'Impressions'] # Added Impressions
        valid_display_cols = [col for col in display_cols if col in df_campaign_summary.columns]

        if not valid_display_cols:
            st.warning("No valid columns available for the details table.")
        else:
            df_display = df_campaign_summary[valid_display_cols]
            # Create format dict using list comprehension for conciseness
            format_dict = {col: format_metric(col, 0).split('0')[0].replace(',', '') + '{:,.2f}' if col in ['Spend', 'Revenue', 'CPC']
                       else format_metric(col, 0).split('0')[0] + '{:.2%}' if col in ['CTR', 'Conversion Rate']
                       else format_metric(col, 0).split('0')[0].replace('.00','') + '{:,.0f}' if col in PRIMARY_METRICS # Counts
                       else format_metric(col, 0).split('0')[0] + '{:.2f}x' if col == 'ROAS' # Specific for ROAS
                       else None # Default no format
                       for col in valid_display_cols}
            # Filter out None values from format_dict
            format_dict = {k: v for k, v in format_dict.items() if v is not None}


            st.dataframe(df_display.style.format(format_dict, na_rep="-", precision=2),
                         use_container_width=True, hide_index=True)

else:
    st.warning("Data could not be loaded. Dashboard cannot be displayed.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Interactive Marketing Dashboard")
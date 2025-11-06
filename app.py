import streamlit as st
import os
import requests
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import plotly.express as px


# Load SQLite data
@st.cache_data
def load_data():
    conn = sqlite3.connect("database.db")
    df = pd.read_sql("SELECT * FROM unemployment", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df

df = load_data()
st.success("✅ Data loaded from SQLite")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
selected_state = st.sidebar.selectbox("Select State", df['State'].unique())
filtered_df = df[df['State'] == selected_state]

# Display data
st.subheader(f"📊 Unemployment Data for {selected_state}")
st.dataframe(filtered_df)

# Bar chart: Estimated Unemployment Rate
st.subheader("📈 Estimated Unemployment Rate Over Time")
fig1, ax1 = plt.subplots()
sns.lineplot(data=filtered_df, x="Date", y="Estimated Unemployment Rate (%)", ax=ax1)
st.pyplot(fig1)

# Heatmap: Correlation
st.subheader("📊 Heatmap of Correlation Between Columns")
fig2, ax2 = plt.subplots()
sns.heatmap(filtered_df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Prophet Forecast for 2025
# Prophet Forecast for 2025
st.subheader("🔮 Forecast: Estimated Unemployment Rate in 2025")

# Prepare data for Prophet
prophet_df = filtered_df[["Date", "Estimated Unemployment Rate (%)"]].rename(columns={
    "Date": "ds",
    "Estimated Unemployment Rate (%)": "y"
})

# Initialize and train model
model = Prophet()
model.fit(prophet_df)

# Forecast until 2025 (add enough months ahead)
future = model.make_future_dataframe(periods=60, freq='M')  # 60 months = 5 years approx
forecast = model.predict(future)

# Ensure datetime
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Filter only forecast for 2025
forecast_2025 = forecast[forecast['ds'].dt.year == 2025].copy()

# Rename columns for display
forecast_2025.rename(columns={
    'ds': 'Date',
    'yhat': 'Estimated Unemployment Rate (%)',
    'yhat_lower': 'Lower Estimate (%)',
    'yhat_upper': 'Upper Estimate (%)'
}, inplace=True)

# Format Date for table display
forecast_2025['Date'] = forecast_2025['Date'].dt.strftime('%b %Y')

# Show Forecast Table
st.subheader("📈 Forecasted Unemployment Rate for 2025")
if not forecast_2025.empty:
    st.dataframe(forecast_2025[['Date', 'Estimated Unemployment Rate (%)', 'Lower Estimate (%)', 'Upper Estimate (%)']])
else:
    st.warning("❌ No forecast data available for 2025. Try increasing the forecast period.")

# Plot full forecast with confidence interval
st.subheader("📊 Forecast Visualization")

# Rename for plotting
plot_df = forecast.rename(columns={
    'ds': 'Date',
    'yhat': 'Estimated Unemployment Rate (%)',
    'yhat_lower': 'Lower Estimate (%)',
    'yhat_upper': 'Upper Estimate (%)'
})

fig = px.line(
    plot_df,
    x='Date',
    y='Estimated Unemployment Rate (%)',
    title="📉 Forecasted Unemployment Rate (with Confidence Interval)",
    labels={"Estimated Unemployment Rate (%)": "Unemployment Rate (%)"},
)

# Add confidence bands
fig.add_scatter(
    x=plot_df['Date'],
    y=plot_df['Upper Estimate (%)'],
    mode='lines',
    line=dict(width=0),
    name='Upper Estimate (%)',
    showlegend=True
)
fig.add_scatter(
    x=plot_df['Date'],
    y=plot_df['Lower Estimate (%)'],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    name='Lower Estimate (%)',
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Smart Summary using Hugging Face summarizer

import os
import requests
import streamlit as st

# Hugging Face settings (set HF_API_TOKEN in Streamlit Secrets)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

@st.cache_resource
def summarize_text_remote(text: str, max_length: int = 150, min_length: int = 40, timeout: int = 60):
    """Call Hugging Face Inference API to summarize text. Raises on HTTP errors."""
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set. Add it to Streamlit Secrets.")
    payload = {"inputs": text, "parameters": {"max_length": max_length, "min_length": min_length}}
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    # Typical response shape: [{ "summary_text": "..." }]
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "summary_text" in result[0]:
        return result[0]["summary_text"]
    if isinstance(result, dict) and "summary_text" in result:
        return result["summary_text"]
    return str(result)

# Ensure 'filtered_df' exists. If not, fall back to entire 'df'. Replace names if your app uses different variable names.
try:
    # if the app created filtered_df earlier from user selections, use it; otherwise fall back
    if "filtered_df" in globals() and filtered_df is not None:
        df_for_summary = filtered_df.copy()
    else:
        df_for_summary = df.copy()  # fallback to full dataset

    st.subheader("AI-Generated Summary")

    # Choose which numeric columns to include (adapt these to your dataset column names)
    cols_for_summary = [
        'Estimated Unemployment Rate (%)',
        'Estimated Employed',
        'Estimated Labour Participation Rate (%)',
        'Literacy Rate (%)',
        'GDP per Capita'
    ]
    # Keep only existing columns
    cols_existing = [c for c in cols_for_summary if c in df_for_summary.columns]

    if df_for_summary.empty:
        st.info("No data available to summarize for the selected filters.")
    elif len(cols_existing) == 0:
        st.info("No matching numeric columns found to summarize. Check column names.")
    else:
        # Build a human-readable summary input string
        summary_stats = df_for_summary[cols_existing].describe().to_string()
        # If you have state/date info, include it
        header_parts = []
        if "State" in df_for_summary.columns:
            header_parts.append("State(s): " + ", ".join(map(str, df_for_summary['State'].unique()[:10])))
        if "Date" in df_for_summary.columns:
            min_date = df_for_summary['Date'].min()
            max_date = df_for_summary['Date'].max()
            header_parts.append(f"Date range: {min_date.date()} to {max_date.date()}")
        header = " | ".join(header_parts)
        summary_input = (header + "\n\n" + summary_stats) if header else summary_stats

        # Call the remote summarizer and display
        try:
            with st.spinner("Generating AI summary..."):
                ai_summary = summarize_text_remote(summary_input, max_length=150, min_length=40)
            st.info(ai_summary)
        except RuntimeError as e:
            st.error(str(e))
            st.write("Make sure HF_API_TOKEN is set in Streamlit Secrets.")
        except requests.HTTPError as e:
            st.error(f"Summarization API error: {e}")
            st.write("Response:", getattr(e.response, "text", "no details"))
        except Exception as e:
            st.error("Unexpected error while generating summary.")
            st.write(repr(e))

except NameError:
    st.warning("filtered_df or df not defined. Ensure this summary block is placed after data filtering code.")
except Exception as e:
    st.error("Error preparing AI summary.")
    st.write(repr(e))




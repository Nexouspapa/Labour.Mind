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

# Replace previous summarize_text_remote with this (uses new HF router endpoint)
import os
import requests
import streamlit as st

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
ROUTER_URL = "https://router.huggingface.co/hf-inference"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"} if HF_API_TOKEN else {}

def local_fallback_summary(text: str, n_sentences: int = 3) -> str:
    """Very small, deterministic fallback summary (no external API)."""
    # Simple heuristic: take first n_sentences from text (split by periods)
    parts = [p.strip() for p in text.split('.') if p.strip()]
    if not parts:
        return "No textual data available to summarize."
    return '. '.join(parts[:n_sentences]) + ('.' if len(parts) >= n_sentences else '')

@st.cache_resource
def summarize_text_remote_router(text: str, model: str = "sshleifer/distilbart-cnn-12-6",
                                 max_length: int = 150, min_length: int = 40, timeout: int = 60) -> str:
    """
    Summarize text using Hugging Face Router API:
    POST https://router.huggingface.co/hf-inference
    Payload: { "model": "<model-id>", "inputs": "...", "parameters": {...} }
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set. Add it in Streamlit Secrets.")

    payload = {
        "model": model,
        "inputs": text,
        "parameters": {"max_length": max_length, "min_length": min_length}
    }

    resp = requests.post(ROUTER_URL, headers=HEADERS, json=payload, timeout=timeout)

    # If model removed or router returns 410/404, raise for handling below
    resp.raise_for_status()
    result = resp.json()

    # Router returns a list of possible outputs or an object; handle common shapes
    # Typical successful shape: [{"generated_text": "..."}] OR [{"summary_text":"..."}] OR {"error": "..."}
    if isinstance(result, list) and len(result) > 0:
        first = result[0]
        if isinstance(first, dict):
            for key in ("summary_text", "generated_text", "text"):
                if key in first:
                    return first[key]
            # last resort: return joined dict text
            return str(first)
        return str(first)
    if isinstance(result, dict):
        # sometimes the router returns {"error": "..."} or {"summary_text": "..."}
        if "summary_text" in result:
            return result["summary_text"]
        if "generated_text" in result:
            return result["generated_text"]
        if "error" in result:
            raise RuntimeError(f"HuggingFace Router Error: {result['error']}")
        return str(result)

    return str(result)

# Example usage integration (defensive wrapper)
def get_ai_summary_or_fallback(text_for_summary: str) -> str:
    try:
        with st.spinner("Generating AI summary..."):
            return summarize_text_remote_router(text_for_summary)
    except requests.HTTPError as http_err:
        # Common HTTP error cases: 410 (gone), 403 (forbidden), 429 (rate limit)
        st.error(f"Summarization API error: {http_err}")
        try:
            # Show server response body if available (safe for logs/preview)
            st.write("Response:", http_err.response.text)
        except Exception:
            pass
        # Fallback to local summarizer
        st.warning("Using local fallback summary due to remote API error.")
        return local_fallback_summary(text_for_summary)
    except RuntimeError as e:
        st.error(str(e))
        st.warning("Using local fallback summary.")
        return local_fallback_summary(text_for_summary)
    except Exception as e:
        st.error("Unexpected error calling summarization API.")
        st.write(repr(e))
        return local_fallback_summary(text_for_summary)





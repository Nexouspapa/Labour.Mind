# Labour.Mind
# Unemployment & Economic Insights Dashboard  
### A Streamlit-Powered Data Analysis and Forecasting App with AI-Generated Summaries  

An interactive Streamlit dashboard that analyzes and forecasts **unemployment trends, literacy rates, labour participation, GDP per capita, and population** across Indian states.  

It integrates **Prophet** for time-series forecasting and **Hugging Face Transformers** for AI-generated summaries, providing deep and intelligent insights into India’s employment landscape.  

---

## Key Features  

- **State-Wise Analytics** — Explore detailed metrics for every Indian state:  
  - Unemployment Rate (%)  
  - Estimated Employed Population  
  - Labour Participation Rate (%)  
  - Literacy Rate (%)  
  - GDP per Capita  
  - Population  

- **Forecasting** — Predict future unemployment trends using Facebook Prophet.  
- **Heatmaps & Graphs** — Visualize unemployment, literacy, and GDP data interactively using Seaborn and Plotly Express.  
- **AI-Generated Summaries** — Automatically generate insights using Hugging Face Transformers (DistilBART).  
- **Historical vs Future Trends** — Compare past and predicted data on the same graph.  
- **Optimized Performance** — Uses Streamlit caching for faster loading.  

---

## Technologies & Tools Used  

| Category | Tools / Libraries | Purpose |
|-----------|------------------|----------|
| Framework | Streamlit | Interactive web app dashboard |
| Data Handling | Pandas, SQLite3 | Data processing and storage |
| Visualization | Matplotlib, Seaborn, Plotly Express | Charts, graphs, and heatmaps |
| Forecasting | Prophet | Time-series unemployment prediction |
| AI Summarization | Transformers, Torch | Text summarization for insights |
| Database | SQLite | Stores state-wise unemployment data |
| Miscellaneous | NumPy, Requests | Data manipulation and API calls |

---

## How It Works  

### 1. Data Loading  
- Reads data from `database.db` using pandas and sqlite3.  
- Data includes unemployment rate, literacy rate, GDP per capita, population, and other metrics.  
- Cached with `@st.cache_data` for faster reloads.  

### 2. Visualization  
- Heatmaps show unemployment, literacy, and GDP patterns using Seaborn.  
- Line graphs display historical vs forecasted unemployment trends.  
- Interactive maps built with Plotly Express.  
- Bar charts compare literacy, GDP, and participation metrics.  

### 3. Forecasting (Prophet)  

```python
from prophet import Prophet
model = Prophet()
model.fit(df[['Date', 'Unemployment Rate (%)']].rename(columns={'Date': 'ds', 'Unemployment Rate (%)': 'y'}))
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

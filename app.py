import os
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# ── Load data ───────────────────────────────────────────────────
DATA_URL = "https://github.com/SYLVESTER1922/QSR/raw/refs/heads/main/simbisa_kenya_master_published.csv"
df = pd.read_csv(DATA_URL)
df['date'] = pd.to_datetime(df['date'])

forecasts_df = pd.read_csv('qsr_forecasts.csv')
forecasts_df['ds'] = pd.to_datetime(forecasts_df['ds'])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

RELIABLE_SITES = [
    'Crossroads Mall','Junction Plaza','Piazza Court',
    'Garden Court','Metro Market','Bon Marché Plaza'
]
BRANDS = df['brand'].unique().tolist()

COLORS = {
    'All Sites & Brands': '#1e2d5e',
    'Crust Co.':          '#c9a84c',
    'Flame & Feather':    '#2ecc71',
    'Cala Grill':         '#e74c3c',
    'Frostbite Creamery': '#9b59b6',
    'Crossroads Mall':    '#1e2d5e',
    'Junction Plaza':     '#c9a84c',
    'Piazza Court':       '#2ecc71',
    'Garden Court':       '#e74c3c',
    'Metro Market':       '#9b59b6',
    'Bon Marché Plaza':   '#1abc9c',
}

def hex_to_rgba(hex_color, alpha=0.1):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def generate_forecast(segment_type, segment_name, horizon):
    seg_map = {'Overall': 'overall', 'By Brand': 'brand', 'By Site': 'site'}
    seg_key = seg_map[segment_type]
    fc = forecasts_df[
        (forecasts_df['segment_type'] == seg_key) &
        (forecasts_df['segment_name'] == segment_name)
    ].copy()
    if fc.empty:
        return go.Figure().update_layout(title="No forecast available for this selection"), ""
    fc = fc.head(int(horizon))
    
if seg_key == 'overall':
    hist = df.groupby('date')['daily_revenue_usd'].sum().reset_index().sort_values('date')
elif seg_key == 'brand':
    hist = df[df['brand'] == segment_name].groupby('date')['daily_revenue_usd'].sum().reset_index().sort_values('date')
else:
    hist = df[df['site'] == segment_name].groupby('date')['daily_revenue_usd'].sum().reset_index().sort_values('date')
hist.columns = ['ds', 'y']
    
    color = COLORS.get(segment_name, '#1e2d5e')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist['ds'], y=hist['y'],
        name='Historical Revenue',
        line=dict(color='#aaaaaa', width=1.5),
        mode='lines', opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=fc['ds'], y=fc['yhat'],
        name=f'{horizon}-Day Forecast',
        line=dict(color=color, width=3),
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc['ds'], fc['ds'][::-1]]),
        y=pd.concat([fc['yhat_upper'], fc['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor=hex_to_rgba(color, 0.15),
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        hoverinfo='skip'
    ))
    fig.update_layout(
        title=dict(text=f"{segment_name} — {horizon}-Day Revenue Forecast", font=dict(size=16, color='#1e2d5e')),
        xaxis=dict(title="Date", gridcolor='#f0f0f0'),
        yaxis=dict(title="Daily Revenue (USD)", tickprefix="$", tickformat=",.0f", gridcolor='#f0f0f0'),
        template="plotly_white", hovermode="x unified", height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    total    = fc['yhat'].sum()
    avg      = fc['yhat'].mean()
    peak     = fc['yhat'].max()
    peak_day = fc.loc[fc['yhat'].idxmax(), 'ds'].strftime('%b %d, %Y')
    summary = f"""**{horizon}-Day Forecast Summary — {segment_name}**

| Metric | Value |
|---|---|
| Predicted Total Revenue | **${total:,.2f}** |
| Average Daily Revenue | **${avg:,.2f}** |
| Peak Day | **${peak:,.2f}** on {peak_day} |
| Forecast Period | {fc['ds'].min().strftime('%b %d')} → {fc['ds'].max().strftime('%b %d, %Y')} |
"""
    return fig, summary

def update_segment_choices(segment_type):
    if segment_type == 'Overall':
        return gr.update(choices=['All Sites & Brands'], value='All Sites & Brands')
    elif segment_type == 'By Brand':
        return gr.update(choices=BRANDS, value=BRANDS[0])
    else:
        return gr.update(choices=RELIABLE_SITES, value=RELIABLE_SITES[0])

def build_context():
    brand_rev   = df.groupby('brand')['daily_revenue_usd'].sum().to_dict()
    site_rev    = df.groupby('site')['daily_revenue_usd'].sum().to_dict()
    yoy         = df.groupby(df['date'].dt.year)['daily_revenue_usd'].sum().to_dict()
    dow         = df.groupby(df['date'].dt.day_name())['daily_revenue_usd'].mean().to_dict()
    monthly     = df.groupby(df['date'].dt.to_period('M'))['daily_revenue_usd'].sum()
    best_month  = str(monthly.idxmax())
    worst_month = str(monthly.idxmin())
    return f"""
You are a QSR (Quick Service Restaurant) data analyst assistant for the Continental QSR Group, Kenya.
You have access to 2 years of daily revenue data (Jan 2020 – Dec 2021) across 8 sites and 4 brands.

KEY DATA FACTS:
- Total revenue: $15,868,509
- Date range: Jan 2020 – Dec 2021
- Brands: {', '.join(BRANDS)}
- Sites: {', '.join(df['site'].unique().tolist())}

REVENUE BY BRAND: {brand_rev}
REVENUE BY SITE: {site_rev}
YEAR ON YEAR: {yoy} (12.7% growth)
AVG REVENUE BY DAY OF WEEK: {dow}
BEST MONTH: {best_month}
WORST MONTH: {worst_month}

DATA NOTES:
- Harbor Plaza: only 3 weeks of data (opened Dec 2021)
- Nairobi Central: data only Jan–Mar 2020, likely closed
- Metro Market: data ends Nov 2021
- Bon Marché Plaza: opened Aug 2021, only 4 months of data
- COVID-19 caused a significant revenue drop in Apr–Jun 2020

90-DAY FORECASTS (Jan–Mar 2022):
- All Sites & Brands: $2,742,612 total
- Crust Co.: $1,688,842
- Flame & Feather: $581,721
- Cala Grill: $313,839
- Frostbite Creamery: $204,011
- Crossroads Mall: $731,583
- Junction Plaza: $722,065
- Piazza Court: $453,255
- Garden Court: $317,608

Answer questions clearly and concisely. Use $ formatting for numbers.
If asked about Harbor Plaza or Nairobi Central forecasts, explain the data limitations.
"""

SYSTEM_PROMPT = build_context()

def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": "user",      "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=500
    )
    return response.choices[0].message.content

def build_dashboard():
    brand_rev = df.groupby('brand')['daily_revenue_usd'].sum().sort_values(ascending=True).reset_index()
    fig1 = go.Figure(go.Bar(
        x=brand_rev['daily_revenue_usd'], y=brand_rev['brand'],
        orientation='h',
        marker_color=['#9b59b6','#e74c3c','#2ecc71','#c9a84c'],
        text=[f"${v:,.0f}" for v in brand_rev['daily_revenue_usd']],
        textposition='outside'
    ))
    fig1.update_layout(
        title="Total Revenue by Brand (2020–2021)",
        xaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f"),
        template="plotly_white", height=300,
        margin=dict(l=140, r=80, t=50, b=40)
    )
    site_rev = df.groupby('site')['daily_revenue_usd'].sum().sort_values(ascending=True).reset_index()
    fig2 = go.Figure(go.Bar(
        x=site_rev['daily_revenue_usd'], y=site_rev['site'],
        orientation='h', marker_color='#1e2d5e',
        text=[f"${v:,.0f}" for v in site_rev['daily_revenue_usd']],
        textposition='outside'
    ))
    fig2.update_layout(
        title="Total Revenue by Site (2020–2021)",
        xaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f"),
        template="plotly_white", height=380,
        margin=dict(l=160, r=80, t=50, b=40)
    )
    monthly = df.groupby(df['date'].dt.to_period('M'))['daily_revenue_usd'].sum().reset_index()
    monthly['date'] = monthly['date'].astype(str)
    fig3 = go.Figure(go.Scatter(
        x=monthly['date'], y=monthly['daily_revenue_usd'],
        mode='lines+markers',
        line=dict(color='#1e2d5e', width=2.5),
        marker=dict(size=5, color='#c9a84c'),
        fill='tozeroy', fillcolor='rgba(30,45,94,0.08)'
    ))
    fig3.add_shape(
        type="rect", x0="2020-03", x1="2020-06",
        y0=0, y1=1, yref="paper",
        fillcolor="red", opacity=0.07, line_width=0
    )
    fig3.add_annotation(
        x="2020-04", y=0.95, yref="paper",
        text="COVID-19", showarrow=False,
        font=dict(color="red", size=11)
    )
    fig3.update_layout(
        title="Monthly Revenue Trend (2020–2021)",
        xaxis=dict(title="Month", tickangle=45),
        yaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f"),
        template="plotly_white", height=380,
        margin=dict(l=60, r=40, t=50, b=80)
    )
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow = df.groupby(df['date'].dt.day_name())['daily_revenue_usd'].mean().reindex(dow_order).reset_index()
    dow.columns = ['day', 'avg_revenue']
    fig4 = go.Figure(go.Bar(
        x=dow['day'], y=dow['avg_revenue'],
        marker_color=['#c9a84c' if d == 'Sunday' else '#1e2d5e' for d in dow['day']],
        text=[f"${v:,.0f}" for v in dow['avg_revenue']],
        textposition='outside'
    ))
    fig4.update_layout(
        title="Average Daily Revenue by Day of Week",
        xaxis_title="Day",
        yaxis=dict(title="Avg Revenue (USD)", tickprefix="$", tickformat=",.0f"),
        template="plotly_white", height=350,
        margin=dict(l=60, r=40, t=50, b=40)
    )
    return fig1, fig2, fig3, fig4

css = """
body, .gradio-container { background: #f8f9fa !important; }
.gr-button-primary { background: #1e2d5e !important; color: white !important; }
h1, h2, h3 { color: #1e2d5e !important; }
"""

with gr.Blocks(title="Continental QSR Intelligence", css=css) as demo:
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1e2d5e,#2a4a8a);
                padding:20px 28px;border-radius:12px;margin-bottom:16px;">
        <h1 style="color:white;margin:0;font-size:26px;">🍔 Continental QSR Intelligence</h1>
        <p style="color:#aed6f1;margin:6px 0 0;font-size:14px;">
            Revenue Analytics & Forecasting | Nairobi, Kenya | 2020–2021
        </p>
        <p style="color:#7fb3d3;margin:4px 0 0;font-size:12px;">
            Powered by Prophet + GPT-4o-mini | Built by Netrisyl Insights
        </p>
    </div>""")

    with gr.Tabs():
        with gr.TabItem("📈 Revenue Forecast"):
            with gr.Row():
                seg_type = gr.Radio(
                    choices=["Overall","By Brand","By Site"],
                    value="Overall", label="Segment Type"
                )
                seg_name = gr.Dropdown(
                    choices=["All Sites & Brands"],
                    value="All Sites & Brands", label="Select Segment"
                )
                horizon = gr.Radio(
                    choices=[30, 60, 90], value=90,
                    label="Forecast Horizon (days)"
                )
                forecast_btn = gr.Button("Generate Forecast", variant="primary")
            forecast_chart   = gr.Plot()
            forecast_summary = gr.Markdown()
            seg_type.change(update_segment_choices, [seg_type], [seg_name])
            forecast_btn.click(
                generate_forecast,
                [seg_type, seg_name, horizon],
                [forecast_chart, forecast_summary]
            )

        with gr.TabItem("💬 Intelligence Chat"):
            gr.Markdown("""
Ask anything about the Continental QSR Group revenue data.

**Example questions:**
- Which brand generates the most revenue?
- What was the impact of COVID-19 on revenue?
- Which site should we invest in next?
- What day of the week has the highest revenue?
- Compare Junction Plaza vs Crossroads Mall
            """)
            chatbot = gr.ChatInterface(
                fn=chat, title="",
                examples=[
                    "Which brand is the top performer?",
                    "What was the COVID-19 impact on revenue?",
                    "Which site should we prioritize for investment?",
                    "What is the revenue forecast for Crust Co.?",
                    "Compare Junction Plaza and Crossroads Mall",
                ]
            )

        with gr.TabItem("📊 Analytics Dashboard"):
            dash_btn = gr.Button("Load Dashboard", variant="primary")
            with gr.Row():
                chart_brand   = gr.Plot()
                chart_site    = gr.Plot()
            with gr.Row():
                chart_monthly = gr.Plot()
                chart_dow     = gr.Plot()
            dash_btn.click(
                build_dashboard, [],
                [chart_brand, chart_site, chart_monthly, chart_dow]
            )

    gr.HTML("""
    <div style="text-align:center;margin-top:16px;color:#7f8c8d;font-size:12px;">
        Netrisyl Insights · Data. Analytics. Intelligence. · netrisyl.com
        <br>⚠️ Data anonymized. All site and brand names are fictional.
    </div>""")

demo.launch(server_name="0.0.0.0", server_port=7860)

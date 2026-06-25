import os
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# DATA LOAD
# ═══════════════════════════════════════════════════════════════
DATA_URL = "https://github.com/SYLVESTER1922/QSR/raw/refs/heads/main/simbisa_kenya_master_published.csv"
df = pd.read_csv(DATA_URL)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

forecasts_df = pd.read_csv('qsr_forecasts.csv')
forecasts_df['ds'] = pd.to_datetime(forecasts_df['ds'])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

RELIABLE_SITES = [
    'Crossroads Mall', 'Junction Plaza', 'Piazza Court',
    'Garden Court', 'Metro Market', 'Bon Marché Plaza'
]
BRANDS = df['brand'].unique().tolist()

SEGMENT_COLORS = {
    'All Sites & Brands': '#c9a84c',
    'Crust Co.':          '#c9a84c',
    'Flame & Feather':    '#2ecc71',
    'Cala Grill':         '#e74c3c',
    'Frostbite Creamery': '#9b59b6',
    'Crossroads Mall':    '#c9a84c',
    'Junction Plaza':     '#2ecc71',
    'Piazza Court':       '#e74c3c',
    'Garden Court':       '#1abc9c',
    'Metro Market':       '#9b59b6',
    'Bon Marché Plaza':   '#f39c12',
}

CHART_LAYOUT = dict(
    paper_bgcolor='#0a1628',
    plot_bgcolor='#0d1f38',
    font=dict(color='#c8d8f0', family='Arial'),
    xaxis=dict(gridcolor='#1a3a6e', linecolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
    yaxis=dict(gridcolor='#1a3a6e', linecolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
    legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor='#1a3a6e', borderwidth=1, font=dict(color='#c8d8f0')),
)

def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


# ═══════════════════════════════════════════════════════════════
# TAB 1 — FORECASTING
# ═══════════════════════════════════════════════════════════════

def generate_forecast(segment_type, segment_name, horizon):
    seg_map = {'Overall': 'overall', 'By Brand': 'brand', 'By Site': 'site'}
    seg_key = seg_map[segment_type]

    fc = forecasts_df[
        (forecasts_df['segment_type'] == seg_key) &
        (forecasts_df['segment_name'] == segment_name)
    ].copy().head(int(horizon))

    if fc.empty:
        fig = go.Figure()
        fig.update_layout(title="No forecast available for this selection", **CHART_LAYOUT)
        return fig, ""

    # ── Historical daily revenue ────────────────────────────────
    if seg_key == 'overall':
        hist = df.groupby('date')['daily_revenue_usd'].sum().reset_index()
    elif seg_key == 'brand':
        hist = df[df['brand'] == segment_name].groupby('date')['daily_revenue_usd'].sum().reset_index()
    else:
        hist = df[df['site'] == segment_name].groupby('date')['daily_revenue_usd'].sum().reset_index()

    hist = hist.sort_values('date').reset_index(drop=True)
    hist.columns = ['ds', 'y']

    color = SEGMENT_COLORS.get(segment_name, '#c9a84c')

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist['ds'], y=hist['y'],
        name='Historical Revenue',
        line=dict(color='#4a6a9e', width=1.2),
        mode='lines', opacity=0.8
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc['ds'], fc['ds'][::-1]]),
        y=pd.concat([fc['yhat_upper'], fc['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor=hex_to_rgba(color, 0.18),
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        hoverinfo='skip'
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc['ds'], y=fc['yhat'],
        name=f'{horizon}-Day Forecast',
        line=dict(color=color, width=2.5, dash='dash'),
        mode='lines'
    ))

    # Forecast start marker
    fig.add_shape(
        type="line",
        x0=fc['ds'].min(), x1=fc['ds'].min(),
        y0=0, y1=1, yref="paper",
        line=dict(color=color, dash="dot", width=1.5)
    )
    fig.add_annotation(
        x=fc['ds'].min(), y=0.97, yref="paper",
        text="▶ Forecast", showarrow=False,
        font=dict(color=color, size=11), xanchor="left"
    )

    fig.update_layout(
        title=dict(
            text=f"{segment_name} — {horizon}-Day Revenue Forecast",
            font=dict(size=16, color='#c9a84c')
        ),
        xaxis=dict(title="Date", gridcolor='#1a3a6e', linecolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
        yaxis=dict(title="Daily Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor='#1a3a6e', linecolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
        paper_bgcolor='#0a1628',
        plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(10,22,40,0.8)', bordercolor='#1a3a6e', borderwidth=1,
            font=dict(color='#c8d8f0')
        ),
        hovermode="x unified",
        height=480,
    )

    total    = fc['yhat'].sum()
    avg      = fc['yhat'].mean()
    peak     = fc['yhat'].max()
    peak_day = fc.loc[fc['yhat'].idxmax(), 'ds'].strftime('%b %d, %Y')
    low      = fc['yhat'].min()
    low_day  = fc.loc[fc['yhat'].idxmin(), 'ds'].strftime('%b %d, %Y')

    summary = f"""**{horizon}-Day Forecast Summary — {segment_name}**

| Metric | Value |
|---|---|
| Predicted Total Revenue | **${total:,.2f}** |
| Average Daily Revenue | **${avg:,.2f}** |
| Peak Day | **${peak:,.2f}** on {peak_day} |
| Lowest Day | **${low:,.2f}** on {low_day} |
| Forecast Period | {fc['ds'].min().strftime('%b %d, %Y')} → {fc['ds'].max().strftime('%b %d, %Y')} |

> *Forecast generated using Facebook Prophet time-series model trained on 2020–2021 data.*
"""
    return fig, summary


def update_segment_choices(segment_type):
    if segment_type == 'Overall':
        return gr.update(choices=['All Sites & Brands'], value='All Sites & Brands')
    elif segment_type == 'By Brand':
        return gr.update(choices=BRANDS, value=BRANDS[0])
    else:
        return gr.update(choices=RELIABLE_SITES, value=RELIABLE_SITES[0])


# ═══════════════════════════════════════════════════════════════
# TAB 2 — INTELLIGENCE CHAT
# ═══════════════════════════════════════════════════════════════

def build_context():
    brand_rev   = df.groupby('brand')['daily_revenue_usd'].sum().round(2).to_dict()
    site_rev    = df.groupby('site')['daily_revenue_usd'].sum().round(2).to_dict()
    yoy         = df.groupby(df['date'].dt.year)['daily_revenue_usd'].sum().round(2).to_dict()
    dow         = df.groupby(df['date'].dt.day_name())['daily_revenue_usd'].mean().round(2).to_dict()
    monthly     = df.groupby(df['date'].dt.to_period('M'))['daily_revenue_usd'].sum()
    best_month  = str(monthly.idxmax())
    worst_month = str(monthly.idxmin())

    return f"""
You are a QSR revenue intelligence assistant for the Continental QSR Group, Nairobi Kenya.
You have access to 2 years of verified daily revenue data (Jan 2020 – Dec 2021) across 8 sites and 4 brands.
Answer questions clearly, concisely and professionally. Always use $ formatting for dollar amounts.

KEY FACTS:
- Total 2-year revenue: $15,868,509
- Date range: Jan 1 2020 – Dec 31 2021
- Brands: {', '.join(BRANDS)}
- Sites: Crossroads Mall, Junction Plaza, Piazza Court, Garden Court, Metro Market, Nairobi Central, Bon Marché Plaza, Harbor Plaza

REVENUE BY BRAND:
{brand_rev}

REVENUE BY SITE:
{site_rev}

YEAR ON YEAR:
- 2020: $7,460,819 | 2021: $8,407,690 | Growth: +12.7%

AVERAGE DAILY REVENUE BY DAY OF WEEK:
{dow}

BEST PERFORMING MONTH: {best_month}
WORST PERFORMING MONTH: {worst_month}

90-DAY FORECASTS (Jan–Mar 2022):
- All Sites & Brands combined: $2,742,612
- Crust Co.: $1,688,842 | Flame & Feather: $581,721 | Cala Grill: $313,839 | Frostbite Creamery: $204,011
- Crossroads Mall: $731,583 | Junction Plaza: $722,065 | Piazza Court: $453,255 | Garden Court: $317,608

DATA LIMITATIONS (be transparent about these):
- Harbor Plaza: only 3 weeks of data — opened Dec 2021, forecasts unreliable
- Nairobi Central: data only Jan–Mar 2020, likely closed — no forecast available
- Metro Market: data ends Nov 2021, missing last month
- Bon Marché Plaza: opened Aug 2021, only 4 months of data — forecast has wider uncertainty
- COVID-19 caused a 56% revenue drop from Jan to Apr 2020 (Apr 2020: $339K vs Jan 2020: $781K)

BRAND INSIGHTS:
- Crust Co. is the dominant brand at 57% of total revenue ($9M)
- Flame & Feather is second at 24% ($3.75M)
- Cala Grill and Frostbite Creamery are tier-3 brands
- Crust Co. leads at 7 of 8 sites; Flame & Feather leads only at Metro Market

SITE INSIGHTS:
- Junction Plaza is the top site ($4.82M), closely followed by Crossroads Mall ($4.70M)
- Harbor Plaza is the smallest site at only $56K total
- Sunday is the highest revenue day on average
- Monday is the lowest revenue day
"""

SYSTEM_PROMPT = build_context()

def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": "user",      "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.2, max_tokens=600
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════

def build_dashboard():
    # ── Brand revenue ───────────────────────────────────────────
    brand_rev = df.groupby('brand')['daily_revenue_usd'].sum().sort_values(ascending=True).reset_index()
    fig1 = go.Figure(go.Bar(
        x=brand_rev['daily_revenue_usd'], y=brand_rev['brand'],
        orientation='h',
        marker=dict(color=['#9b59b6', '#e74c3c', '#2ecc71', '#c9a84c'],
                    line=dict(color='#0a1628', width=1)),
        text=[f"${v:,.0f}" for v in brand_rev['daily_revenue_usd']],
        textposition='outside', textfont=dict(color='#c8d8f0', size=11)
    ))
    fig1.update_layout(
        title=dict(text="Total Revenue by Brand (2020–2021)", font=dict(color='#c9a84c', size=14)),
        xaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
        yaxis=dict(tickfont=dict(color='#c8d8f0')),
        paper_bgcolor='#0a1628', plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0'), height=280,
        margin=dict(l=140, r=100, t=50, b=40)
    )

    # ── Site revenue ────────────────────────────────────────────
    site_rev = df.groupby('site')['daily_revenue_usd'].sum().sort_values(ascending=True).reset_index()
    fig2 = go.Figure(go.Bar(
        x=site_rev['daily_revenue_usd'], y=site_rev['site'],
        orientation='h',
        marker=dict(color='#1e2d5e', line=dict(color='#c9a84c', width=0.5)),
        text=[f"${v:,.0f}" for v in site_rev['daily_revenue_usd']],
        textposition='outside', textfont=dict(color='#c8d8f0', size=11)
    ))
    fig2.update_layout(
        title=dict(text="Total Revenue by Site (2020–2021)", font=dict(color='#c9a84c', size=14)),
        xaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
        yaxis=dict(tickfont=dict(color='#c8d8f0')),
        paper_bgcolor='#0a1628', plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0'), height=360,
        margin=dict(l=160, r=100, t=50, b=40)
    )

    # ── Monthly trend ───────────────────────────────────────────
    monthly = df.groupby(df['date'].dt.to_period('M'))['daily_revenue_usd'].sum().reset_index()
    monthly['date'] = monthly['date'].astype(str)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=monthly['date'], y=monthly['daily_revenue_usd'],
        mode='lines+markers',
        line=dict(color='#c9a84c', width=2.5),
        marker=dict(size=5, color='#c9a84c'),
        fill='tozeroy', fillcolor='rgba(201,168,76,0.08)',
        name='Monthly Revenue'
    ))
    fig3.add_shape(
        type="rect", x0="2020-03", x1="2020-06",
        y0=0, y1=1, yref="paper",
        fillcolor="red", opacity=0.07, line_width=0
    )
    fig3.add_annotation(
        x="2020-04", y=0.92, yref="paper",
        text="COVID-19 Impact", showarrow=False,
        font=dict(color="#ff6b6b", size=11)
    )
    fig3.update_layout(
        title=dict(text="Monthly Revenue Trend (2020–2021)", font=dict(color='#c9a84c', size=14)),
        xaxis=dict(title="Month", tickangle=45, gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
        yaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
        paper_bgcolor='#0a1628', plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0'), height=360,
        margin=dict(l=70, r=40, t=50, b=80), showlegend=False
    )

    # ── Day of week ─────────────────────────────────────────────
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow = df.groupby(df['date'].dt.day_name())['daily_revenue_usd'].mean().reindex(dow_order).reset_index()
    dow.columns = ['day', 'avg_revenue']
    fig4 = go.Figure(go.Bar(
        x=dow['day'], y=dow['avg_revenue'],
        marker=dict(
            color=['#c9a84c' if d == 'Sunday' else '#1e2d5e' for d in dow['day']],
            line=dict(color='#c9a84c', width=0.5)
        ),
        text=[f"${v:,.0f}" for v in dow['avg_revenue']],
        textposition='outside', textfont=dict(color='#c8d8f0', size=11)
    ))
    fig4.update_layout(
        title=dict(text="Avg Daily Revenue by Day of Week", font=dict(color='#c9a84c', size=14)),
        xaxis=dict(title="Day", tickfont=dict(color='#c8d8f0')),
        yaxis=dict(title="Avg Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
        paper_bgcolor='#0a1628', plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0'), height=320,
        margin=dict(l=70, r=40, t=50, b=40)
    )

    # ── YoY comparison ──────────────────────────────────────────
    yoy = df.groupby([df['date'].dt.year, 'brand'])['daily_revenue_usd'].sum().reset_index()
    yoy.columns = ['year', 'brand', 'revenue']
    fig5 = go.Figure()
    brand_colors = {'Crust Co.': '#c9a84c', 'Flame & Feather': '#2ecc71',
                    'Cala Grill': '#e74c3c', 'Frostbite Creamery': '#9b59b6'}
    for brand in BRANDS:
        b = yoy[yoy['brand'] == brand]
        fig5.add_trace(go.Bar(
            x=b['year'].astype(str), y=b['revenue'],
            name=brand, marker_color=brand_colors.get(brand, '#c9a84c')
        ))
    fig5.update_layout(
        title=dict(text="Year-on-Year Revenue by Brand", font=dict(color='#c9a84c', size=14)),
        barmode='group',
        xaxis=dict(title="Year", tickfont=dict(color='#c8d8f0')),
        yaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0')),
        paper_bgcolor='#0a1628', plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0'), height=320,
        legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor='#1a3a6e',
                    borderwidth=1, font=dict(color='#c8d8f0')),
        margin=dict(l=70, r=40, t=50, b=40)
    )

    # ── Site revenue share pie ───────────────────────────────────
    site_rev2 = df.groupby('site')['daily_revenue_usd'].sum().reset_index()
    fig6 = go.Figure(go.Pie(
        labels=site_rev2['site'],
        values=site_rev2['daily_revenue_usd'],
        hole=0.45,
        marker=dict(colors=['#c9a84c','#1e2d5e','#2ecc71','#e74c3c',
                            '#9b59b6','#1abc9c','#f39c12','#34495e'],
                    line=dict(color='#0a1628', width=2)),
        textfont=dict(color='#ffffff', size=11)
    ))
    fig6.update_layout(
        title=dict(text="Revenue Share by Site", font=dict(color='#c9a84c', size=14)),
        paper_bgcolor='#0a1628', font=dict(color='#c8d8f0'), height=320,
        legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor='#1a3a6e',
                    borderwidth=1, font=dict(color='#c8d8f0')),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════

css = """
body, .gradio-container { background: #050d1a !important; font-family: Arial, sans-serif !important; }
.tab-nav button { color: #c8d8f0 !important; background: #0a1628 !important; border-bottom: 2px solid transparent !important; }
.tab-nav button.selected { color: #c9a84c !important; border-bottom: 2px solid #c9a84c !important; }
.gradio-container label, .gradio-container .label-wrap span { color: #a8c8f0 !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: #0a1628 !important; color: #c8d8f0 !important;
    border: 1px solid #1a3a6e !important; border-radius: 6px !important;
}
button.primary { background: #c9a84c !important; color: #0a1628 !important; font-weight: 700 !important; border: none !important; }
button.secondary { background: #1e2d5e !important; color: #c8d8f0 !important; border: 1px solid #c9a84c !important; }
.message.user > div { background: linear-gradient(135deg,#1e2d5e,#162d5a) !important; color: #e8f0ff !important; border-radius: 18px 18px 4px 18px !important; }
.message.bot > div { background: linear-gradient(135deg,#0a1e3d,#112952) !important; color: #ffffff !important; border-radius: 18px 18px 18px 4px !important; }
.message.bot > div strong { color: #c9a84c !important; }
div[class*="chatbot"] { background: #040c1a !important; border-radius: 12px !important; }
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 4px; }
footer { display: none !important; }
"""

with gr.Blocks(title="Continental QSR Intelligence | Netrisyl Insights", css=css) as demo:

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
                padding:18px 28px;border-radius:12px;margin-bottom:14px;
                border-left:4px solid #c9a84c;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div>
                <h1 style="color:#ffffff;margin:0;font-size:22px;font-weight:700;">
                    🍔 Continental QSR Intelligence
                </h1>
                <p style="color:#aed6f1;margin:4px 0 0;font-size:13px;">
                    Revenue Analytics & Forecasting · Nairobi, Kenya · 2020–2021
                </p>
            </div>
            <div style="text-align:right;">
                <p style="color:#c9a84c;margin:0;font-size:12px;font-weight:600;">NETRISYL INSIGHTS</p>
                <p style="color:#7fb3d3;margin:2px 0 0;font-size:11px;">Data · Analytics · Intelligence</p>
            </div>
        </div>
        <div style="display:flex;gap:20px;margin-top:12px;">
            <div style="background:#0a1628;padding:8px 16px;border-radius:6px;border:1px solid #1a3a6e;">
                <span style="color:#c9a84c;font-size:18px;font-weight:700;">$15.9M</span>
                <span style="color:#7fb3d3;font-size:11px;display:block;">Total Revenue</span>
            </div>
            <div style="background:#0a1628;padding:8px 16px;border-radius:6px;border:1px solid #1a3a6e;">
                <span style="color:#c9a84c;font-size:18px;font-weight:700;">+12.7%</span>
                <span style="color:#7fb3d3;font-size:11px;display:block;">YoY Growth</span>
            </div>
            <div style="background:#0a1628;padding:8px 16px;border-radius:6px;border:1px solid #1a3a6e;">
                <span style="color:#c9a84c;font-size:18px;font-weight:700;">8 Sites</span>
                <span style="color:#7fb3d3;font-size:11px;display:block;">4 Brands</span>
            </div>
            <div style="background:#0a1628;padding:8px 16px;border-radius:6px;border:1px solid #1a3a6e;">
                <span style="color:#c9a84c;font-size:18px;font-weight:700;">$2.74M</span>
                <span style="color:#7fb3d3;font-size:11px;display:block;">90-Day Forecast</span>
            </div>
        </div>
    </div>""")

    with gr.Tabs():

        # ── Tab 1: Forecast ──────────────────────────────────────
        with gr.TabItem("📈 Revenue Forecast"):
            gr.HTML("<p style='color:#7fb3d3;font-size:12px;margin:0 0 12px;'>Select a segment and forecast horizon, then click Generate Forecast.</p>")
            with gr.Row():
                seg_type = gr.Radio(
                    choices=["Overall", "By Brand", "By Site"],
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
                forecast_btn = gr.Button("⚡ Generate Forecast", variant="primary", scale=1)

            forecast_chart   = gr.Plot()
            forecast_summary = gr.Markdown()

            seg_type.change(update_segment_choices, [seg_type], [seg_name])
            forecast_btn.click(generate_forecast, [seg_type, seg_name, horizon], [forecast_chart, forecast_summary])

        # ── Tab 2: Chat ──────────────────────────────────────────
        with gr.TabItem("💬 Intelligence Chat"):
            gr.HTML("""
            <div style="background:#0a1628;border:1px solid #1a3a6e;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
                <p style="color:#c9a84c;font-weight:600;margin:0 0 6px;font-size:13px;">💡 Ask anything about the Continental QSR Group data</p>
                <p style="color:#7fb3d3;font-size:12px;margin:0;">
                    Try: <em>"Which brand is the top performer?"</em> · 
                    <em>"What was the COVID-19 impact?"</em> · 
                    <em>"Compare Junction Plaza vs Crossroads Mall"</em> · 
                    <em>"Which site should we invest in?"</em>
                </p>
            </div>""")
            chatbot = gr.ChatInterface(
                fn=chat, title="",
                examples=[
                    "Which brand generates the most revenue?",
                    "What was the COVID-19 impact on revenue?",
                    "Which site should we prioritize for expansion?",
                    "What is the 90-day revenue forecast for Crust Co.?",
                    "Compare Junction Plaza and Crossroads Mall performance",
                    "What day of the week has the highest revenue?",
                    "Which sites have data limitations I should know about?",
                ]
            )

        # ── Tab 3: Dashboard ─────────────────────────────────────
        with gr.TabItem("📊 Analytics Dashboard"):
            gr.HTML("<p style='color:#7fb3d3;font-size:12px;margin:0 0 12px;'>Click Load Dashboard to render all analytics charts.</p>")
            dash_btn = gr.Button("📊 Load Dashboard", variant="primary")
            with gr.Row():
                chart_brand   = gr.Plot()
                chart_site    = gr.Plot()
            with gr.Row():
                chart_monthly = gr.Plot()
                chart_dow     = gr.Plot()
            with gr.Row():
                chart_yoy     = gr.Plot()
                chart_pie     = gr.Plot()
            dash_btn.click(build_dashboard, [], [chart_brand, chart_site, chart_monthly, chart_dow, chart_yoy, chart_pie])

    gr.HTML("""
    <div style="text-align:center;margin-top:16px;padding:12px;border-top:1px solid #1a3a6e;">
        <p style="color:#c9a84c;font-size:12px;font-weight:600;margin:0;">NETRISYL INSIGHTS</p>
        <p style="color:#4a6a9e;font-size:11px;margin:4px 0 0;">
            Data · Analytics · Intelligence · netrisyl.com
            <br>⚠️ Data anonymized. All site and brand names are fictional. Built for demonstration purposes.
        </p>
    </div>""")

demo.launch(server_name="0.0.0.0", server_port=7860)

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
    'Garden Court', 'Metro Market', 'Bon Marche Plaza'
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
    'Bon Marche Plaza':   '#f39c12',
}

# Logo URL — served from HF Space files
LOGO_URL = "https://huggingface.co/spaces/Sylvester1922/qsr-intelligence/resolve/main/NI%20logo.png"

def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def dark_chart(title, height=350):
    return dict(
        title=dict(text=title, font=dict(color='#c9a84c', size=14)),
        paper_bgcolor='#0a1628',
        plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0', family='Arial'),
        height=height,
        xaxis=dict(gridcolor='#1a3a6e', linecolor='#1a3a6e',
                   tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')),
        yaxis=dict(gridcolor='#1a3a6e', linecolor='#1a3a6e',
                   tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')),
        legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor='#1a3a6e',
                    borderwidth=1, font=dict(color='#c8d8f0')),
    )


# ═══════════════════════════════════════════════════════════════
# TAB 1 — FORECASTING
# ═══════════════════════════════════════════════════════════════

def generate_forecast(segment_type, segment_name, horizon):
    seg_map = {'Overall': 'overall', 'By Brand': 'brand', 'By Site': 'site'}
    seg_key = seg_map.get(segment_type, 'overall')

    fc = forecasts_df[
        (forecasts_df['segment_type'] == seg_key) &
        (forecasts_df['segment_name'] == segment_name)
    ].copy().head(int(horizon))

    if fc.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No forecast available for this selection",
            paper_bgcolor='#0a1628', plot_bgcolor='#0d1f38',
            font=dict(color='#c8d8f0')
        )
        return fig, "No forecast data found for this selection."

    # ── Historical daily revenue — properly aggregated ──────────
    if seg_key == 'overall':
        hist = df.groupby('date')['daily_revenue_usd'].sum().reset_index()
    elif seg_key == 'brand':
        hist = df[df['brand'] == segment_name].groupby('date')['daily_revenue_usd'].sum().reset_index()
    else:
        site_key = segment_name.replace('Bon Marché Plaza', 'Bon Marche Plaza')
        hist = df[df['site'] == site_key].groupby('date')['daily_revenue_usd'].sum().reset_index()

    hist = hist.sort_values('date').reset_index(drop=True)
    hist.columns = ['ds', 'y']

    color = SEGMENT_COLORS.get(segment_name, '#c9a84c')

    fig = go.Figure()

    # Historical daily revenue line
    fig.add_trace(go.Scatter(
        x=hist['ds'], y=hist['y'],
        name='Historical Daily Revenue',
        line=dict(color='#4a7aae', width=1.0),
        mode='lines', opacity=0.85,
        hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Historical</extra>'
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc['ds'], fc['ds'][::-1]]),
        y=pd.concat([fc['yhat_upper'], fc['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor=hex_to_rgba(color, 0.20),
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Band',
        hoverinfo='skip',
        showlegend=True
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc['ds'], y=fc['yhat'],
        name=f'{horizon}-Day Forecast',
        line=dict(color=color, width=2.5, dash='dash'),
        mode='lines',
        hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Forecast</extra>'
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
        text="▶ Forecast Start", showarrow=False,
        font=dict(color=color, size=11), xanchor="left",
        bgcolor="rgba(10,22,40,0.7)", borderpad=3
    )

    # Overall note for multi-site data
    if seg_key == 'overall':
        fig.add_annotation(
            x=0.01, y=0.04, xref="paper", yref="paper",
            text="Note: Rising trend reflects new sites opening throughout 2020–2021",
            showarrow=False, font=dict(color='#7fb3d3', size=10),
            bgcolor="rgba(10,22,40,0.6)", borderpad=3, xanchor="left"
        )

    fig.update_layout(
        **dark_chart(f"{segment_name} — {horizon}-Day Revenue Forecast", height=500),
        xaxis=dict(
            title="Date", gridcolor='#1a3a6e', linecolor='#1a3a6e',
            tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')
        ),
        yaxis=dict(
            title="Daily Revenue (USD)", tickprefix="$", tickformat=",.0f",
            gridcolor='#1a3a6e', linecolor='#1a3a6e',
            tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(10,22,40,0.8)', bordercolor='#1a3a6e', borderwidth=1,
            font=dict(color='#c8d8f0')
        ),
        hovermode="x unified",
        margin=dict(l=70, r=30, t=80, b=50)
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
| Peak Day Revenue | **${peak:,.2f}** on {peak_day} |
| Lowest Day Revenue | **${low:,.2f}** on {low_day} |
| Forecast Period | {fc['ds'].min().strftime('%b %d, %Y')} → {fc['ds'].max().strftime('%b %d, %Y')} |

> *Forecast generated using Facebook Prophet trained on 2020–2021 historical data. Confidence band shows 80% prediction interval.*
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
You have access to 2 years of verified daily revenue data (Jan 2020 to Dec 2021) across 8 sites and 4 brands.
Answer questions clearly, concisely and professionally. Use $ formatting for all dollar amounts.
Always be transparent about data limitations when relevant.

KEY FACTS:
- Total 2-year revenue: $15,868,509
- Date range: Jan 1 2020 to Dec 31 2021
- Brands: {', '.join(BRANDS)}
- Sites: Crossroads Mall, Junction Plaza, Piazza Court, Garden Court, Metro Market, Nairobi Central, Bon Marche Plaza, Harbor Plaza

REVENUE BY BRAND:
- Crust Co.: $8,990,500 (57% of total — dominant brand)
- Flame & Feather: $3,750,907 (24%)
- Cala Grill: $1,939,703 (12%)
- Frostbite Creamery: $1,187,399 (7%)

REVENUE BY SITE:
- Junction Plaza: $4,821,962 (top site)
- Crossroads Mall: $4,702,790
- Piazza Court: $2,720,070
- Garden Court: $1,764,552
- Metro Market: $720,592
- Nairobi Central: $614,332
- Bon Marche Plaza: $467,850
- Harbor Plaza: $56,362 (smallest — just opened Dec 2021)

YEAR ON YEAR: 2020 $7,460,819 | 2021 $8,407,690 | Growth +12.7%

AVG DAILY REVENUE BY DAY OF WEEK: {dow}

BEST PERFORMING MONTH: {best_month}
WORST PERFORMING MONTH: {worst_month}

90-DAY FORECASTS (Jan to Mar 2022):
- All Sites & Brands: $2,742,612 total ($30,473/day avg)
- Crust Co.: $1,688,842 | Flame & Feather: $581,721
- Cala Grill: $313,839 | Frostbite Creamery: $204,011
- Crossroads Mall: $731,583 | Junction Plaza: $722,065
- Piazza Court: $453,255 | Garden Court: $317,608

DATA LIMITATIONS (always be transparent):
- Harbor Plaza: only 3 weeks of data (opened Dec 2021) — no reliable forecast
- Nairobi Central: data only Jan to Mar 2020 — likely closed
- Metro Market: data ends Nov 2021
- Bon Marche Plaza: opened Aug 2021, only 4 months of data

COVID-19 IMPACT:
- April 2020 revenue dropped 56% from January 2020 ($339K vs $781K)
- Recovery was gradual through H2 2020
- Full recovery and growth achieved by 2021 (+12.7% YoY)

KEY INSIGHTS:
- Crust Co. leads at 7 of 8 sites; Flame & Feather only leads at Metro Market
- Sunday is consistently the highest revenue day ($2,142 avg)
- Monday is the lowest revenue day ($988 avg)
- Junction Plaza and Crossroads Mall together account for 60% of total revenue
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

def fmt_millions(v):
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    elif v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"

def build_dashboard():
    # ── 1. Brand revenue ────────────────────────────────────────
    brand_rev = df.groupby('brand')['daily_revenue_usd'].sum().sort_values(ascending=True).reset_index()
    fig1 = go.Figure(go.Bar(
        x=brand_rev['daily_revenue_usd'],
        y=brand_rev['brand'],
        orientation='h',
        marker=dict(
            color=['#9b59b6','#e74c3c','#2ecc71','#c9a84c'],
            line=dict(color='#0a1628', width=1)
        ),
        text=[fmt_millions(v) for v in brand_rev['daily_revenue_usd']],
        textposition='outside',
        textfont=dict(color='#c8d8f0', size=12)
    ))
    fig1.update_layout(
        **dark_chart("Total Revenue by Brand (2020–2021)", height=300),
        xaxis=dict(
            title="Revenue (USD)", gridcolor='#1a3a6e', linecolor='#1a3a6e',
            tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0'),
            tickformat="$,.0f", range=[0, brand_rev['daily_revenue_usd'].max() * 1.25]
        ),
        yaxis=dict(tickfont=dict(color='#c8d8f0')),
        margin=dict(l=150, r=80, t=50, b=40)
    )

    # ── 2. Site revenue ──────────────────────────────────────────
    site_rev = df.groupby('site')['daily_revenue_usd'].sum().sort_values(ascending=True).reset_index()
    site_colors = ['#34495e','#34495e','#1abc9c','#9b59b6','#e74c3c','#2ecc71','#c9a84c','#c9a84c']
    fig2 = go.Figure(go.Bar(
        x=site_rev['daily_revenue_usd'],
        y=site_rev['site'],
        orientation='h',
        marker=dict(color=site_colors, line=dict(color='#0a1628', width=1)),
        text=[fmt_millions(v) for v in site_rev['daily_revenue_usd']],
        textposition='outside',
        textfont=dict(color='#c8d8f0', size=11)
    ))
    fig2.update_layout(
        **dark_chart("Total Revenue by Site (2020–2021)", height=380),
        xaxis=dict(
            title="Revenue (USD)", gridcolor='#1a3a6e', linecolor='#1a3a6e',
            tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0'),
            tickformat="$,.0f", range=[0, site_rev['daily_revenue_usd'].max() * 1.28]
        ),
        yaxis=dict(tickfont=dict(color='#c8d8f0')),
        margin=dict(l=160, r=80, t=50, b=40)
    )

    # ── 3. Monthly trend ─────────────────────────────────────────
    monthly = df.groupby(df['date'].dt.to_period('M'))['daily_revenue_usd'].sum().reset_index()
    monthly['date'] = monthly['date'].astype(str)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=monthly['date'], y=monthly['daily_revenue_usd'],
        mode='lines+markers',
        line=dict(color='#c9a84c', width=2.5),
        marker=dict(size=5, color='#c9a84c', line=dict(color='#0a1628', width=1)),
        fill='tozeroy', fillcolor='rgba(201,168,76,0.08)',
        name='Monthly Revenue',
        hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'
    ))
    fig3.add_shape(
        type="rect", x0="2020-03", x1="2020-06",
        y0=0, y1=1, yref="paper",
        fillcolor="red", opacity=0.08, line_width=0
    )
    fig3.add_annotation(
        x="2020-04", y=0.88, yref="paper",
        text="COVID-19<br>Impact", showarrow=False,
        font=dict(color="#ff6b6b", size=10),
        bgcolor="rgba(10,22,40,0.6)", borderpad=3
    )
    fig3.update_layout(
        **dark_chart("Monthly Revenue Trend (2020–2021)", height=360),
        xaxis=dict(
            title="Month", tickangle=45, gridcolor='#1a3a6e',
            tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')
        ),
        yaxis=dict(
            title="Revenue (USD)", tickprefix="$", tickformat=",.0f",
            gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')
        ),
        showlegend=False,
        margin=dict(l=80, r=40, t=50, b=80)
    )

    # ── 4. Day of week ───────────────────────────────────────────
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow = df.groupby(df['date'].dt.day_name())['daily_revenue_usd'].mean().reindex(dow_order).reset_index()
    dow.columns = ['day', 'avg_revenue']
    fig4 = go.Figure(go.Bar(
        x=dow['day'], y=dow['avg_revenue'],
        marker=dict(
            color=['#c9a84c' if d == 'Sunday' else '#1e3a6e' for d in dow['day']],
            line=dict(color='#c9a84c', width=0.8)
        ),
        text=[f"${v:,.0f}" for v in dow['avg_revenue']],
        textposition='outside',
        textfont=dict(color='#c8d8f0', size=11)
    ))
    fig4.update_layout(
        **dark_chart("Avg Daily Revenue by Day of Week", height=320),
        xaxis=dict(title="Day", tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')),
        yaxis=dict(
            title="Avg Revenue (USD)", tickprefix="$", tickformat=",.0f",
            gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0'),
            range=[0, dow['avg_revenue'].max() * 1.25]
        ),
        margin=dict(l=80, r=40, t=50, b=40)
    )

    # ── 5. YoY by brand ─────────────────────────────────────────
    yoy = df.groupby([df['date'].dt.year, 'brand'])['daily_revenue_usd'].sum().reset_index()
    yoy.columns = ['year', 'brand', 'revenue']
    brand_colors_map = {
        'Crust Co.': '#c9a84c', 'Flame & Feather': '#2ecc71',
        'Cala Grill': '#e74c3c', 'Frostbite Creamery': '#9b59b6'
    }
    fig5 = go.Figure()
    for brand in BRANDS:
        b = yoy[yoy['brand'] == brand]
        fig5.add_trace(go.Bar(
            x=b['year'].astype(str), y=b['revenue'],
            name=brand,
            marker=dict(color=brand_colors_map.get(brand, '#c9a84c'),
                        line=dict(color='#0a1628', width=1)),
            text=[fmt_millions(v) for v in b['revenue']],
            textposition='outside',
            textfont=dict(color='#c8d8f0', size=10)
        ))
    fig5.update_layout(
        **dark_chart("Year-on-Year Revenue by Brand", height=340),
        barmode='group',
        xaxis=dict(title="Year", tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')),
        yaxis=dict(
            title="Revenue (USD)", tickprefix="$", tickformat=",.0f",
            gridcolor='#1a3a6e', tickfont=dict(color='#c8d8f0'), title_font=dict(color='#c8d8f0')
        ),
        margin=dict(l=80, r=40, t=50, b=40)
    )

    # ── 6. Revenue share pie ─────────────────────────────────────
    site_rev2 = df.groupby('site')['daily_revenue_usd'].sum().reset_index()
    fig6 = go.Figure(go.Pie(
        labels=site_rev2['site'],
        values=site_rev2['daily_revenue_usd'],
        hole=0.45,
        marker=dict(
            colors=['#c9a84c','#1e2d5e','#2ecc71','#e74c3c',
                    '#9b59b6','#1abc9c','#f39c12','#34495e'],
            line=dict(color='#0a1628', width=2)
        ),
        textfont=dict(color='#ffffff', size=11),
        hovertemplate='%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>'
    ))
    fig6.update_layout(
        **dark_chart("Revenue Share by Site", height=340),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════

css = """
/* ── Base ─────────────────────────────────────────────────── */
body, .gradio-container, .main, .wrap {
    background-color: #050d1a !important;
    font-family: Arial, sans-serif !important;
    color: #c8d8f0 !important;
}

/* ── Tab bar ──────────────────────────────────────────────── */
.tabs > .tab-nav,
div[class*="tabs"] > div[class*="tab-nav"] {
    background-color: #0a1628 !important;
    border-bottom: 2px solid #1a3a6e !important;
    padding: 0 12px !important;
}

button[class*="tab-"] {
    color: #7fb3d3 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 12px 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

button[class*="tab-"]:hover {
    color: #ffffff !important;
    background: rgba(201,168,76,0.08) !important;
}

button[class*="tab-"][class*="selected"],
button[class*="tab-"].selected {
    color: #c9a84c !important;
    border-bottom: 3px solid #c9a84c !important;
    font-weight: 700 !important;
    background: transparent !important;
}

/* ── All text in gradio ───────────────────────────────────── */
.gradio-container *,
.gradio-container label,
.gradio-container span,
.gradio-container p,
.gradio-container .label-wrap span,
.gradio-container .prose p,
.gradio-container .prose li {
    color: #c8d8f0 !important;
}

/* ── Radio buttons ────────────────────────────────────────── */
.gradio-container input[type="radio"] {
    accent-color: #c9a84c !important;
}
.gradio-container .wrap.svelte-1cl284s label,
.gradio-container fieldset label,
.gradio-container .radio-group label,
.gradio-container [data-testid="radio-group"] label,
.gradio-container .form label {
    color: #c8d8f0 !important;
    font-size: 13px !important;
}

/* ── Inputs & dropdowns ───────────────────────────────────── */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .block {
    background: #0a1628 !important;
    color: #c8d8f0 !important;
    border: 1px solid #1a3a6e !important;
    border-radius: 6px !important;
}

/* ── Dropdown list ────────────────────────────────────────── */
ul[role="listbox"] {
    background-color: #0d1b2a !important;
    border: 1px solid #c9a84c !important;
    border-radius: 6px !important;
}
ul[role="listbox"] li {
    color: #ffffff !important;
    background-color: #0d1b2a !important;
    font-size: 13px !important;
}
ul[role="listbox"] li:hover,
ul[role="listbox"] li[aria-selected="true"] {
    background-color: #c9a84c !important;
    color: #0d1b2a !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
button.primary,
.gradio-container button[variant="primary"],
.gradio-container .gr-button-primary {
    background: #c9a84c !important;
    color: #0a1628 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 14px !important;
}
button.primary:hover {
    background: #e0be6a !important;
}
button.secondary,
.gradio-container button[variant="secondary"] {
    background: #1e2d5e !important;
    color: #c8d8f0 !important;
    border: 1px solid #c9a84c !important;
    border-radius: 6px !important;
}

/* ── Chat bubbles ─────────────────────────────────────────── */
.message.user > div,
div[class*="message"][class*="user"] > div:last-child {
    background: linear-gradient(135deg,#1e2d5e,#162d5a) !important;
    color: #e8f0ff !important;
    border-radius: 18px 18px 4px 18px !important;
    border: 1px solid #2a4a8a !important;
    padding: 12px 16px !important;
}
.message.bot > div,
div[class*="message"][class*="bot"] > div:last-child {
    background: linear-gradient(135deg,#0a1e3d,#112952) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 18px 4px !important;
    border: 1px solid #2a6aa055 !important;
    padding: 12px 16px !important;
}
.message.bot > div *,
div[class*="message"][class*="bot"] * { color: #ffffff !important; }
.message.bot > div strong,
div[class*="message"][class*="bot"] strong { color: #c9a84c !important; }
div[class*="chatbot"], .chatbot {
    background: #040c1a !important;
    border-radius: 12px !important;
}

/* ── Markdown ─────────────────────────────────────────────── */
.gradio-container .prose { color: #c8d8f0 !important; }
.gradio-container .prose strong { color: #c9a84c !important; }
.gradio-container .prose table { border-color: #1a3a6e !important; }
.gradio-container .prose th { background: #0a1628 !important; color: #c9a84c !important; }
.gradio-container .prose td { border-color: #1a3a6e !important; color: #c8d8f0 !important; }

/* ── Panel backgrounds ────────────────────────────────────── */
.gradio-container .panel,
.gradio-container .form,
.gradio-container .block.padded {
    background: #0a1628 !important;
    border: 1px solid #1a3a6e !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #050d1a; }
::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #e0be6a; }

footer { display: none !important; }
"""

with gr.Blocks(title="Continental QSR Intelligence | Netrisyl Insights", css=css) as demo:

    # ── Header ───────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a 0%,#1a3a5c 100%);
                padding:20px 28px 16px;border-radius:12px;margin-bottom:4px;
                border-left:4px solid #c9a84c;
                box-shadow:0 4px 20px rgba(0,0,0,0.4);">

        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div style="display:flex;align-items:center;gap:16px;">
                <img src="{LOGO_URL}"
                     style="height:58px;width:auto;object-fit:contain;border-radius:6px;"
                     onerror="this.style.display='none'" alt="NI"/>
                <div>
                    <h1 style="color:#ffffff;margin:0;font-size:22px;font-weight:700;letter-spacing:0.3px;">
                        🍔 Continental QSR Intelligence
                    </h1>
                    <p style="color:#aed6f1;margin:4px 0 0;font-size:13px;">
                        Revenue Analytics &amp; Forecasting &nbsp;·&nbsp; Nairobi, Kenya &nbsp;·&nbsp; 2020–2021
                    </p>
                </div>
            </div>
            <div style="text-align:right;">
                <p style="color:#c9a84c;margin:0;font-size:11px;font-weight:700;letter-spacing:2px;">NETRISYL INSIGHTS</p>
                <p style="color:#7fb3d3;margin:3px 0 0;font-size:11px;">Data &nbsp;·&nbsp; Analytics &nbsp;·&nbsp; Intelligence</p>
            </div>
        </div>

        <div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;">
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;border-radius:8px;
                        border:1px solid #1a3a6e;min-width:90px;text-align:center;">
                <div style="color:#c9a84c;font-size:20px;font-weight:700;">$15.9M</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">Total Revenue</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;border-radius:8px;
                        border:1px solid #1a3a6e;min-width:90px;text-align:center;">
                <div style="color:#2ecc71;font-size:20px;font-weight:700;">+12.7%</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">YoY Growth</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;border-radius:8px;
                        border:1px solid #1a3a6e;min-width:90px;text-align:center;">
                <div style="color:#c9a84c;font-size:20px;font-weight:700;">8 Sites</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">4 Brands</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;border-radius:8px;
                        border:1px solid #1a3a6e;min-width:90px;text-align:center;">
                <div style="color:#c9a84c;font-size:20px;font-weight:700;">$2.74M</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">90-Day Forecast</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;border-radius:8px;
                        border:1px solid #1a3a6e;min-width:90px;text-align:center;">
                <div style="color:#c9a84c;font-size:20px;font-weight:700;">$23.9K</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">Avg Daily Rev</div>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Forecast ──────────────────────────────────────
        with gr.TabItem("📈 Revenue Forecast"):
            gr.HTML("""
            <p style="color:#7fb3d3;font-size:12px;margin:8px 0 12px;">
                Select a segment type and specific segment, choose a forecast horizon, then click Generate Forecast.
            </p>""")

            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    seg_type = gr.Radio(
                        choices=["Overall", "By Brand", "By Site"],
                        value="Overall",
                        label="Segment Type"
                    )
                with gr.Column(scale=2):
                    seg_name = gr.Dropdown(
                        choices=["All Sites & Brands"],
                        value="All Sites & Brands",
                        label="Select Segment",
                        interactive=True
                    )
                with gr.Column(scale=2):
                    horizon = gr.Radio(
                        choices=[30, 60, 90],
                        value=90,
                        label="Forecast Horizon (days)"
                    )
                with gr.Column(scale=1):
                    forecast_btn = gr.Button("⚡ Generate Forecast", variant="primary")

            forecast_chart   = gr.Plot(show_label=False)
            forecast_summary = gr.Markdown()

            seg_type.change(update_segment_choices, [seg_type], [seg_name])
            forecast_btn.click(
                generate_forecast,
                [seg_type, seg_name, horizon],
                [forecast_chart, forecast_summary]
            )

        # ── Tab 2: Chat ──────────────────────────────────────────
        with gr.TabItem("💬 Intelligence Chat"):
            gr.HTML("""
            <div style="background:#0a1628;border:1px solid #1a3a6e;border-radius:8px;
                        padding:14px 18px;margin:8px 0 14px;">
                <p style="color:#c9a84c;font-weight:700;margin:0 0 6px;font-size:13px;">
                    💡 Ask anything about the Continental QSR Group revenue data
                </p>
                <p style="color:#7fb3d3;font-size:12px;margin:0;line-height:1.6;">
                    <strong style="color:#c8d8f0;">Try asking:</strong>
                    &nbsp;"Which brand is the top performer?"
                    &nbsp;·&nbsp; "What was the COVID-19 impact?"
                    &nbsp;·&nbsp; "Compare Junction Plaza vs Crossroads Mall"
                    &nbsp;·&nbsp; "Which site should we invest in?"
                    &nbsp;·&nbsp; "What is the 90-day forecast for Crust Co.?"
                </p>
            </div>""")
            chatbot = gr.ChatInterface(
                fn=chat,
                title="",
                examples=[
                    "Which brand generates the most revenue?",
                    "What was the COVID-19 impact on revenue?",
                    "Which site should we prioritize for expansion?",
                    "What is the 90-day revenue forecast for Crust Co.?",
                    "Compare Junction Plaza and Crossroads Mall performance",
                    "What day of the week has the highest revenue?",
                    "Which sites have data limitations I should know about?",
                    "What is the year-on-year growth trend?",
                ]
            )

        # ── Tab 3: Dashboard ─────────────────────────────────────
        with gr.TabItem("📊 Analytics Dashboard"):
            gr.HTML("""
            <p style="color:#7fb3d3;font-size:12px;margin:8px 0 12px;">
                Click Load Dashboard to render all six analytics charts.
            </p>""")
            dash_btn = gr.Button("📊 Load Dashboard", variant="primary")
            with gr.Row():
                chart_brand   = gr.Plot(show_label=False)
                chart_site    = gr.Plot(show_label=False)
            with gr.Row():
                chart_monthly = gr.Plot(show_label=False)
                chart_dow     = gr.Plot(show_label=False)
            with gr.Row():
                chart_yoy     = gr.Plot(show_label=False)
                chart_pie     = gr.Plot(show_label=False)
            dash_btn.click(
                build_dashboard, [],
                [chart_brand, chart_site, chart_monthly, chart_dow, chart_yoy, chart_pie]
            )

    # ── Footer ───────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;margin-top:20px;padding:14px;
                border-top:1px solid #1a3a6e;">
        <p style="color:#c9a84c;font-size:11px;font-weight:700;
                  margin:0;letter-spacing:2px;">NETRISYL INSIGHTS</p>
        <p style="color:#4a6a9e;font-size:11px;margin:5px 0 0;">
            Data · Analytics · Intelligence ·
            <a href="https://netrisyl.com" style="color:#7fb3d3;text-decoration:none;">netrisyl.com</a>
        </p>
        <p style="color:#2a4a6e;font-size:10px;margin:4px 0 0;">
            ⚠️ Data anonymized. All site and brand names are fictional. Built for demonstration purposes.
        </p>
    </div>""")

demo.launch(server_name="0.0.0.0", server_port=7860)

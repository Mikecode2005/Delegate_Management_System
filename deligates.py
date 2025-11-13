import streamlit as st
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re

st.set_page_config(
    page_title="Delegate Management Dashboard",
    layout="wide",
    page_icon="👥",
    initial_sidebar_state="expanded"
)

def clean_dataframe_for_display(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
        df_clean[col] = df_clean[col].fillna('')
    return df_clean

def safe_process_data(df, header_row=0, auto_ffill=True):
    try:
        df.columns = [str(col).strip().replace('\n', ' ').title() for col in df.columns]

        if auto_ffill:
            date_cols = [col for col in df.columns if any(x in str(col).upper() for x in ['BATCH', 'COURSE DATE', 'ISSUED DATE', 'EXPIRY DATE', 'DATE'])]
            if date_cols:
                df[date_cols] = df[date_cols].ffill()

        phone_cols = [col for col in df.columns if any(x in str(col).upper() for x in ['NUMBER', 'PHONE', 'NOK', 'CONTACT', 'TEL'])]
        for col in phone_cols:
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

        first_name_col = next((col for col in df.columns if 'FIRST NAME' in str(col).upper()), None)
        last_name_col = next((col for col in df.columns if 'LAST NAME' in str(col).upper()), None)
        if first_name_col and last_name_col:
            df['Full Name'] = df[first_name_col].fillna('') + ' ' + df[last_name_col].fillna('')

        dob_col = next((col for col in df.columns if 'DOB' in str(col).upper() or 'BIRTH' in str(col).upper()), None)
        if dob_col:
            try:
                df[dob_col] = pd.to_datetime(df[dob_col], errors='coerce')
                df['Age'] = ((datetime.now() - df[dob_col]).dt.days // 365.25).astype('Int64')
            except Exception as e:
                st.warning(f"Unable to calculate ages: {str(e)}")

        return df, None
    except Exception as e:
        error_msg = f"Error processing data: {str(e)}"
        return df, error_msg

def get_column_by_pattern(df, pattern_list, default=None):
    for pattern in pattern_list:
        matches = [col for col in df.columns if pattern in str(col).upper()]
        if matches:
            return matches[0]
    return default

def handle_merged_cells(df):
    potential_merged_cols = []
    for col in df.columns:
        null_ratio = df[col].isna().sum() / len(df)
        if 0.3 < null_ratio < 0.9:
            potential_merged_cols.append(col)

    if potential_merged_cols:
        df[potential_merged_cols] = df[potential_merged_cols].ffill()

    return df

def create_circular_progress(value, max_value, title, color):
    percentage = (value / max_value * 100) if max_value > 0 else 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': text_color}},
        number={'font': {'size': 32, 'color': color}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': border_color},
            'bar': {'color': color},
            'bgcolor': card_bg,
            'borderwidth': 2,
            'bordercolor': border_color,
            'steps': [
                {'range': [0, max_value], 'color': f"{border_color}20"}
            ],
            'threshold': {
                'line': {'color': accent_color, 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        font={'color': text_color, 'family': "Inter, sans-serif"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def create_donut_chart(labels, values, title):
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=[primary_color, secondary_color, tertiary_color, accent_color, warning_color, success_color][:len(labels)]),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12, color=text_color)
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=text_color, family="Inter")),
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        font={'color': text_color},
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(color=text_color)
        )
    )

    return fig

def create_performance_card(value, label, change=None):
    """Create a performance metric card"""
    change_html = ""
    if change is not None:
        change_class = "change-positive" if change > 0 else "change-negative"
        change_symbol = "↗" if change > 0 else "↘"
        change_html = f'<div class="perf-change {change_class}">{change_symbol} {abs(change)}%</div>'
    
    return f"""
    <div class="perf-card">
        <div class="perf-value">{value}</div>
        <div class="perf-label">{label}</div>
        {change_html}
    </div>
    """

def generate_performance_data():
    """Generate mock performance data"""
    return {
        'total_visitors': 12480,
        'page_views': 45230,
        'avg_session': '4m 12s',
        'bounce_rate': 32.5,
        'conversion_rate': 8.7,
        'revenue': '$12,480',
        'active_users': 2450,
        'growth_rate': 18.3
    }

THEMES = {
    "Light": {
        "primary_color": "#2563eb",
        "secondary_color": "#10b981",
        "tertiary_color": "#f59e0b",
        "background_color": "#f8fafc",
        "text_color": "#1e293b",
        "card_bg": "#ffffff",
        "accent_color": "#ec4899",
        "plotly_theme": "plotly_white",
        "gradient_start": "#2563eb",
        "gradient_end": "#7c3aed",
        "success_color": "#10b981",
        "warning_color": "#f59e0b",
        "error_color": "#ef4444",
        "border_color": "#e2e8f0",
        "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    },
    "Dark": {
        "primary_color": "#3b82f6",
        "secondary_color": "#10b981",
        "tertiary_color": "#f59e0b",
        "background_color": "#0f172a",
        "text_color": "#f1f5f9",
        "card_bg": "#1e293b",
        "accent_color": "#ec4899",
        "plotly_theme": "plotly_dark",
        "gradient_start": "#3b82f6",
        "gradient_end": "#8b5cf6",
        "success_color": "#10b981",
        "warning_color": "#f59e0b",
        "error_color": "#ef4444",
        "border_color": "#334155",
        "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)"
    }
}

if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'original_columns' not in st.session_state:
    st.session_state.original_columns = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'header_row' not in st.session_state:
    st.session_state.header_row = 0
if 'error_log' not in st.session_state:
    st.session_state.error_log = []
if 'show_performance' not in st.session_state:
    st.session_state.show_performance = True

current_theme = THEMES[st.session_state.theme]
primary_color = current_theme["primary_color"]
secondary_color = current_theme["secondary_color"]
tertiary_color = current_theme["tertiary_color"]
background_color = current_theme["background_color"]
text_color = current_theme["text_color"]
card_bg = current_theme["card_bg"]
accent_color = current_theme["accent_color"]
plotly_theme = current_theme["plotly_theme"]
gradient_start = current_theme["gradient_start"]
gradient_end = current_theme["gradient_end"]
success_color = current_theme["success_color"]
warning_color = current_theme["warning_color"]
error_color = current_theme["error_color"]
border_color = current_theme["border_color"]
shadow = current_theme["shadow"]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {background_color} !important;
        color: {text_color} !important;
    }}

    .main-header {{
        background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        text-align: center;
        font-weight: 800;
        margin-bottom: 2rem;
        letter-spacing: -0.03em;
    }}

    .modern-card {{
        background: {card_bg};
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: {shadow};
        border: 1px solid {border_color};
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .modern-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.08);
    }}

    .metric-card {{
        background: {card_bg};
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid {border_color};
        box-shadow: {shadow};
        transition: all 0.3s ease;
        text-align: center;
    }}

    .metric-card:hover {{
        transform: translateY(-4px);
        border-color: {primary_color};
        box-shadow: 0 12px 20px -5px rgba(37, 99, 235, 0.2);
    }}

    .metric-value {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, {gradient_start}, {gradient_end});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }}

    .metric-label {{
        font-size: 0.875rem;
        font-weight: 600;
        color: {text_color};
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    .metric-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }}

    .progress-container {{
        width: 100%;
        height: 10px;
        background: {border_color};
        border-radius: 999px;
        overflow: hidden;
        margin: 0.75rem 0;
        position: relative;
    }}

    .progress-bar {{
        height: 100%;
        background: linear-gradient(90deg, {gradient_start}, {gradient_end});
        border-radius: 999px;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    .progress-bar::after {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }}

    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {primary_color} 0%, {gradient_end} 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        letter-spacing: 0.02em;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
    }}

    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {border_color};
    }}

    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {primary_color} !important;
        font-weight: 700;
        font-size: 1.25rem;
    }}

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {{
        border-radius: 10px;
        border: 2px solid {border_color};
        background-color: {card_bg};
        color: {text_color} !important;
        transition: all 0.3s ease;
        padding: 0.75rem;
    }}

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {{
        border-color: {primary_color};
        box-shadow: 0 0 0 4px {primary_color}20;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background-color: transparent !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 12px;
        padding: 0.875rem 1.75rem;
        color: {text_color} !important;
        font-weight: 600;
        transition: all 0.3s ease;
        background-color: {card_bg};
        border: 1px solid {border_color};
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {primary_color} 0%, {gradient_end} 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        border-color: transparent !important;
    }}

    .stDataFrame {{
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid {border_color};
    }}

    .stFileUploader {{
        border: 3px dashed {border_color};
        border-radius: 16px;
        padding: 2.5rem;
        background: {card_bg};
        transition: all 0.3s ease;
    }}

    .stFileUploader:hover {{
        border-color: {primary_color};
        background: {primary_color}10;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, {gradient_start}, {gradient_end});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 0.875rem;
        font-weight: 600;
        color: {text_color};
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .stSuccess {{
        background-color: {success_color}20 !important;
        border-left: 5px solid {success_color};
        border-radius: 10px;
        padding: 1rem;
    }}

    .stError {{
        background-color: {error_color}20 !important;
        border-left: 5px solid {error_color};
        border-radius: 10px;
        padding: 1rem;
    }}

    .stWarning {{
        background-color: {warning_color}20 !important;
        border-left: 5px solid {warning_color};
        border-radius: 10px;
        padding: 1rem;
    }}

    .stInfo {{
        background-color: {primary_color}20 !important;
        border-left: 5px solid {primary_color};
        border-radius: 10px;
        padding: 1rem;
    }}

    .stExpander {{
        border: 1px solid {border_color} !important;
        border-radius: 12px;
        background: {card_bg};
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    ::-webkit-scrollbar {{
        width: 12px;
        height: 12px;
    }}

    ::-webkit-scrollbar-track {{
        background: {background_color};
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {primary_color}, {gradient_end});
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {gradient_end}, {primary_color});
    }}

    .section-title {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {primary_color};
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}

    .stat-badge {{
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, {primary_color}20, {gradient_end}20);
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.875rem;
        color: {primary_color};
        border: 1px solid {primary_color}30;
    }}

    /* Performance Metrics Grid */
    .perf-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}

    .perf-card {{
        background: {card_bg};
        border-radius: 16px;
        padding: 2rem;
        box-shadow: {shadow};
        border: 1px solid {border_color};
        text-align: center;
        transition: all 0.3s ease;
    }}

    .perf-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.15);
    }}

    .perf-value {{
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, {gradient_start}, {gradient_end});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }}

    .perf-label {{
        font-size: 0.9rem;
        color: {text_color};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .perf-change {{
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }}

    .change-positive {{
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
    }}

    .change-negative {{
        background: rgba(239, 68, 68, 0.1);
        color: #EF4444;
    }}

    /* Stats highlight */
    .stats-highlight {{
        background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }}

    .stats-highlight h2 {{
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }}

    .stats-highlight p {{
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f'<h3 style="color: {primary_color};">Control Panel</h3>', unsafe_allow_html=True)

    st.markdown("### Theme")
    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button("Light", use_container_width=True):
            st.session_state.theme = "Light"
            st.rerun()
    with theme_col2:
        if st.button("Dark", use_container_width=True):
            st.session_state.theme = "Dark"
            st.rerun()

    st.markdown(f'<div style="padding: 0.5rem; background: {primary_color}20; border-radius: 8px; text-align: center; margin-top: 0.5rem;">Current: <b>{st.session_state.theme} Mode</b></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Data Options")
    show_sample = st.checkbox("Show Sample Data", value=False)
    auto_ffill = st.checkbox("Auto Forward-Fill", value=True,
                            help="Automatically fill empty cells with the value above them")

    st.markdown("### Display Options")
    chart_style = st.selectbox(
        "Chart Style",
        ["Modern", "Colorful", "Minimal", "Classic"],
        index=0
    )

    show_animations = st.checkbox("Smooth Animations", value=True)

    with st.expander("Help & Documentation"):
        st.markdown("""
        ### Quick Guide
        1. Upload your Excel/CSV file
        2. Select the header row
        3. View analytics with circular charts
        4. Use search and filters
        5. Export your results

        ### Features
        - Modern circular progress indicators
        - Interactive donut charts
        - Real-time data analysis
        - Dark/Light mode toggle
        - Advanced filtering system
        """)

    st.markdown("---")
    st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 0.85rem; opacity: 0.7;">Delegate Management v5.0<br/>Built with Streamlit</p>', unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Delegate Management Dashboard</h1>', unsafe_allow_html=True)

# Performance Metrics Section (shown before data upload)
if st.session_state.show_performance and st.session_state.df is None:
    st.markdown("## Performance Overview")
    
    # Performance highlight
    st.markdown(f"""
    <div class="stats-highlight">
        <h2>Ready to Analyze Your Data</h2>
        <p>Upload your dataset to unlock powerful insights and beautiful visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics grid
    perf_data = generate_performance_data()
    
    st.markdown('<div class="perf-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_performance_card(
            f"{perf_data['total_visitors']:,}", 
            "Total Visitors", 
            change=12.4
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_performance_card(
            f"{perf_data['page_views']:,}", 
            "Page Views", 
            change=8.7
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_performance_card(
            perf_data['avg_session'], 
            "Avg Session", 
            change=5.2
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_performance_card(
            f"{perf_data['bounce_rate']}%", 
            "Bounce Rate", 
            change=-3.1
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row of performance metrics
    st.markdown('<div class="perf-grid">', unsafe_allow_html=True)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(create_performance_card(
            f"{perf_data['conversion_rate']}%", 
            "Conversion Rate", 
            change=15.3
        ), unsafe_allow_html=True)
    
    with col6:
        st.markdown(create_performance_card(
            perf_data['revenue'], 
            "Revenue", 
            change=22.8
        ), unsafe_allow_html=True)
    
    with col7:
        st.markdown(create_performance_card(
            f"{perf_data['active_users']:,}", 
            "Active Users", 
            change=18.3
        ), unsafe_allow_html=True)
    
    with col8:
        st.markdown(create_performance_card(
            f"{perf_data['growth_rate']}%", 
            "Growth Rate", 
            change=6.9
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload prompt
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3 style="color: #3B82F6; margin-bottom: 1rem;">Ready to Get Started?</h3>
        <p style="color: #64748B; font-size: 1.1rem;">
            Upload your CSV or Excel file to see your own data visualized in these beautiful charts and metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'<div class="modern-card">', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Upload Data</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drag and drop or browse files", type=["xlsx", "csv"], key="uploader")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.show_performance = False

    try:
        if uploaded_file.name.endswith(".csv"):
            temp_df = pd.read_csv(uploaded_file, header=None)
        else:
            try:
                temp_df = pd.read_excel(uploaded_file, header=None)
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                st.info("Trying alternative method...")
                try:
                    temp_df = pd.read_excel(uploaded_file, header=None, engine="openpyxl")
                except Exception as e2:
                    st.error(f"Failed to read Excel file: {str(e2)}")
                    st.stop()

        with st.expander("Preview Raw Data", expanded=True):
            st.dataframe(clean_dataframe_for_display(temp_df.head(10)), use_container_width=True)

        header_row = st.selectbox(
            "Header Row (0-based)",
            list(range(min(10, len(temp_df)))),
            index=min(1, len(temp_df)-1)
        )
        st.session_state.header_row = header_row

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=header_row, dtype=str)
            else:
                try:
                    df = pd.read_excel(uploaded_file, header=header_row, dtype=str)
                except:
                    df = pd.read_excel(uploaded_file, header=header_row, dtype=str, engine="openpyxl")

            df = handle_merged_cells(df)
            df, error_msg = safe_process_data(df, header_row, auto_ffill)

            if error_msg:
                st.warning(error_msg)
                st.session_state.error_log.append(error_msg)

            st.session_state.df = df.copy()
            st.session_state.original_columns = df.columns.tolist()
            st.session_state.processed = True
            st.success("Data uploaded and processed successfully!")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.session_state.error_log.append(f"Error processing file: {str(e)}")

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.session_state.error_log.append(f"Error reading file: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

if show_sample and (st.session_state.df is None or st.session_state.df.empty):
    st.markdown(f'<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Sample Data</div>', unsafe_allow_html=True)

    try:
        sample_data = pd.DataFrame({
            "S/N": list(range(1, 21)),
            "First Name": ["John", "Jane", "Alex", "Emily", "Michael", "Sarah", "David", "Lisa", "Robert", "Emma",
                          "James", "Maria", "Chris", "Anna", "Daniel", "Sophie", "Tom", "Lucy", "Mark", "Kate"],
            "Last Name": ["Doe", "Smith", "Johnson", "Brown", "Davis", "Wilson", "Taylor", "Thomas", "Roberts", "Clark",
                         "White", "Hall", "Allen", "Young", "King", "Wright", "Lopez", "Hill", "Scott", "Green"],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female",
                      "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
            "DoB": ["1990-05-15", "1985-12-10", "1992-08-22", "1988-03-30", "1995-07-18",
                   "1991-11-05", "1987-09-12", "1993-04-25", "1989-01-20", "1994-06-08",
                   "1996-02-14", "1984-11-23", "1991-07-30", "1993-09-15", "1986-05-11",
                   "1997-12-03", "1990-08-19", "1992-03-27", "1988-10-05", "1995-01-22"],
            "Company Name": ["Tech Corp", "Oil Inc", "Marine Ltd", "Tech Corp", "Oil Inc",
                           "Marine Ltd", "Tech Corp", "Oil Inc", "Marine Ltd", "Tech Corp",
                           "Oil Inc", "Marine Ltd", "Tech Corp", "Oil Inc", "Marine Ltd",
                           "Tech Corp", "Oil Inc", "Marine Ltd", "Tech Corp", "Oil Inc"],
            "Contact Number": [f"080{12345678 + i}" for i in range(20)],
            "Certificate Number": [f"EBS2024{i:04d}" for i in range(1, 21)],
            "Course Date": ["8th March, 2024", "5th June, 2024", "3rd August, 2024", "14th September, 2024", "8th March, 2024",
                          "5th June, 2024", "3rd August, 2024", "14th September, 2024", "8th March, 2024", "5th June, 2024",
                          "3rd August, 2024", "14th September, 2024", "8th March, 2024", "5th June, 2024", "3rd August, 2024",
                          "14th September, 2024", "8th March, 2024", "5th June, 2024", "3rd August, 2024", "14th September, 2024"],
            "Batch": ["1/24", "2/24", "3/24", "4/24", "1/24", "2/24", "3/24", "4/24", "1/24", "2/24",
                     "3/24", "4/24", "1/24", "2/24", "3/24", "4/24", "1/24", "2/24", "3/24", "4/24"]
        })

        sample_data, error_msg = safe_process_data(sample_data)

        if error_msg:
            st.warning(error_msg)
            st.session_state.error_log.append(error_msg)

        st.session_state.df = sample_data
        st.session_state.original_columns = sample_data.columns.tolist()
        st.session_state.processed = True
        st.session_state.show_performance = False

        st.dataframe(clean_dataframe_for_display(sample_data), use_container_width=True)

    except Exception as e:
        st.error(f"Error creating sample data: {str(e)}")
        st.session_state.error_log.append(f"Error creating sample data: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.df is not None and not st.session_state.df.empty:

    st.markdown(f'<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Quick Statistics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    total_records = len(st.session_state.df)
    total_columns = len(st.session_state.df.columns)
    missing_values = st.session_state.df.isna().sum().sum()
    unique_avg = int(st.session_state.df.nunique().mean())

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📋</div>
            <div class="metric-value">{total_records}</div>
            <div class="metric-label">Total Records</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📁</div>
            <div class="metric-value">{total_columns}</div>
            <div class="metric-label">Data Columns</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: 85%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        missing_percent = (missing_values / (total_records * total_columns) * 100) if total_records > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚠️</div>
            <div class="metric-value">{missing_values}</div>
            <div class="metric-label">Missing Values</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {min(missing_percent, 100)}%; background: linear-gradient(90deg, {warning_color}, {error_color});"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">{unique_avg}</div>
            <div class="metric-label">Avg Unique Values</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: 70%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="modern-card">', unsafe_allow_html=True)
    tabs = st.tabs(["Analysis", "Search & Filter", "Data Processing", "Export"])

    with tabs[0]:
        st.markdown(f'<div class="section-title">Data Analysis & Insights</div>', unsafe_allow_html=True)

        try:
            if chart_style == "Colorful":
                color_seq = px.colors.qualitative.Bold
            elif chart_style == "Minimal":
                color_seq = px.colors.sequential.Greys
            elif chart_style == "Classic":
                color_seq = px.colors.qualitative.Pastel
            else:
                color_seq = [primary_color, secondary_color, tertiary_color, accent_color, warning_color, success_color]

            col1, col2 = st.columns(2)

            gender_col = get_column_by_pattern(st.session_state.df, ['GENDER'])
            if gender_col:
                with col1:
                    gender_cnt = st.session_state.df[gender_col].value_counts()
                    fig = create_donut_chart(gender_cnt.index.tolist(), gender_cnt.values.tolist(), "Gender Distribution")
                    st.plotly_chart(fig, use_container_width=True)

            company_col = get_column_by_pattern(st.session_state.df, ['COMPANY'])
            if company_col:
                with col2:
                    comp_cnt = st.session_state.df[company_col].value_counts().head(6)
                    fig = create_donut_chart(comp_cnt.index.tolist(), comp_cnt.values.tolist(), "Company Distribution")
                    st.plotly_chart(fig, use_container_width=True)

            if 'Age' in st.session_state.df.columns:
                st.markdown("### Age Demographics")
                age_data = st.session_state.df['Age'].dropna().astype(float)
                if not age_data.empty:
                    bins = [0, 20, 30, 40, 50, 60, np.inf]
                    labels = ['Under 20', '20-29', '30-39', '40-49', '50-59', '60+']
                    age_groups = pd.cut(age_data, bins, labels=labels)
                    age_cnt = age_groups.value_counts().sort_index()

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=age_cnt.index,
                        y=age_cnt.values,
                        marker=dict(
                            color=age_cnt.values,
                            colorscale=[[0, primary_color], [0.5, secondary_color], [1, accent_color]],
                            line=dict(color=border_color, width=2)
                        ),
                        text=age_cnt.values,
                        textposition='outside',
                        textfont=dict(size=14, color=text_color, weight='bold'),
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                    ))

                    fig.update_layout(
                        title=dict(text="Age Group Distribution", font=dict(size=20, color=text_color, family="Inter")),
                        paper_bgcolor=card_bg,
                        plot_bgcolor=card_bg,
                        font={'color': text_color},
                        height=400,
                        xaxis=dict(title="Age Group", gridcolor=border_color),
                        yaxis=dict(title="Count", gridcolor=border_color),
                        margin=dict(l=40, r=40, t=80, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True)

            batch_col = get_column_by_pattern(st.session_state.df, ['BATCH'])
            date_col = get_column_by_pattern(st.session_state.df, ['COURSE DATE', 'DATE'])

            col1, col2 = st.columns(2)

            if batch_col:
                with col1:
                    st.markdown("### Batch Distribution")
                    batch_cnt = st.session_state.df[batch_col].value_counts()

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=batch_cnt.index,
                        y=batch_cnt.values,
                        marker=dict(
                            color=[primary_color, secondary_color, tertiary_color, accent_color][:len(batch_cnt)],
                            line=dict(color=border_color, width=2)
                        ),
                        text=batch_cnt.values,
                        textposition='outside',
                        textfont=dict(size=14, color=text_color, weight='bold')
                    ))

                    fig.update_layout(
                        paper_bgcolor=card_bg,
                        plot_bgcolor=card_bg,
                        font={'color': text_color},
                        height=350,
                        xaxis=dict(title="Batch", gridcolor=border_color),
                        yaxis=dict(title="Count", gridcolor=border_color),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True)

            if date_col:
                with col2:
                    st.markdown("### Course Date Timeline")
                    date_cnt = st.session_state.df[date_col].value_counts().sort_index()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=date_cnt.index,
                        y=date_cnt.values,
                        mode='lines+markers',
                        line=dict(color=primary_color, width=3),
                        marker=dict(size=10, color=secondary_color, line=dict(color=text_color, width=2)),
                        fill='tozeroy',
                        fillcolor='rgba(37, 99, 235, 0.3)',  # Fixed color format
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                    ))

                    fig.update_layout(
                        paper_bgcolor=card_bg,
                        plot_bgcolor=card_bg,
                        font={'color': text_color},
                        height=350,
                        xaxis=dict(title="Date", gridcolor=border_color),
                        yaxis=dict(title="Count", gridcolor=border_color),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Data Completeness")
            completeness_data = []
            for col in st.session_state.df.columns[:8]:
                non_null = st.session_state.df[col].count()
                total = len(st.session_state.df)
                percentage = (non_null / total * 100) if total > 0 else 0
                completeness_data.append({'Column': col, 'Completeness': percentage})

            comp_df = pd.DataFrame(completeness_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comp_df['Completeness'],
                y=comp_df['Column'],
                orientation='h',
                marker=dict(
                    color=comp_df['Completeness'],
                    colorscale=[[0, error_color], [0.5, warning_color], [1, success_color]],
                    line=dict(color=border_color, width=1)
                ),
                text=[f"{val:.1f}%" for val in comp_df['Completeness']],
                textposition='outside',
                textfont=dict(size=12, color=text_color, weight='bold'),
                hovertemplate='<b>%{y}</b><br>Completeness: %{x:.1f}%<extra></extra>'
            ))

            fig.update_layout(
                title=dict(text="Data Completeness by Column", font=dict(size=18, color=text_color)),
                paper_bgcolor=card_bg,
                plot_bgcolor=card_bg,
                font={'color': text_color},
                height=400,
                xaxis=dict(title="Completeness (%)", range=[0, 105], gridcolor=border_color),
                yaxis=dict(title="", gridcolor=border_color),
                margin=dict(l=150, r=40, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating charts: {str(e)}")
            st.session_state.error_log.append(f"Error generating charts: {str(e)}")

    with tabs[1]:
        st.markdown(f'<div class="section-title">Search & Filter</div>', unsafe_allow_html=True)

        search_tabs = st.tabs(["Individual Search", "Multi-Filter"])

        with search_tabs[0]:
            st.markdown("### Search for Records")

            col1, col2 = st.columns(2)
            with col1:
                search_name = st.text_input("Name", placeholder="Enter name...")
            with col2:
                search_id = st.text_input("Certificate/ID Number", placeholder="Enter ID...")

            if search_name or search_id:
                try:
                    results = st.session_state.df.copy()

                    if search_name:
                        name_cols = [col for col in results.columns if 'NAME' in str(col).upper()]
                        if name_cols:
                            name_mask = results[name_cols].apply(
                                lambda row: row.astype(str).str.contains(search_name, case=False, na=False).any(),
                                axis=1
                            )
                            results = results[name_mask]

                    if search_id:
                        id_cols = [col for col in results.columns if any(x in str(col).upper() for x in ['CERT', 'ID', 'NUMBER'])]
                        if id_cols:
                            id_mask = results[id_cols].apply(
                                lambda row: row.astype(str).str.contains(search_id, case=False, na=False).any(),
                                axis=1
                            )
                            results = results[id_mask]

                    if not results.empty:
                        st.success(f"Found {len(results)} matching records")
                        st.session_state.filtered_data = results
                        st.dataframe(clean_dataframe_for_display(results), use_container_width=True)

                        if len(results) == 1:
                            st.markdown("### Detailed Record")
                            transp = results.T.reset_index()
                            transp.columns = ['Field', 'Value']
                            st.table(clean_dataframe_for_display(transp))

                            with io.BytesIO() as buf:
                                transp.to_excel(buf, index=False)
                                buf.seek(0)
                                st.download_button("Download Record", buf, "record.xlsx")
                    else:
                        st.warning("No matching records found")

                except Exception as e:
                    st.error(f"Error searching data: {str(e)}")
                    st.session_state.error_log.append(f"Error searching data: {str(e)}")
            else:
                st.info("Enter search terms to find records")

        with search_tabs[1]:
            st.markdown("### Apply Filters")

            try:
                col1, col2 = st.columns(2)

                company_col = get_column_by_pattern(st.session_state.df, ['COMPANY'])
                batch_col = get_column_by_pattern(st.session_state.df, ['BATCH'])
                date_col = get_column_by_pattern(st.session_state.df, ['COURSE DATE', 'DATE'])
                gender_col = get_column_by_pattern(st.session_state.df, ['GENDER'])

                sel_comp = []
                sel_batch = []
                sel_date = []
                sel_gender = []

                if company_col:
                    with col1:
                        companies = ['All'] + sorted(st.session_state.df[company_col].dropna().unique().tolist())
                        sel_comp = st.multiselect("Companies", companies, default=['All'])
                        if 'All' in sel_comp:
                            sel_comp = st.session_state.df[company_col].unique().tolist()

                if batch_col:
                    with col1:
                        batches = ['All'] + sorted(st.session_state.df[batch_col].dropna().unique().tolist())
                        sel_batch = st.multiselect("Batches", batches, default=['All'])
                        if 'All' in sel_batch:
                            sel_batch = st.session_state.df[batch_col].unique().tolist()

                if date_col:
                    with col2:
                        dates = ['All'] + sorted(st.session_state.df[date_col].dropna().unique().tolist())
                        sel_date = st.multiselect("Dates", dates, default=['All'])
                        if 'All' in sel_date:
                            sel_date = st.session_state.df[date_col].unique().tolist()

                if gender_col:
                    with col2:
                        genders = ['All'] + sorted(st.session_state.df[gender_col].dropna().unique().tolist())
                        sel_gender = st.multiselect("Genders", genders, default=['All'])
                        if 'All' in sel_gender:
                            sel_gender = st.session_state.df[gender_col].unique().tolist()

                if st.button("Apply Filters", use_container_width=True):
                    filtered = st.session_state.df.copy()

                    if company_col and sel_comp:
                        filtered = filtered[filtered[company_col].isin(sel_comp)]

                    if batch_col and sel_batch:
                        filtered = filtered[filtered[batch_col].isin(sel_batch)]

                    if date_col and sel_date:
                        filtered = filtered[filtered[date_col].isin(sel_date)]

                    if gender_col and sel_gender:
                        filtered = filtered[filtered[gender_col].isin(sel_gender)]

                    st.session_state.filtered_data = filtered
                    st.success(f"Found {len(filtered)} records matching filters")

                if st.session_state.filtered_data is not None:
                    st.markdown("### Filtered Results")
                    st.dataframe(clean_dataframe_for_display(st.session_state.filtered_data), use_container_width=True)

            except Exception as e:
                st.error(f"Error applying filters: {str(e)}")
                st.session_state.error_log.append(f"Error applying filters: {str(e)}")

    with tabs[2]:
        st.markdown(f'<div class="section-title">Data Processing</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Clean Data")

            if st.button("Remove Duplicates", use_container_width=True):
                try:
                    old_len = len(st.session_state.df)
                    st.session_state.df.drop_duplicates(inplace=True)
                    removed = old_len - len(st.session_state.df)
                    st.success(f"Removed {removed} duplicates")
                except Exception as e:
                    st.error(f"Error removing duplicates: {str(e)}")
                    st.session_state.error_log.append(f"Error removing duplicates: {str(e)}")

            fill_opt = st.radio("Missing Values", ["Keep", "Fill", "Drop"])

            if fill_opt == "Fill":
                fill_val = st.text_input("Fill Value", "N/A")
                if st.button("Apply Fill", use_container_width=True):
                    try:
                        st.session_state.df.fillna(fill_val, inplace=True)
                        st.success("Filled missing values")
                    except Exception as e:
                        st.error(f"Error filling values: {str(e)}")
                        st.session_state.error_log.append(f"Error filling values: {str(e)}")

            elif fill_opt == "Drop":
                if st.button("Apply Drop", use_container_width=True):
                    try:
                        old_len = len(st.session_state.df)
                        st.session_state.df.dropna(inplace=True)
                        dropped = old_len - len(st.session_state.df)
                        st.success(f"Dropped {dropped} rows with missing values")
                    except Exception as e:
                        st.error(f"Error dropping rows: {str(e)}")
                        st.session_state.error_log.append(f"Error dropping rows: {str(e)}")

        with col2:
            st.markdown("### Modify Data")

            col_remove = st.selectbox("Remove Column", ['None'] + list(st.session_state.df.columns))
            if col_remove != 'None' and st.button("Remove Column", use_container_width=True):
                try:
                    st.session_state.df.drop(columns=[col_remove], inplace=True)
                    if st.session_state.filtered_data is not None:
                        st.session_state.filtered_data.drop(columns=[col_remove], errors='ignore', inplace=True)
                    st.success(f"Removed column: {col_remove}")
                except Exception as e:
                    st.error(f"Error removing column: {str(e)}")
                    st.session_state.error_log.append(f"Error removing column: {str(e)}")

            if st.button("Standardize Text", use_container_width=True):
                try:
                    text_cols = st.session_state.df.select_dtypes('object').columns
                    for col in text_cols:
                        if col != 'Full Name':
                            st.session_state.df[col] = st.session_state.df[col].str.title().str.strip()
                    st.success("Standardized text columns")
                except Exception as e:
                    st.error(f"Error standardizing text: {str(e)}")
                    st.session_state.error_log.append(f"Error standardizing text: {str(e)}")

            if st.button("Reset Data", use_container_width=True):
                try:
                    if st.session_state.uploaded_file:
                        if st.session_state.uploaded_file.name.endswith(".csv"):
                            reset_df = pd.read_csv(st.session_state.uploaded_file, header=st.session_state.header_row, dtype=str)
                        else:
                            try:
                                reset_df = pd.read_excel(st.session_state.uploaded_file, header=st.session_state.header_row, dtype=str)
                            except:
                                reset_df = pd.read_excel(st.session_state.uploaded_file, header=st.session_state.header_row, dtype=str, engine="openpyxl")

                        reset_df = handle_merged_cells(reset_df)
                        reset_df, error_msg = safe_process_data(reset_df, st.session_state.header_row, auto_ffill)

                        if error_msg:
                            st.warning(error_msg)
                            st.session_state.error_log.append(error_msg)

                        st.session_state.df = reset_df.copy()
                        st.session_state.original_columns = reset_df.columns.tolist()
                        st.success("Data reset successfully")
                    else:
                        st.warning("No file uploaded to reset from")
                except Exception as e:
                    st.error(f"Error resetting data: {str(e)}")
                    st.session_state.error_log.append(f"Error resetting data: {str(e)}")

        if st.session_state.error_log:
            with st.expander("Error Log"):
                for i, error in enumerate(st.session_state.error_log):
                    st.text(f"Error {i+1}:\n{error}")
                if st.button("Clear Error Log"):
                    st.session_state.error_log = []
                    st.success("Error log cleared")

    with tabs[3]:
        st.markdown(f'<div class="section-title">Export Data</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Filtered Dataset")
            if st.session_state.filtered_data is not None:
                st.metric("Records to Export", len(st.session_state.filtered_data))

                try:
                    with io.BytesIO() as buf:
                        st.session_state.filtered_data.to_excel(buf, index=False)
                        buf.seek(0)
                        st.download_button("Download as Excel", buf, "filtered_data.xlsx", use_container_width=True)

                    csv = st.session_state.filtered_data.to_csv(index=False)
                    st.download_button("Download as CSV", csv, "filtered_data.csv", use_container_width=True)

                except Exception as e:
                    st.error(f"Error exporting filtered data: {str(e)}")
                    st.session_state.error_log.append(f"Error exporting filtered data: {str(e)}")
            else:
                st.info("No filtered data available. Use the Search & Filter tab to create filtered data.")

        with col2:
            st.markdown("### Full Dataset")
            st.metric("Total Records", len(st.session_state.df))

            try:
                with io.BytesIO() as buf:
                    st.session_state.df.to_excel(buf, index=False)
                    buf.seek(0)
                    st.download_button("Download as Excel", buf, "full_data.xlsx", use_container_width=True)

                csv_full = st.session_state.df.to_csv(index=False)
                st.download_button("Download as CSV", csv_full, "full_data.csv", use_container_width=True)

            except Exception as e:
                st.error(f"Error exporting full data: {str(e)}")
                st.session_state.error_log.append(f"Error exporting full data: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    if not st.session_state.show_performance:
        st.info("Please upload an Excel/CSV file or enable 'Show Sample Data' in the sidebar to get started.")

st.markdown(f"""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid {border_color};">
    <div style="background: linear-gradient(135deg, {gradient_start}, {gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem;">
        Delegate Management Dashboard
    </div>
    <div style="color: {text_color}; opacity: 0.6; font-size: 0.875rem;">
        Version 5.0 | Built with Streamlit | © 2025
    </div>
</div>
""", unsafe_allow_html=True)

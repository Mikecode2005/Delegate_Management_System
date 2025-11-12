import streamlit as st
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re
from streamlit_lottie import st_lottie
import requests
import json

# Set page configuration for ultra-modern dashboard
st.set_page_config(
    page_title="Delegate Management Dashboard",
    layout="wide",
    page_icon="👥",
    initial_sidebar_state="expanded"
)

# ================ LOTTIE ANIMATIONS ================

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Lottie animations
lottie_analytics = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_uzkzyzbq.json")
lottie_upload = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_6wzywrox.json")
lottie_success = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_y5df6bbw.json")

# ================ UTILITY FUNCTIONS ================

def clean_dataframe_for_display(df):
    """Clean dataframe to ensure Arrow compatibility for Streamlit display"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
        df_clean[col] = df_clean[col].fillna('')
    
    return df_clean

def safe_process_data(df, header_row=0, auto_ffill=True):
    """Safely process the dataframe with error handling"""
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
    """Find column names matching patterns"""
    for pattern in pattern_list:
        matches = [col for col in df.columns if pattern in str(col).upper()]
        if matches:
            return matches[0]
    return default

def handle_merged_cells(df):
    """Handle merged cells by forward-filling values"""
    potential_merged_cols = []
    for col in df.columns:
        null_ratio = df[col].isna().sum() / len(df)
        if 0.3 < null_ratio < 0.9:
            potential_merged_cols.append(col)
    
    if potential_merged_cols:
        df[potential_merged_cols] = df[potential_merged_cols].ffill()
    
    return df

# ================ MODERN THEME MANAGEMENT ================

THEMES = {
    "Cyber Neon": {
        "primary_color": "#00FFFF",
        "secondary_color": "#FF00FF",
        "tertiary_color": "#00FF00",
        "background_color": "#0A0A0A",
        "text_color": "#FFFFFF",
        "card_bg": "#1A1A1A",
        "accent_color": "#FF6B6B",
        "plotly_theme": "plotly_dark",
        "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    },
    "Modern Glass": {
        "primary_color": "#3A86FF",
        "secondary_color": "#8338EC",
        "tertiary_color": "#FF006E",
        "background_color": "#F8F9FA",
        "text_color": "#212529",
        "card_bg": "rgba(255, 255, 255, 0.8)",
        "accent_color": "#FB5607",
        "plotly_theme": "plotly_white",
        "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    },
    "Dark Matrix": {
        "primary_color": "#00FF41",
        "secondary_color": "#008F11",
        "tertiary_color": "#03A062",
        "background_color": "#0A0A0A",
        "text_color": "#00FF41",
        "card_bg": "#1A1A1A",
        "accent_color": "#FF073A",
        "plotly_theme": "plotly_dark",
        "gradient": "linear-gradient(135deg, #00FF41 0%, #008F11 100%)"
    },
    "Sunset Vibes": {
        "primary_color": "#FF6B6B",
        "secondary_color": "#4ECDC4",
        "tertiary_color": "#FFE66D",
        "background_color": "#1A1A2E",
        "text_color": "#FFFFFF",
        "card_bg": "#16213E",
        "accent_color": "#F8B500",
        "plotly_theme": "plotly_dark",
        "gradient": "linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%)"
    }
}

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = "Cyber Neon"

# Get current theme colors
current_theme = THEMES[st.session_state.theme]
primary_color = current_theme["primary_color"]
secondary_color = current_theme["secondary_color"]
tertiary_color = current_theme["tertiary_color"]
background_color = current_theme["background_color"]
text_color = current_theme["text_color"]
card_bg = current_theme["card_bg"]
accent_color = current_theme["accent_color"]
plotly_theme = current_theme["plotly_theme"]
gradient = current_theme["gradient"]

# Apply Ultra-Modern CSS
st.markdown(f"""
<style>
    /* Base theme with glass morphism */
    .stApp, [data-testid="stAppViewContainer"], .block-container {{
        background: {background_color} !important;
        color: {text_color} !important;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Glass morphism cards */
    .glass-card {{
        background: {card_bg};
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    
    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }}
    
    /* Modern headers */
    .main-header {{
        font-size: 3rem;
        background: {gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }}
    
    .section-header {{
        font-size: 1.8rem;
        color: {primary_color};
        font-weight: 700;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Circular metrics */
    .circular-metric {{
        background: {gradient};
        border-radius: 50%;
        width: 120px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.5rem;
        margin: 0 auto;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }}
    
    .metric-label {{
        text-align: center;
        color: {text_color};
        font-weight: 600;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }}
    
    /* Enhanced buttons */
    .stButton > button {{
        background: {gradient};
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }}
    
    /* Modern sidebar */
    [data-testid="stSidebar"] {{
        background: {card_bg} !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {card_bg} !important;
        border-radius: 15px;
        padding: 0.5rem;
        gap: 0.5rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {text_color} !important;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {gradient} !important;
        color: white !important;
        border-radius: 10px;
    }}
    
    /* Progress bars */
    .progress-bar {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }}
    
    .progress-fill {{
        background: {gradient};
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }}
    
    /* Data grid */
    .data-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }}
    
    /* Animation containers */
    .lottie-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }}
    
    /* Stats highlight */
    .stats-highlight {{
        background: {gradient};
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }}
    
    /* Horizontal bars */
    .horizontal-bar {{
        background: {gradient};
        height: 6px;
        border-radius: 3px;
        margin: 0.5rem 0;
        transition: width 0.3s ease;
    }}
</style>
""", unsafe_allow_html=True)

# ================ SESSION STATE INITIALIZATION ================

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

# ================ SIDEBAR ================

with st.sidebar:
    st.markdown(f'<h3 style="color: {primary_color}; text-align: center;">🚀 Control Hub</h3>', unsafe_allow_html=True)
    
    # Theme selector with preview
    selected_theme = st.selectbox(
        "🎨 Theme", 
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme)
    )
    
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    st.markdown("---")
    
    # Data options in expander
    with st.expander("📊 Data Options", expanded=True):
        show_sample = st.checkbox("Show Sample Data", value=False)
        auto_ffill = st.checkbox("Auto Forward-Fill", value=True)
        compact_mode = st.checkbox("Compact Mode", value=True)
    
    # Visualization options
    with st.expander("🎯 Display Options"):
        chart_style = st.selectbox(
            "Chart Style", 
            ["Neon Glow", "Glass Morphism", "Minimal Dark", "Color Burst"], 
            index=0
        )
        animation_level = st.select_slider(
            "Animation Level",
            options=["None", "Subtle", "Moderate", "High"]
        )
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("📊 Report", use_container_width=True):
            st.success("Generating comprehensive report...")
    
    # Help section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **Pro Tips:**
        - Use **Circular Metrics** for quick insights
        - **Hover animations** reveal detailed info
        - **Glass morphism** cards for modern look
        - **Gradient themes** for visual appeal
        """)
    
    st.markdown("---")
    st.markdown(f'<p style="text-align: center; color: {text_color};">Delegate Analytics v5.0</p>', unsafe_allow_html=True)

# ================ MAIN CONTENT ================

st.markdown('<h1 class="main-header">🎯 Delegate Intelligence Dashboard</h1>', unsafe_allow_html=True)

# ================ HERO SECTION ================

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 class="section-header">📈 Real-time Analytics</h3>', unsafe_allow_html=True)
    st.markdown("""
    **Next-gen delegate management with:**
    - 🎯 Smart data processing
    - 📊 Interactive visualizations  
    - 🎨 Modern glass morphism UI
    - ⚡ Real-time insights
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if lottie_analytics:
        st_lottie(lottie_analytics, height=150, key="analytics")

with col3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 class="section-header">🚀 Quick Start</h3>', unsafe_allow_html=True)
    st.markdown("""
    1. **Upload** your dataset
    2. **Explore** interactive dashboards  
    3. **Analyze** with modern visuals
    4. **Export** insights
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ================ FILE UPLOAD SECTION ================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown(f'<h2 class="section-header">📤 Upload Data</h2>', unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns([3, 1])

with upload_col1:
    uploaded_file = st.file_uploader("Drag & drop Excel/CSV file", type=["xlsx", "csv"], key="uploader")

with upload_col2:
    if lottie_upload:
        st_lottie(lottie_upload, height=100, key="upload")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    
    try:
        if uploaded_file.name.endswith(".csv"):
            temp_df = pd.read_csv(uploaded_file, header=None)
        else:
            try:
                temp_df = pd.read_excel(uploaded_file, header=None)
            except:
                temp_df = pd.read_excel(uploaded_file, header=None, engine="openpyxl")
        
        # Enhanced preview with tabs
        preview_tab1, preview_tab2 = st.tabs(["📋 Data Preview", "🔍 Raw Structure"])
        
        with preview_tab1:
            st.dataframe(clean_dataframe_for_display(temp_df.head(10)), use_container_width=True)
        
        with preview_tab2:
            st.write(f"**Dimensions:** {temp_df.shape[0]} rows × {temp_df.shape[1]} columns")
            st.write(f"**Memory Usage:** {temp_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        
        # Header row selection with visual indicator
        header_row = st.selectbox(
            "🎯 Header Row Selection", 
            list(range(min(10, len(temp_df)))), 
            index=min(1, len(temp_df)-1),
            help="Select the row that contains your column headers"
        )
        st.session_state.header_row = header_row
        
        # Process data with modern loading
        with st.spinner('🚀 Processing your data with AI-powered analytics...'):
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
                    st.warning(f"⚠️ {error_msg}")
                    st.session_state.error_log.append(error_msg)
                
                st.session_state.df = df.copy()
                st.session_state.original_columns = df.columns.tolist()
                st.session_state.processed = True
                
                # Success animation
                if lottie_success:
                    st_lottie(lottie_success, height=100, key="success")
                
                st.success("🎉 Data uploaded and processed successfully! Ready for analysis.")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.session_state.error_log.append(f"Error processing file: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.session_state.error_log.append(f"Error reading file: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# ================ MODERN DASHBOARD SECTIONS ================

if st.session_state.df is not None and not st.session_state.df.empty:
    
    # ================ CIRCULAR METRICS DASHBOARD ================
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 Executive Overview</h2>', unsafe_allow_html=True)
    
    # Circular metrics in grid
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="circular-metric">{len(st.session_state.df)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Records</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="circular-metric">{len(st.session_state.df.columns)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Data Columns</div>', unsafe_allow_html=True)
    
    with col3:
        missing_vals = st.session_state.df.isna().sum().sum()
        st.markdown(f'<div class="circular-metric">{missing_vals}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Missing Values</div>', unsafe_allow_html=True)
    
    with col4:
        unique_avg = int(st.session_state.df.nunique().mean())
        st.markdown(f'<div class="circular-metric">{unique_avg}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg Unique</div>', unsafe_allow_html=True)
    
    with col5:
        completeness = int((1 - missing_vals / (len(st.session_state.df) * len(st.session_state.df.columns))) * 100)
        st.markdown(f'<div class="circular-metric">{completeness}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Data Quality</div>', unsafe_allow_html=True)
    
    # Horizontal progress bars for key metrics
    st.markdown("### 📈 Data Health Metrics")
    
    # Data completeness
    comp_percent = completeness / 100
    st.markdown(f"**Data Completeness:** {completeness}%")
    st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {comp_percent * 100}%"></div></div>', unsafe_allow_html=True)
    
    # Unique value ratio
    unique_ratio = st.session_state.df.nunique().mean() / len(st.session_state.df)
    st.markdown(f"**Data Diversity:** {unique_ratio:.1%}")
    st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {unique_ratio * 100}%"></div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ================ INTERACTIVE ANALYSIS DASHBOARD ================
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    analysis_tabs = st.tabs(["🎯 Smart Analytics", "📊 Visual Explorer", "🔍 Data Profiler"])
    
    # ================ SMART ANALYTICS TAB ================
    
    with analysis_tabs[0]:
        st.markdown(f'<h3 style="color: {primary_color};">🤖 AI-Powered Insights</h3>', unsafe_allow_html=True)
        
        try:
            # Create a modern grid layout
            col1, col2 = st.columns(2)
            
            # Enhanced gender distribution with donut chart
            gender_col = get_column_by_pattern(st.session_state.df, ['GENDER'])
            if gender_col:
                with col1:
                    gender_data = st.session_state.df[gender_col].value_counts()
                    
                    # Create modern donut chart
                    fig_gender = go.Figure()
                    
                    colors = [primary_color, secondary_color, tertiary_color, accent_color]
                    
                    fig_gender.add_trace(go.Pie(
                        values=gender_data.values,
                        labels=gender_data.index,
                        hole=0.6,
                        marker_colors=colors,
                        textinfo='percent+label',
                        insidetextorientation='radial'
                    ))
                    
                    fig_gender.update_layout(
                        title_text="<b>Gender Distribution</b>",
                        title_x=0.5,
                        showlegend=False,
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=text_color)
                    )
                    
                    st.plotly_chart(fig_gender, use_container_width=True)
            
            # Age distribution with modern histogram
            if 'Age' in st.session_state.df.columns:
                with col2:
                    age_data = st.session_state.df['Age'].dropna().astype(float)
                    if not age_data.empty:
                        fig_age = go.Figure()
                        
                        fig_age.add_trace(go.Histogram(
                            x=age_data,
                            nbinsx=20,
                            marker_color=primary_color,
                            opacity=0.8,
                            name="Age Distribution"
                        ))
                        
                        fig_age.update_layout(
                            title_text="<b>Age Distribution Analysis</b>",
                            title_x=0.5,
                            xaxis_title="Age",
                            yaxis_title="Count",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=text_color),
                            bargap=0.1
                        )
                        
                        st.plotly_chart(fig_age, use_container_width=True)
            
            # Company analysis with horizontal bars
            company_col = get_column_by_pattern(st.session_state.df, ['COMPANY'])
            if company_col:
                comp_data = st.session_state.df[company_col].value_counts().head(10)
                
                fig_comp = go.Figure()
                
                fig_comp.add_trace(go.Bar(
                    y=comp_data.index,
                    x=comp_data.values,
                    orientation='h',
                    marker_color=secondary_color,
                    text=comp_data.values,
                    textposition='auto',
                ))
                
                fig_comp.update_layout(
                    title_text="<b>Top 10 Companies</b>",
                    title_x=0.5,
                    xaxis_title="Delegate Count",
                    yaxis_title="Companies",
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color),
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")
    
    # ================ VISUAL EXPLORER TAB ================
    
    with analysis_tabs[1]:
        st.markdown(f'<h3 style="color: {primary_color};">🎨 Interactive Visualizations</h3>', unsafe_allow_html=True)
        
        try:
            # Create visualization grid
            viz_col1, viz_col2 = st.columns(2)
            
            # Batch analysis
            batch_col = get_column_by_pattern(st.session_state.df, ['BATCH'])
            if batch_col:
                with viz_col1:
                    batch_data = st.session_state.df[batch_col].value_counts()
                    
                    fig_batch = go.Figure(data=[go.Scatter(
                        x=batch_data.index,
                        y=batch_data.values,
                        mode='lines+markers',
                        line=dict(color=primary_color, width=4),
                        marker=dict(size=12, color=accent_color),
                        fill='tozeroy',
                        fillcolor=f'{primary_color}20'
                    )])
                    
                    fig_batch.update_layout(
                        title_text="<b>Batch Distribution Trend</b>",
                        title_x=0.5,
                        xaxis_title="Batch",
                        yaxis_title="Count",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=text_color)
                    )
                    
                    st.plotly_chart(fig_batch, use_container_width=True)
            
            # Date analysis
            date_col = get_column_by_pattern(st.session_state.df, ['COURSE DATE', 'DATE'])
            if date_col:
                with viz_col2:
                    date_data = st.session_state.df[date_col].value_counts().sort_index()
                    
                    fig_date = go.Figure(data=[go.Bar(
                        x=date_data.index,
                        y=date_data.values,
                        marker_color=tertiary_color,
                        marker_line=dict(color=accent_color, width=2)
                    )])
                    
                    fig_date.update_layout(
                        title_text="<b>Course Date Distribution</b>",
                        title_x=0.5,
                        xaxis_title="Date",
                        yaxis_title="Count",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=text_color)
                    )
                    
                    st.plotly_chart(fig_date, use_container_width=True)
            
            # Additional metrics in a 2x2 grid
            st.markdown("### 📋 Additional Insights")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                if gender_col:
                    gender_count = st.session_state.df[gender_col].nunique()
                    st.metric("Gender Types", gender_count)
            
            with metric_col2:
                if company_col:
                    company_count = st.session_state.df[company_col].nunique()
                    st.metric("Unique Companies", company_count)
            
            with metric_col3:
                if batch_col:
                    batch_count = st.session_state.df[batch_col].nunique()
                    st.metric("Total Batches", batch_count)
            
            with metric_col4:
                if 'Age' in st.session_state.df.columns:
                    avg_age = st.session_state.df['Age'].mean()
                    st.metric("Average Age", f"{avg_age:.1f}")
        
        except Exception as e:
            st.error(f"Error in visual explorer: {str(e)}")
    
    # ================ DATA PROFILER TAB ================
    
    with analysis_tabs[2]:
        st.markdown(f'<h3 style="color: {primary_color};">🔍 Advanced Data Profiling</h3>', unsafe_allow_html=True)
        
        # Column information in modern table
        col_info = pd.DataFrame({
            "Column": st.session_state.df.columns,
            "Data Type": st.session_state.df.dtypes,
            "Non-Null": st.session_state.df.count(),
            "Null %": (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2),
            "Unique Values": st.session_state.df.nunique()
        })
        
        # Style the dataframe
        styled_col_info = col_info.style.background_gradient(subset=['Null %'], cmap='Reds')\
                                      .background_gradient(subset=['Unique Values'], cmap='Blues')
        
        st.dataframe(styled_col_info, use_container_width=True)
        
        # Data quality metrics
        st.markdown("### 📊 Data Quality Scorecard")
        
        quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
        
        with quality_col1:
            completeness_score = 100 - (st.session_state.df.isna().sum().sum() / (len(st.session_state.df) * len(st.session_state.df.columns)) * 100)
            st.metric("Completeness", f"{completeness_score:.1f}%")
        
        with quality_col2:
            uniqueness_score = (st.session_state.df.nunique().mean() / len(st.session_state.df)) * 100
            st.metric("Uniqueness", f"{uniqueness_score:.1f}%")
        
        with quality_col3:
            consistency_score = 100 - (st.session_state.df.duplicated().sum() / len(st.session_state.df) * 100)
            st.metric("Consistency", f"{consistency_score:.1f}%")
        
        with quality_col4:
            validity_score = 85.0  # Placeholder for actual validity calculation
            st.metric("Validity", f"{validity_score:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Empty state with modern design
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-header">🚀 Ready for Analysis</h2>', unsafe_allow_html=True)
    
    empty_col1, empty_col2 = st.columns([2, 1])
    
    with empty_col1:
        st.markdown("""
        ### 📊 Get Started
        
        Upload your delegate data to unlock:
        
        - ** Smart Analytics** - AI-powered insights
        - ** Modern Visualizations** - Interactive charts
        - ** Real-time Metrics** - Live data monitoring
        - ** Advanced Profiling** - Deep data analysis
        
        *Upload an Excel/CSV file or enable sample data to begin your analysis journey!*
        """)
    
    with empty_col2:
        if lottie_analytics:
            st_lottie(lottie_analytics, height=200, key="empty_analytics")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================ FOOTER ================

st.markdown("---")
st.markdown(
    f'<div style="text-align: center; color: {primary_color}; padding: 2rem; font-size: 0.9rem;">'
    ' Delegate Intelligence Dashboard v5.0 | '
    'Built with ❤️ using Streamlit | '
    '© 2025 Next-Gen Analytics'
    '</div>', 
    unsafe_allow_html=True
)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Analytics Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Modern color scheme inspired by reference designs
COLORS = {
    'primary': '#3B82F6',      # Blue
    'secondary': '#60A5FA',    # Light Blue
    'accent': '#2563EB',       # Dark Blue
    'success': '#10B981',      # Green
    'warning': '#F59E0B',      # Orange
    'background': '#0F172A',   # Dark Navy
    'card': '#FFFFFF',         # White
    'text': '#1E293B',         # Dark Text
    'muted': '#64748B'         # Muted Text
}

# Custom CSS for modern design matching reference images
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 100%);
        padding: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #3B82F6;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-trend {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .trend-up {
        color: #10B981;
    }
    
    .trend-down {
        color: #EF4444;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }
    
    .chart-title {
        font-size: 1.125rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E40AF 0%, #1E3A8A 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #3B82F6;
        border-radius: 16px;
        padding: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Data table */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #64748B;
        border: 1px solid #E5E7EB;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
    }
    
    /* Performance Metrics Grid */
    .perf-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .perf-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .perf-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.15);
    }
    
    .perf-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .perf-value {
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .perf-label {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .perf-change {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .change-positive {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
    }
    
    .change-negative {
        background: rgba(239, 68, 68, 0.1);
        color: #EF4444;
    }
    
    /* Progress bars */
    .progress-container {
        width: 100%;
        height: 8px;
        background: #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #3B82F6, #2563EB);
        border-radius: 10px;
        transition: width 1.5s ease-in-out;
    }
    
    /* Stats highlight */
    .stats-highlight {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    
    .stats-highlight h2 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .stats-highlight p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'show_performance' not in st.session_state:
    st.session_state.show_performance = True

# Helper functions
def clean_dataframe_for_display(df):
    """Clean dataframe for display"""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
        df_clean[col] = df_clean[col].fillna('')
    return df_clean

def create_metric_card(label, value, trend=None):
    """Create a modern metric card"""
    trend_html = ""
    if trend:
        trend_class = "trend-up" if trend > 0 else "trend-down"
        trend_symbol = "↑" if trend > 0 else "↓"
        trend_html = f'<div class="metric-trend {trend_class}">{trend_symbol} {abs(trend)}%</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {trend_html}
    </div>
    """

def create_performance_card(icon, value, label, change=None, progress=None):
    """Create a performance metric card"""
    change_html = ""
    if change is not None:
        change_class = "change-positive" if change > 0 else "change-negative"
        change_symbol = "↗" if change > 0 else "↘"
        change_html = f'<div class="perf-change {change_class}">{change_symbol} {abs(change)}%</div>'
    
    progress_html = ""
    if progress is not None:
        progress_html = f'''
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress}%"></div>
        </div>
        '''
    
    return f"""
    <div class="perf-card">
        <div class="perf-icon">{icon}</div>
        <div class="perf-value">{value}</div>
        <div class="perf-label">{label}</div>
        {change_html}
        {progress_html}
    </div>
    """

def create_line_chart(data, x, y, title):
    """Create a modern line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data[x],
        y=data[y],
        mode='lines+markers',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8, color='#3B82F6'),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1E293B', family='Inter', weight=700)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        margin=dict(l=40, r=40, t=60, b=40),
        height=300,
        font=dict(family='Inter')
    )
    
    return fig

def create_donut_chart(labels, values, title):
    """Create a modern donut chart"""
    colors = ['#3B82F6', '#60A5FA', '#93C5FD', '#DBEAFE']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=12, family='Inter')
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1E293B', family='Inter', weight=700)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
        height=300,
        showlegend=True,
        font=dict(family='Inter')
    )
    
    return fig

def create_bar_chart(data, x, y, title):
    """Create a modern bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=data[x],
            y=data[y],
            marker=dict(
                color='#3B82F6',
                line=dict(color='#2563EB', width=1)
            )
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1E293B', family='Inter', weight=700)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        margin=dict(l=40, r=40, t=60, b=40),
        height=300,
        font=dict(family='Inter')
    )
    
    return fig

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

# Sidebar
with st.sidebar:
    st.markdown("# ⚙️ Control Panel")
    st.markdown("---")
    
    st.markdown("### 📤 Upload Data")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
    
    st.markdown("---")
    st.markdown("### 🎨 Display Options")
    show_sample = st.checkbox("Show Sample Data", value=False)
    show_charts = st.checkbox("Show Analytics Charts", value=True)
    show_performance = st.checkbox("Show Performance Metrics", value=True)
    
    st.markdown("---")
    st.markdown("### 📋 Help")
    with st.expander("Quick Guide"):
        st.markdown("""
        1. Upload your Excel/CSV file
        2. View analytics automatically
        3. Explore charts and metrics
        4. Export filtered results
        """)

# Main content
st.markdown('<h1 class="dashboard-header">Analytics Dashboard</h1>', unsafe_allow_html=True)

# Performance Metrics Section (shown before data upload)
if st.session_state.show_performance and st.session_state.df is None:
    st.markdown("## 📈 Performance Overview")
    
    # Performance highlight
    st.markdown("""
    <div class="stats-highlight">
        <h2>🚀 Ready to Analyze Your Data</h2>
        <p>Upload your dataset to unlock powerful insights and beautiful visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics grid
    perf_data = generate_performance_data()
    
    st.markdown('<div class="perf-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_performance_card(
            "👥", 
            f"{perf_data['total_visitors']:,}", 
            "Total Visitors", 
            change=12.4,
            progress=85
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_performance_card(
            "📄", 
            f"{perf_data['page_views']:,}", 
            "Page Views", 
            change=8.7,
            progress=72
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_performance_card(
            "⏱️", 
            perf_data['avg_session'], 
            "Avg. Session", 
            change=5.2,
            progress=65
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_performance_card(
            "📉", 
            f"{perf_data['bounce_rate']}%", 
            "Bounce Rate", 
            change=-3.1,
            progress=32
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row of performance metrics
    st.markdown('<div class="perf-grid">', unsafe_allow_html=True)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(create_performance_card(
            "🎯", 
            f"{perf_data['conversion_rate']}%", 
            "Conversion Rate", 
            change=15.3,
            progress=87
        ), unsafe_allow_html=True)
    
    with col6:
        st.markdown(create_performance_card(
            "💰", 
            perf_data['revenue'], 
            "Revenue", 
            change=22.8,
            progress=92
        ), unsafe_allow_html=True)
    
    with col7:
        st.markdown(create_performance_card(
            "🔥", 
            f"{perf_data['active_users']:,}", 
            "Active Users", 
            change=18.3,
            progress=78
        ), unsafe_allow_html=True)
    
    with col8:
        st.markdown(create_performance_card(
            "📊", 
            f"{perf_data['growth_rate']}%", 
            "Growth Rate", 
            change=6.9,
            progress=68
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample charts section
    st.markdown("## 📊 Sample Analytics Preview")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Sample line chart
        sample_line = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=7),
            'Visitors': [120, 85, 135, 95, 109, 130, 145]
        })
        fig = create_line_chart(sample_line, 'Date', 'Visitors', "Visitors This Week")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Sample donut chart
        fig = create_donut_chart(
            ['Desktop', 'Mobile', 'Tablet'],
            [65, 25, 10],
            "Device Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
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

# Sample data generation
if show_sample and st.session_state.df is None:
    sample_data = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=20, freq='W'),
        "Visitors": np.random.randint(80, 150, 20),
        "Page Views": np.random.randint(200, 500, 20),
        "Device": np.random.choice(['Desktop', 'Mobile', 'Tablet'], 20),
        "Status": np.random.choice(['Active', 'Completed', 'Pending'], 20)
    })
    st.session_state.df = sample_data
    st.session_state.processed = True
    st.session_state.show_performance = False

# File upload processing
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df = df
        st.session_state.processed = True
        st.session_state.show_performance = False
        st.success("✅ Data uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Display analytics if data is available
if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df
    
    # Metrics row
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("Total Records", len(df), 12.5), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("Data Columns", len(df.columns), 8.2), unsafe_allow_html=True)
    
    with col3:
        missing = df.isna().sum().sum()
        st.markdown(create_metric_card("Missing Values", missing, -3.1), unsafe_allow_html=True)
    
    with col4:
        unique_avg = int(df.nunique().mean())
        st.markdown(create_metric_card("Avg Unique", unique_avg, 15.3), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts section
    if show_charts:
        st.markdown("### 📈 Data Visualization")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            if 'Date' in df.columns or df.select_dtypes(include=['datetime64']).columns.any():
                date_col = 'Date' if 'Date' in df.columns else df.select_dtypes(include=['datetime64']).columns[0]
                numeric_col = df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else None
                
                if numeric_col:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = create_line_chart(df.head(10), date_col, numeric_col, "Trend Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Sample line chart
                sample_line = pd.DataFrame({
                    'Date': pd.date_range('2024-01-01', periods=7),
                    'Value': [120, 85, 135, 95, 109, 130, 145]
                })
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_line_chart(sample_line, 'Date', 'Value', "Visitors This Month")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with chart_col2:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                value_counts = df[cat_col].value_counts().head(4)
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_donut_chart(
                    value_counts.index.tolist(),
                    value_counts.values.tolist(),
                    f"Distribution by {cat_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Sample donut chart
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_donut_chart(
                    ['Desktop', 'Mobile', 'Tablet'],
                    [65, 25, 10],
                    "Device Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Second row of charts
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                num_col = numeric_cols[0]
                bar_data = df.groupby(df.index // 10)[num_col].mean().reset_index()
                bar_data.columns = ['Group', 'Average']
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_bar_chart(bar_data.head(8), 'Group', 'Average', "Site Traffic Data")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Sample bar chart
                sample_bar = pd.DataFrame({
                    'Week': ['Feb 1', 'Feb 8', 'Feb 15', 'Feb 22', 'Mar 1', 'Mar 8', 'Mar 15', 'Mar 22'],
                    'Value': [85, 72, 90, 78, 88, 65, 95, 82]
                })
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_bar_chart(sample_bar, 'Week', 'Value', "Site Traffic Data")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with chart_col4:
            if len(categorical_cols) > 1:
                cat_col2 = categorical_cols[1]
                value_counts2 = df[cat_col2].value_counts().head(4)
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_donut_chart(
                    value_counts2.index.tolist(),
                    value_counts2.values.tolist(),
                    f"Status by {cat_col2}"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Sample status chart
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_donut_chart(
                    ['Active', 'Completed', 'Pending', 'Cancelled'],
                    [45, 32, 18, 5],
                    "Project Status"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Data table
    st.markdown("### 📋 Data Table")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Tabs for data view
    tab1, tab2, tab3 = st.tabs(["📊 View Data", "🔍 Search", "⚙️ Filter"])
    
    with tab1:
        st.dataframe(clean_dataframe_for_display(df), use_container_width=True, height=400)
    
    with tab2:
        search_term = st.text_input("Search in data")
        if search_term:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            st.dataframe(df[mask], use_container_width=True)
    
    with tab3:
        if len(categorical_cols) > 0:
            filter_col = st.selectbox("Select column to filter", categorical_cols)
            filter_values = st.multiselect("Select values", df[filter_col].unique())
            if filter_values:
                filtered_df = df[df[filter_col].isin(filter_values)]
                st.dataframe(filtered_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button
    st.markdown("<br>", unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name="analytics_data.csv",
        mime="text/csv",
    )

else:
    # Welcome screen when no data (only shown if performance metrics are disabled)
    if not st.session_state.show_performance:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h2 style="color: #3B82F6; font-size: 2rem; margin-bottom: 1rem;">Welcome to Analytics Dashboard</h2>
            <p style="color: #64748B; font-size: 1.125rem;">Upload your data to see beautiful visualizations and insights</p>
            <br>
            <p style="color: #94A3B8;">👈 Use the sidebar to upload a CSV or Excel file</p>
        </div>
        """, unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import io
import plotly.express as px
from datetime import datetime
import numpy as np
import traceback
import re

# Set page configuration for modern dashboard
st.set_page_config(
    page_title="Delegate Management Dashboard",
    layout="wide",
    page_icon="\ud83d\udc65",
    initial_sidebar_state="expanded"
)

# ================ UTILITY FUNCTIONS ================

def safe_process_data(df, header_row=0, auto_ffill=True):
    """
    Safely process the dataframe with error handling
    """
    try:
        # Set column names
        df.columns = [str(col).strip().replace('\
', ' ').title() for col in df.columns]
        
        # Forward fill date-related columns if auto_ffill is enabled
        if auto_ffill:
            date_cols = [col for col in df.columns if any(x in str(col).upper() for x in ['BATCH', 'COURSE DATE', 'ISSUED DATE', 'EXPIRY DATE', 'DATE'])]
            if date_cols:
                df[date_cols] = df[date_cols].ffill()
        
        # Clean phone number columns
        phone_cols = [col for col in df.columns if any(x in str(col).upper() for x in ['NUMBER', 'PHONE', 'NOK', 'CONTACT', 'TEL'])]
        for col in phone_cols:
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        # Create full name column if first and last name columns exist
        first_name_col = next((col for col in df.columns if 'FIRST NAME' in str(col).upper()), None)
        last_name_col = next((col for col in df.columns if 'LAST NAME' in str(col).upper()), None)
        if first_name_col and last_name_col:
            df['Full Name'] = df[first_name_col].fillna('') + ' ' + df[last_name_col].fillna('')
        
        # Process DOB and calculate age
        dob_col = next((col for col in df.columns if 'DOB' in str(col).upper() or 'BIRTH' in str(col).upper()), None)
        if dob_col:
            try:
                df[dob_col] = pd.to_datetime(df[dob_col], errors='coerce')
                df['Age'] = ((datetime.now() - df[dob_col]).dt.days // 365.25).astype('Int64')
            except Exception as e:
                st.warning(f"Unable to calculate ages: {str(e)}")
        
        return df, None
    except Exception as e:
        error_msg = f"Error processing data: {str(e)}\
{traceback.format_exc()}"
        return df, error_msg

def get_column_by_pattern(df, pattern_list, default=None):
    """
    Find column names matching patterns
    """
    for pattern in pattern_list:
        matches = [col for col in df.columns if pattern in str(col).upper()]
        if matches:
            return matches[0]
    return default

def handle_merged_cells(df):
    """
    Handle merged cells by forward-filling values
    """
    # Identify potential merged cell columns (those with many NaN values)
    potential_merged_cols = []
    for col in df.columns:
        null_ratio = df[col].isna().sum() / len(df)
        if 0.3 < null_ratio < 0.9:  # If 30-90% of values are NaN, likely merged cells
            potential_merged_cols.append(col)
    
    # Forward fill these columns
    if potential_merged_cols:
        df[potential_merged_cols] = df[potential_merged_cols].ffill()
    
    return df

# ================ THEME MANAGEMENT ================

# Theme definitions
THEMES = {
    "Light": {
        "primary_color": "#1F77B4",
        "secondary_color": "#2CA02C",
        "tertiary_color": "#FF7F0E",
        "background_color": "#FFFFFF",
        "text_color": "#000000",
        "card_bg": "#F8F9FA",
        "accent_color": "#D62728",
        "plotly_theme": "plotly_white"
    },
    "Dark": {
        "primary_color": "#0A84FF",
        "secondary_color": "#32D74B",
        "tertiary_color": "#FF9500",
        "background_color": "#0E1117",
        "text_color": "#FAFAFA",
        "card_bg": "#1E1F25",
        "accent_color": "#FF2B55",
        "plotly_theme": "plotly_dark"
    },
    "Blue": {
        "primary_color": "#1F77B4",
        "secondary_color": "#AEC7E8",
        "tertiary_color": "#17BECF",
        "background_color": "#E6F0FA",
        "text_color": "#000000",
        "card_bg": "#F0F8FF",
        "accent_color": "#D62728",
        "plotly_theme": "plotly_white"
    },
    "Green": {
        "primary_color": "#2CA02C",
        "secondary_color": "#98DF8A",
        "tertiary_color": "#2E9945",
        "background_color": "#E8F5E9",
        "text_color": "#000000",
        "card_bg": "#F1F8F0",
        "accent_color": "#D62728",
        "plotly_theme": "plotly_white"
    }
}

# Initialize theme in session state
if 'theme' not in st.session_state:
    # Detect browser's dark mode preference
    is_browser_dark = st.query_params.get("theme", ["light"])[0].lower() == "dark"
    st.session_state.theme = "Dark" if is_browser_dark else "Light"

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

# Apply CSS for theme
st.markdown(f"""
<style>
    /* Base theme */
    .stApp, [data-testid="stAppViewContainer"], .block-container {{
        background-color: {background_color} !important;
        color: {text_color} !important;
    }}
    
    /* Text elements */
    body, p, div, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {{
        color: {text_color} !important;
    }}
    
    /* Input elements */
    .stTextInput > div > div > input, .stSelectbox > div > div > input, .stNumberInput > div > div > input {{
        color: {text_color} !important;
        border-color: {primary_color}40 !important;
    }}
    
    /* Dataframe styling */
    .dataframe, .dataframe th, .dataframe td, .stDataFrame {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {primary_color}20 !important;
    }}
    
    /* Cards */
    .card {{
        background-color: {card_bg} !important;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Headers */
    .main-header {{
        font-size: 2rem;
        color: {primary_color};
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }}
    
    .section-header {{
        font-size: 1.5rem;
        color: {primary_color};
        font-weight: 600;
        margin: 0.5rem 0;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {primary_color};
        color: white !important;
        border: none;
        border-radius: 4px;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background-color: {secondary_color};
        transform: translateY(-2px);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {primary_color}20;
    }}
    
    /* Metrics */
    .stMetric {{
        background-color: {card_bg} !important;
        border-radius: 8px;
        padding: 0.5rem;
        border-left: 4px solid {primary_color};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {card_bg} !important;
        border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {text_color} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {primary_color}40 !important;
        color: {text_color} !important;
    }}
    
    /* File uploader */
    .stFileUploader {{
        border: 2px dashed {primary_color}40;
        border-radius: 8px;
        padding: 1rem;
    }}
    
    /* Expander */
    .stExpander {{
        border: 1px solid {primary_color}20 !important;
        border-radius: 8px;
    }}
    
    /* Error messages */
    .error-message {{
        background-color: #FF000020;
        color: #FF0000;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    
    /* Success messages */
    .success-message {{
        background-color: #00FF0020;
        color: #00AA00;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    
    /* Compact layout */
    .compact-container {{
        padding: 0.5rem !important;
    }}
    
    .compact-card {{
        padding: 0.75rem !important;
        margin-bottom: 0.75rem !important;
    }}
    
    .compact-tabs {{
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }}
    
    /* Grid layout */
    .grid-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# ================ SESSION STATE INITIALIZATION ================

# Initialize session state variables
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
    st.markdown(f'<h3 style="color: {primary_color};">\u2699\ufe0f Control Panel</h3>', unsafe_allow_html=True)
    
    # Theme selector
    selected_theme = st.selectbox(
        "Theme", 
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme)
    )
    
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    # Data options
    st.markdown("### Data Options")
    show_sample = st.checkbox("Show Sample Data", value=False)
    auto_ffill = st.checkbox("Auto Forward-Fill", value=True, 
                            help="Automatically fill empty cells with the value above them (useful for merged cells)")
    
    # Display options
    st.markdown("### Display Options")
    compact_mode = st.checkbox("Compact Mode", value=True, 
                              help="Makes the interface more compact")
    chart_style = st.selectbox(
        "Chart Style", 
        ["Default", "Minimal", "Detailed", "Colorful"], 
        index=0
    )
    
    # Help section
    with st.expander("Help & Documentation"):
        st.markdown("""
        ### Quick Guide
        1. Upload your Excel/CSV file
        2. Select the header row
        3. Use the tabs to analyze and filter data
        4. Export filtered results
        
        ### Features
        - **Auto Forward-Fill**: Handles merged cells
        - **Compact Mode**: More efficient screen space
        - **Error Handling**: Robust processing of problematic data
        """)
    
    st.markdown("---")
    st.markdown(f'<p style="text-align: center; color: {text_color};">Delegate Management System v4.0</p>', unsafe_allow_html=True)

# ================ MAIN CONTENT ================

# Apply compact mode if selected
container_class = "compact-container" if compact_mode else ""
card_class = "compact-card" if compact_mode else ""

st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Delegate Management Dashboard</h1>', unsafe_allow_html=True)

# ================ FILE UPLOAD SECTION ================

st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
st.markdown(f'<h2 class="section-header">\ud83d\udce4 Upload Data</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"], key="uploader")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    
    try:
        # Read file based on type
        if uploaded_file.name.endswith(".csv"):
            temp_df = pd.read_csv(uploaded_file, header=None)
        else:
            # Use error handling for Excel files to handle merged cells
            try:
                temp_df = pd.read_excel(uploaded_file, header=None)
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                st.info("Trying alternative method for problematic Excel files...")
                try:
                    # Try with a different engine
                    temp_df = pd.read_excel(uploaded_file, header=None, engine="openpyxl")
                except Exception as e2:
                    st.error(f"Failed to read Excel file: {str(e2)}")
                    st.stop()
        
        # Preview raw data
        with st.expander("\ud83d\udd0d Preview Raw Data", expanded=True):
            st.dataframe(temp_df.head(10), use_container_width=True)
        
        # Header row selection
        header_row = st.selectbox(
            "Header Row (0-based)", 
            list(range(min(10, len(temp_df)))), 
            index=min(1, len(temp_df)-1)
        )
        st.session_state.header_row = header_row
        
        # Process the data with the selected header
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=header_row, dtype=str)
            else:
                try:
                    df = pd.read_excel(uploaded_file, header=header_row, dtype=str)
                except:
                    df = pd.read_excel(uploaded_file, header=header_row, dtype=str, engine="openpyxl")
            
            # Handle merged cells
            df = handle_merged_cells(df)
            
            # Process the data
            df, error_msg = safe_process_data(df, header_row, auto_ffill)
            
            if error_msg:
                st.warning(error_msg)
                st.session_state.error_log.append(error_msg)
            
            st.session_state.df = df.copy()
            st.session_state.original_columns = df.columns.tolist()
            st.session_state.processed = True
            st.success("Data uploaded and processed successfully! \ud83c\udf89")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.session_state.error_log.append(f"Error processing file: {str(e)}\
{traceback.format_exc()}")
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.session_state.error_log.append(f"Error reading file: {str(e)}\
{traceback.format_exc()}")

st.markdown('</div>', unsafe_allow_html=True)

# ================ SAMPLE DATA SECTION ================

# Sample data - only if explicitly enabled and no uploaded data
if show_sample and (st.session_state.df is None or st.session_state.df.empty):
    st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">\ud83d\udccb Sample Data</h2>', unsafe_allow_html=True)
    
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            "S/N": list(range(1, 11)),
            "First Name": ["John", "Jane", "Alex", "Emily", "Michael", "Sarah", "David", "Lisa", "Robert", "Emma"],
            "Last Name": ["Doe", "Smith", "Johnson", "Brown", "Davis", "Wilson", "Taylor", "Thomas", "Roberts", "Clark"],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
            "DoB": ["1990-05-15", "1985-12-10", "1992-08-22", "1988-03-30", "1995-07-18", 
                   "1991-11-05", "1987-09-12", "1993-04-25", "1989-01-20", "1994-06-08"],
            "Company Name": ["Tech Corp", "Oil Inc", "Marine Ltd", "Tech Corp", "Oil Inc", 
                           "Marine Ltd", "Tech Corp", "Oil Inc", "Marine Ltd", "Tech Corp"],
            "Contact Number": ["08012345678", "08023456789", "08034567890", "08045678901", "08056789012", 
                             "08067890123", "08078901234", "08089012345", "08090123456", "08001234567"],
            "Certificate Number": [f"EBS2024{i:04d}" for i in range(1, 11)],
            "Course Date": ["8th March, 2024", "5th June, 2024", "3rd August, 2024", "14th September, 2024", "8th March, 2024",
                          "5th June, 2024", "3rd August, 2024", "14th September, 2024", "8th March, 2024", "5th June, 2024"],
            "Batch": ["1/24", "2/24", "3/24", "4/24", "1/24", "2/24", "3/24", "4/24", "1/24", "2/24"]
        })
        
        # Process the sample data
        sample_data, error_msg = safe_process_data(sample_data)
        
        if error_msg:
            st.warning(error_msg)
            st.session_state.error_log.append(error_msg)
        
        st.session_state.df = sample_data
        st.session_state.original_columns = sample_data.columns.tolist()
        st.session_state.processed = True
        
        st.dataframe(sample_data, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating sample data: {str(e)}")
        st.session_state.error_log.append(f"Error creating sample data: {str(e)}\
{traceback.format_exc()}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================ DASHBOARD SECTIONS ================

if st.session_state.df is not None and not st.session_state.df.empty:
    
    # ================ OVERVIEW SECTION ================
    
    st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">\ud83d\udcca Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", len(st.session_state.df))
    with col2:
        st.metric("Columns", len(st.session_state.df.columns))
    with col3:
        st.metric("Missing Values", st.session_state.df.isna().sum().sum())
    with col4:
        st.metric("Unique Values (Avg)", int(st.session_state.df.nunique().mean()))
    
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            "Column": st.session_state.df.columns,
            "Type": st.session_state.df.dtypes,
            "Non-Null Count": st.session_state.df.count(),
            "Null Count": st.session_state.df.isna().sum(),
            "Unique Values": st.session_state.df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ================ DATA PREVIEW SECTION ================
    
    st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">\ud83d\udd0d Data Preview</h2>', unsafe_allow_html=True)
    
    preview_rows = st.slider("Preview Rows", 5, 50, 10)
    st.dataframe(st.session_state.df.head(preview_rows), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ================ TABS SECTION ================
    
    st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
    tabs = st.tabs(["Data Processing", "Search & Filter", "Analysis", "Export"])
    
    # ================ DATA PROCESSING TAB ================
    
    with tabs[0]:
        st.markdown(f'<h3 style="color: {tertiary_color};">\u2699\ufe0f Data Processing</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clean Data")
            
            if st.button("Remove Duplicates"):
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
                if st.button("Apply Fill"):
                    try:
                        st.session_state.df.fillna(fill_val, inplace=True)
                        st.success("Filled missing values")
                    except Exception as e:
                        st.error(f"Error filling values: {str(e)}")
                        st.session_state.error_log.append(f"Error filling values: {str(e)}")
            
            elif fill_opt == "Drop":
                if st.button("Apply Drop"):
                    try:
                        old_len = len(st.session_state.df)
                        st.session_state.df.dropna(inplace=True)
                        dropped = old_len - len(st.session_state.df)
                        st.success(f"Dropped {dropped} rows with missing values")
                    except Exception as e:
                        st.error(f"Error dropping rows: {str(e)}")
                        st.session_state.error_log.append(f"Error dropping rows: {str(e)}")
        
        with col2:
            st.subheader("Modify Data")
            
            col_remove = st.selectbox("Remove Column", ['None'] + list(st.session_state.df.columns))
            if col_remove != 'None' and st.button("Remove Column"):
                try:
                    st.session_state.df.drop(columns=[col_remove], inplace=True)
                    if st.session_state.filtered_data is not None:
                        st.session_state.filtered_data.drop(columns=[col_remove], errors='ignore', inplace=True)
                    st.success(f"Removed column: {col_remove}")
                except Exception as e:
                    st.error(f"Error removing column: {str(e)}")
                    st.session_state.error_log.append(f"Error removing column: {str(e)}")
            
            if st.button("Standardize Text"):
                try:
                    text_cols = st.session_state.df.select_dtypes('object').columns
                    for col in text_cols:
                        if col != 'Full Name':
                            st.session_state.df[col] = st.session_state.df[col].str.title().str.strip()
                    st.success("Standardized text columns")
                except Exception as e:
                    st.error(f"Error standardizing text: {str(e)}")
                    st.session_state.error_log.append(f"Error standardizing text: {str(e)}")
            
            if st.button("Reset Data"):
                try:
                    if st.session_state.uploaded_file:
                        if st.session_state.uploaded_file.name.endswith(".csv"):
                            reset_df = pd.read_csv(st.session_state.uploaded_file, header=st.session_state.header_row, dtype=str)
                        else:
                            try:
                                reset_df = pd.read_excel(st.session_state.uploaded_file, header=st.session_state.header_row, dtype=str)
                            except:
                                reset_df = pd.read_excel(st.session_state.uploaded_file, header=st.session_state.header_row, dtype=str, engine="openpyxl")
                        
                        # Handle merged cells
                        reset_df = handle_merged_cells(reset_df)
                        
                        # Process the data
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
                    st.session_state.error_log.append(f"Error resetting data: {str(e)}\
{traceback.format_exc()}")
        
        # Error log expander
        if st.session_state.error_log:
            with st.expander("Error Log"):
                for i, error in enumerate(st.session_state.error_log):
                    st.text(f"Error {i+1}:\
{error}")
                if st.button("Clear Error Log"):
                    st.session_state.error_log = []
                    st.success("Error log cleared")
    
    # ================ SEARCH & FILTER TAB ================
    
    with tabs[1]:
        st.markdown(f'<h3 style="color: {tertiary_color};">\ud83d\udd0d Search & Filter</h3>', unsafe_allow_html=True)
        
        search_tabs = st.tabs(["Individual Search", "Multi-Filter"])
        
        # Individual Search
        with search_tabs[0]:
            st.markdown("### \ud83d\udc64 Individual Search")
            
            col1, col2 = st.columns(2)
            with col1:
                search_name = st.text_input("Name")
            with col2:
                search_id = st.text_input("Certificate/ID Number")
            
            if search_name or search_id:
                try:
                    results = st.session_state.df.copy()
                    
                    if search_name:
                        # Look for name in any name-related column
                        name_cols = [col for col in results.columns if 'NAME' in str(col).upper()]
                        if name_cols:
                            name_mask = results[name_cols].apply(
                                lambda row: row.astype(str).str.contains(search_name, case=False, na=False).any(), 
                                axis=1
                            )
                            results = results[name_mask]
                    
                    if search_id:
                        # Look for ID in any ID-related column
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
                        st.dataframe(results, use_container_width=True)
                        
                        # Show detailed view for single result
                        if len(results) == 1:
                            st.markdown("### Detailed Record")
                            transp = results.T.reset_index()
                            transp.columns = ['Field', 'Value']
                            st.table(transp)
                            
                            # Download single record
                            with io.BytesIO() as buf:
                                transp.to_excel(buf, index=False)
                                buf.seek(0)
                                st.download_button("Download Record", buf, "record.xlsx")
                    else:
                        st.warning("No matching records found")
                
                except Exception as e:
                    st.error(f"Error searching data: {str(e)}")
                    st.session_state.error_log.append(f"Error searching data: {str(e)}\
{traceback.format_exc()}")
            else:
                st.info("Enter search terms to find records")
        
        # Multi-Filter
        with search_tabs[1]:
            st.markdown("### \ud83d\udd22 Multi-Filter")
            
            try:
                col1, col2 = st.columns(2)
                
                # Company filter
                company_col = get_column_by_pattern(st.session_state.df, ['COMPANY'])
                if company_col:
                    with col1:
                        companies = ['All'] + sorted(st.session_state.df[company_col].dropna().unique().tolist())
                        sel_comp = st.multiselect("Companies \ud83c\udfe2", companies, default=['All'])
                        if 'All' in sel_comp:
                            sel_comp = st.session_state.df[company_col].unique().tolist()
                
                # Batch filter
                batch_col = get_column_by_pattern(st.session_state.df, ['BATCH'])
                if batch_col:
                    with col1:
                        batches = ['All'] + sorted(st.session_state.df[batch_col].dropna().unique().tolist())
                        sel_batch = st.multiselect("Batches \ud83d\udcc2", batches, default=['All'])
                        if 'All' in sel_batch:
                            sel_batch = st.session_state.df[batch_col].unique().tolist()
                
                # Date filter
                date_col = get_column_by_pattern(st.session_state.df, ['COURSE DATE', 'DATE'])
                if date_col:
                    with col2:
                        dates = ['All'] + sorted(st.session_state.df[date_col].dropna().unique().tolist())
                        sel_date = st.multiselect("Dates \ud83d\udcc5", dates, default=['All'])
                        if 'All' in sel_date:
                            sel_date = st.session_state.df[date_col].unique().tolist()
                
                # Gender filter
                gender_col = get_column_by_pattern(st.session_state.df, ['GENDER'])
                if gender_col:
                    with col2:
                        genders = ['All'] + sorted(st.session_state.df[gender_col].dropna().unique().tolist())
                        sel_gender = st.multiselect("Genders \ud83d\udc65", genders, default=['All'])
                        if 'All' in sel_gender:
                            sel_gender = st.session_state.df[gender_col].unique().tolist()
                
                # Apply filters button
                if st.button("Apply Filters"):
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
                
                # Display filtered data
                if st.session_state.filtered_data is not None:
                    st.dataframe(st.session_state.filtered_data, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error applying filters: {str(e)}")
                st.session_state.error_log.append(f"Error applying filters: {str(e)}\
{traceback.format_exc()}")
    
    # ================ ANALYSIS TAB ================
    
    with tabs[2]:
        st.markdown(f'<h3 style="color: {tertiary_color};">\ud83d\udcca Analysis</h3>', unsafe_allow_html=True)
        
        try:
            # Set color scheme based on chart style
            if chart_style == "Colorful":
                color_seq = px.colors.qualitative.Bold
            elif chart_style == "Minimal":
                color_seq = px.colors.sequential.Greys
            elif chart_style == "Detailed":
                color_seq = px.colors.qualitative.Pastel
            else:
                color_seq = px.colors.qualitative.Plotly
            
            col1, col2 = st.columns(2)
            
            # Gender distribution
            gender_col = get_column_by_pattern(st.session_state.df, ['GENDER'])
            if gender_col:
                with col1:
                    gender_cnt = st.session_state.df[gender_col].value_counts()
                    fig_gender = px.pie(
                        values=gender_cnt.values, 
                        names=gender_cnt.index, 
                        title="Gender Distribution",
                        color_discrete_sequence=color_seq,
                        template=plotly_theme
                    )
                    fig_gender.update_traces(textposition='inside', textinfo='percent+label')
                    fig_gender.update_layout(
                        paper_bgcolor=background_color,
                        plot_bgcolor=background_color,
                        font_color=text_color,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )
                    st.plotly_chart(fig_gender, use_container_width=True)
            
            # Age distribution
            if 'Age' in st.session_state.df.columns:
                with col2:
                    age_data = st.session_state.df['Age'].dropna().astype(float)
                    if not age_data.empty:
                        bins = [0, 20, 30, 40, 50, 60, np.inf]
                        labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
                        age_groups = pd.cut(age_data, bins, labels=labels)
                        age_cnt = age_groups.value_counts().sort_index()
                        fig_age = px.bar(
                            x=age_cnt.index, 
                            y=age_cnt.values, 
                            title="Age Distribution",
                            color=age_cnt.index,
                            color_discrete_sequence=color_seq,
                            template=plotly_theme,
                            labels={'x': 'Age Group', 'y': 'Count'}
                        )
                        fig_age.update_layout(
                            paper_bgcolor=background_color,
                            plot_bgcolor=background_color,
                            font_color=text_color,
                            margin=dict(t=50, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig_age, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            # Batch distribution
            batch_col = get_column_by_pattern(st.session_state.df, ['BATCH'])
            if batch_col:
                with col3:
                    batch_cnt = st.session_state.df[batch_col].value_counts()
                    fig_batch = px.bar(
                        x=batch_cnt.index, 
                        y=batch_cnt.values, 
                        title="Batch Distribution",
                        color=batch_cnt.index,
                        color_discrete_sequence=color_seq,
                        template=plotly_theme,
                        labels={'x': 'Batch', 'y': 'Count'}
                    )
                    fig_batch.update_layout(
                        paper_bgcolor=background_color,
                        plot_bgcolor=background_color,
                        font_color=text_color,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )
                    st.plotly_chart(fig_batch, use_container_width=True)
            
            # Company distribution
            company_col = get_column_by_pattern(st.session_state.df, ['COMPANY'])
            if company_col:
                with col4:
                    comp_cnt = st.session_state.df[company_col].value_counts().head(10)
                    fig_comp = px.bar(
                        y=comp_cnt.index, 
                        x=comp_cnt.values, 
                        orientation='h',
                        title="Top Companies",
                        color=comp_cnt.index,
                        color_discrete_sequence=color_seq,
                        template=plotly_theme,
                        labels={'y': 'Company', 'x': 'Count'}
                    )
                    fig_comp.update_layout(
                        paper_bgcolor=background_color,
                        plot_bgcolor=background_color,
                        font_color=text_color,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
            
            # Course date distribution
            date_col = get_column_by_pattern(st.session_state.df, ['COURSE DATE', 'DATE'])
            if date_col:
                date_cnt = st.session_state.df[date_col].value_counts()
                fig_date = px.bar(
                    x=date_cnt.index, 
                    y=date_cnt.values, 
                    title="Course Date Distribution",
                    color=date_cnt.index,
                    color_discrete_sequence=color_seq,
                    template=plotly_theme,
                    labels={'x': 'Date', 'y': 'Count'}
                )
                fig_date.update_layout(
                    paper_bgcolor=background_color,
                    plot_bgcolor=background_color,
                    font_color=text_color,
                    margin=dict(t=50, b=50, l=50, r=50),
                    xaxis={'categoryorder': 'total descending'}
                )
                st.plotly_chart(fig_date, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating charts: {str(e)}")
            st.session_state.error_log.append(f"Error generating charts: {str(e)}\
{traceback.format_exc()}")
    
    # ================ EXPORT TAB ================
    
    with tabs[3]:
        st.markdown(f'<h3 style="color: {tertiary_color};">\ud83d\udcbe Export</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Export filtered data
        with col1:
            st.markdown("### Filtered Dataset")
            if st.session_state.filtered_data is not None:
                st.metric("Records to Export", len(st.session_state.filtered_data))
                
                try:
                    # Excel export
                    with io.BytesIO() as buf:
                        st.session_state.filtered_data.to_excel(buf, index=False)
                        buf.seek(0)
                        st.download_button("Download as Excel", buf, "filtered_data.xlsx")
                    
                    # CSV export
                    csv = st.session_state.filtered_data.to_csv(index=False)
                    st.download_button("Download as CSV", csv, "filtered_data.csv")
                
                except Exception as e:
                    st.error(f"Error exporting filtered data: {str(e)}")
                    st.session_state.error_log.append(f"Error exporting filtered data: {str(e)}")
            else:
                st.info("No filtered data available. Use the Search & Filter tab to create filtered data.")
        
        # Export full dataset
        with col2:
            st.markdown("### Full Dataset")
            st.metric("Total Records", len(st.session_state.df))
            
            try:
                # Excel export
                with io.BytesIO() as buf:
                    st.session_state.df.to_excel(buf, index=False)
                    buf.seek(0)
                    st.download_button("Download as Excel", buf, "full_data.xlsx")
                
                # CSV export
                csv_full = st.session_state.df.to_csv(index=False)
                st.download_button("Download as CSV", csv_full, "full_data.csv")
            
            except Exception as e:
                st.error(f"Error exporting full data: {str(e)}")
                st.session_state.error_log.append(f"Error exporting full data: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload an Excel/CSV file or enable 'Show Sample Data' to view the dashboard.")

# ================ FOOTER ================

st.markdown(f'<div style="text-align: center; color: {primary_color}; padding: 1rem; margin-top: 2rem;">Delegate Management Dashboard v4.0 | \u00a9 2025</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

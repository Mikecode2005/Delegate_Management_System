import streamlit as st
import pandas as pd
import io
import plotly.express as px
from datetime import datetime
import numpy as np
import random

# Set page configuration
st.set_page_config(
    page_title="Delegate Management System",
    layout="wide",
    page_icon="👥",
    initial_sidebar_state="expanded"
)

# Detect current theme
config = st._config.get_option("theme.base")  # "light" or "dark"

# Define color schemes with more vibrancy
if config == "dark":
    primary_color = "#0A84FF"  # Bright Blue
    secondary_color = "#32D74B"  # Bright Green
    tertiary_color = "#FF9F0A"  # Orange accent
    background_color = "#0E1117"
    text_color = "#FAFAFA"
    accent_color = "#FF2B55"  # Red accent
    card_bg = "#1E1F25"
else:
    primary_color = "#1F77B4"  # Blue
    secondary_color = "#2CA02C"  # Green
    tertiary_color = "#FF7F0E"  # Orange
    background_color = "#FFFFFF"
    text_color = "#000000"
    accent_color = "#D62728"  # Red
    card_bg = "#F8F9FA"

# Enhanced Custom CSS with more colors, animations, and better formatting
st.markdown(f"""
<style>
    /* Global styles */
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}

    /* Main headers with animation */
    .main-header {{
        font-size: 3rem;
        color: {primary_color};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {primary_color}, {secondary_color}, {tertiary_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
        animation: fadeIn 1s ease-in-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}

    /* Subheaders with hover effect */
    .subheader {{
        font-size: 1.8rem;
        color: {primary_color};
        border-bottom: 3px solid {secondary_color};
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        transition: color 0.3s ease;
    }}

    .subheader:hover {{
        color: {tertiary_color};
    }}

    /* Buttons with gradient and shadow */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(135deg, {primary_color}, {secondary_color});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    .stButton>button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, {secondary_color}, {primary_color});
    }}

    .stDownloadButton>button {{
        width: 100%;
        background: linear-gradient(135deg, {tertiary_color}, {accent_color});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    .stDownloadButton>button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }}

    /* Cards for sections */
    .card {{
        background-color: {card_bg};
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }}

    .card:hover {{
        transform: translateY(-5px);
    }}

    /* Metrics with colors */
    .stMetric {{
        background-color: {card_bg};
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid {primary_color};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}

    /* Tabs with color accents */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background-color: {card_bg};
        padding: 0.5rem;
        border-radius: 12px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: linear-gradient(135deg, {primary_color}10, {secondary_color}10);
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: {text_color};
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {primary_color}, {secondary_color});
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}

    /* Dataframes with hover */
    .dataframe {{
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        overflow: auto;
    }}

    .dataframe tr:hover {{
        background-color: {primary_color}10;
    }}

    /* Sidebar enhancements */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {primary_color}20, {secondary_color}20);
        padding: 1rem;
        border-right: 1px solid {primary_color}20;
    }}

    /* File uploader styled */
    .stFileUploader {{
        border: 2px dashed {primary_color};
        border-radius: 12px;
        padding: 20px;
        background-color: {card_bg};
        text-align: center;
    }}

    /* Messages with gradients */
    .stSuccess {{
        border-radius: 8px;
        background: linear-gradient(135deg, {secondary_color}20, {secondary_color}40);
        color: {text_color};
    }}

    .stWarning {{
        border-radius: 8px;
        background: linear-gradient(135deg, {accent_color}20, {accent_color}40);
        color: {text_color};
    }}

    .stInfo {{
        border-radius: 8px;
        background: linear-gradient(135deg, {primary_color}20, {primary_color}40);
        color: {text_color};
    }}

    /* Expander enhancements */
    .stExpander {{
        border: 1px solid {primary_color}20;
        border-radius: 8px;
        margin-bottom: 1rem;
    }}

    .stExpander summary {{
        font-weight: 600;
        color: {primary_color};
    }}

    /* Selectboxes */
    .stSelectbox div[role="combobox"], .stMultiselect div[role="combobox"] {{
        background-color: {card_bg};
        border: 1px solid {primary_color}20;
        border-radius: 4px;
        color: {text_color};
    }}
</style>
""", unsafe_allow_html=True)

# App title with emoji
st.markdown('<h1 class="main-header">👥 Delegate Management System</h1>', unsafe_allow_html=True)

# Sidebar for additional options
with st.sidebar:
    st.markdown(f'<h3 style="color: {primary_color};">⚙️ Settings</h3>', unsafe_allow_html=True)
    theme = st.selectbox("Color Theme Override", ["Auto", "Blue", "Green", "Purple", "Vibrant"], index=0)
    if theme != "Auto":
        if theme == "Blue":
            primary_color = "#1F77B4"
            secondary_color = "#AEC7E8"
            tertiary_color = "#17BECF"
        elif theme == "Green":
            primary_color = "#2CA02C"
            secondary_color = "#98DF8A"
            tertiary_color = "#2E9945"
        elif theme == "Purple":
            primary_color = "#9467BD"
            secondary_color = "#C5B0D5"
            tertiary_color = "#756BB1"
        elif theme == "Vibrant":
            primary_color = "#FF9500"
            secondary_color = "#FF2D55"
            tertiary_color = "#5856D6"
    st.markdown("---")
    st.markdown(f'<h3 style="color: {primary_color};">📊 Data Options</h3>', unsafe_allow_html=True)
    show_sample = st.checkbox("Show Sample Data", value=True)
    auto_ffill = st.checkbox("Auto Forward-Fill Merged Cells", value=True)
    st.markdown("---")
    st.markdown(f'<h3 style="color: {primary_color};">📈 Visualization Options</h3>', unsafe_allow_html=True)
    chart_style = st.selectbox("Chart Style", ["Default", "Minimal", "Detailed", "Colorful"], index=3)
    show_charts = st.checkbox("Show Advanced Charts", value=True)
    st.markdown("---")
    st.markdown(f'<h3 style="color: {primary_color};">🔗 Quick Links</h3>', unsafe_allow_html=True)
    st.button("View Documentation")
    st.button("Contact Support")
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit*")

# File uploader in a card
st.markdown(f'<div class="card"><h2 class="subheader">📤 Upload Your Delegate Data</h2></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["xlsx", "csv"], help="Upload Excel or CSV file with delegate information")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'original_columns' not in st.session_state:
    st.session_state.original_columns = None

header_row = None

if uploaded_file:
    # Initial read to detect possible header
    if uploaded_file.name.endswith(".csv"):
        temp_df = pd.read_csv(uploaded_file, header=None)
    else:
        temp_df = pd.read_excel(uploaded_file, header=None)

    # Show preview in expander
    with st.expander("🔍 Raw Data Preview (First 10 Rows)", expanded=True):
        st.dataframe(temp_df.head(10), use_container_width=True)

    # Header selection with default to 1
    header_options = list(range(len(temp_df.head(10))))
    header_row = st.selectbox(
        "Select Header Row (0-based index)",
        options=header_options,
        index=1,
        help="Choose which row contains the column headers. Often the second row if first is title."
    )

    # Read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=header_row, dtype=str)
    else:
        df = pd.read_excel(uploaded_file, header=header_row, dtype=str)

    # Clean columns
    df.columns = [str(col).strip().replace('\n', ' ').title() for col in df.columns]

    # Auto ffill
    if auto_ffill:
        ffill_cols = [col for col in df.columns if any(x in str(col).upper() for x in ['BATCH', 'COURSE DATE', 'ISSUED DATE', 'EXPIRY DATE', 'DATE'])]
        if ffill_cols:
            df[ffill_cols] = df[ffill_cols].ffill()

    # Phone preservation
    for col in df.columns:
        if any(x in str(col).upper() for x in ['NUMBER', 'PHONE', 'NOK', 'CONTACT', 'TEL']):
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    st.session_state.original_columns = df.columns.tolist()

    # Full name
    first_name_col = next((col for col in df.columns if 'FIRST NAME' in str(col).upper()), None)
    last_name_col = next((col for col in df.columns if 'LAST NAME' in str(col).upper()), None)
    if first_name_col and last_name_col:
        df['Full Name'] = df[first_name_col].fillna('') + ' ' + df[last_name_col].fillna('')

    # Age calculation
    dob_col = next((col for col in df.columns if 'DOB' in str(col).upper() or 'BIRTH' in str(col).upper()), None)
    if dob_col:
        try:
            df[dob_col] = pd.to_datetime(df[dob_col], errors='coerce')
            df['Age'] = ((datetime.now() - df[dob_col]).dt.days // 365.25).astype('Int64')
        except:
            st.warning("Unable to calculate ages.")

    st.session_state.df = df.copy()

    # Overview in card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df), delta=None, delta_color="normal")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum(), delta_color="inverse")
    with col4:
        st.metric("Unique Values (Avg)", int(df.nunique().mean()))
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📝 Original Column Names", expanded=False):
        st.write(st.session_state.original_columns)

    with st.expander("📊 Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    # Preprocessing in card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">⚙️ Data Preprocessing</h2>', unsafe_allow_html=True)
    prep_col1, prep_col2 = st.columns(2)

    with prep_col1:
        st.markdown(f'<h3 style="color: {secondary_color};">🧹 Cleaning Options</h3>', unsafe_allow_html=True)
        if st.button("Remove Duplicates"):
            old_len = len(st.session_state.df)
            st.session_state.df.drop_duplicates(inplace=True)
            st.success(f"Removed {old_len - len(st.session_state.df)} duplicates")

        fill_opt = st.radio("Missing Values", ["Keep", "Fill", "Drop"])
        if fill_opt == "Fill":
            fill_val = st.text_input("Fill Value", "N/A")
            if st.button("Apply"):
                st.session_state.df.fillna(fill_val, inplace=True)
                st.success("Filled missing values")
        elif fill_opt == "Drop":
            if st.button("Apply"):
                old_len = len(st.session_state.df)
                st.session_state.df.dropna(inplace=True)
                st.success(f"Dropped {old_len - len(st.session_state.df)} rows")

        col_remove = st.selectbox("Remove Column", ['None'] + list(st.session_state.df.columns))
        if col_remove != 'None' and st.button("Remove"):
            st.session_state.df.drop(columns=[col_remove], inplace=True)
            if st.session_state.filtered_data is not None:
                st.session_state.filtered_data.drop(columns=[col_remove], errors='ignore', inplace=True)
            st.success(f"Removed {col_remove}")

    with prep_col2:
        st.markdown(f'<h3 style="color: {secondary_color};">📝 Transformation Options</h3>', unsafe_allow_html=True)
        if st.button("Standardize Text"):
            text_cols = st.session_state.df.select_dtypes('object').columns
            for col in text_cols:
                if col != 'Full Name':
                    st.session_state.df[col] = st.session_state.df[col].str.title().str.strip()
            st.success("Standardized text")

        if st.button("Reset to Original"):
            if uploaded_file.name.endswith(".csv"):
                reset_df = pd.read_csv(uploaded_file, header=header_row, dtype=str)
            else:
                reset_df = pd.read_excel(uploaded_file, header=header_row, dtype=str)
            reset_df.columns = [str(col).strip().replace('\n', ' ').title() for col in reset_df.columns]
            if auto_ffill:
                ffill_cols = [col for col in reset_df.columns if any(x in str(col).upper() for x in ['BATCH', 'COURSE DATE', 'ISSUED DATE', 'EXPIRY DATE', 'DATE'])]
                reset_df[ffill_cols] = reset_df[ffill_cols].ffill()
            for col in reset_df.columns:
                if any(x in str(col).upper() for x in ['NUMBER', 'PHONE', 'NOK', 'CONTACT', 'TEL']):
                    reset_df[col] = reset_df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            st.session_state.df = reset_df.copy()
            st.success("Reset successful")

    st.session_state.processed = True

    if st.session_state.processed:
        st.success("✅ Data Processed Successfully!", icon="🎉")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processed Records", len(st.session_state.df), delta_color="normal")
        with col2:
            st.metric("Memory Usage", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔍 View Processed Data", expanded=False):
        st.dataframe(st.session_state.df, use_container_width=True)

    # Search and Filter in card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">🔍 Search & Filter</h2>', unsafe_allow_html=True)

    tabs = st.tabs(["🔎 Individual Search", "🎚️ Multi-Filter", "📊 Analysis", "💾 Export"])

    with tabs[0]:
        st.markdown(f'<h3 style="color: {tertiary_color};">👤 Individual Search</h3>', unsafe_allow_html=True)
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            search_name = st.text_input("Search by Name", help="Enter name to search")
        with s_col2:
            search_id = st.text_input("Search by Cert No.", help="Enter certificate number")

        if search_name or search_id:
            results = st.session_state.df.copy()
            if search_name:
                name_cols = [col for col in results.columns if 'NAME' in str(col).upper()]
                if name_cols:
                    mask = results[name_cols].apply(lambda row: row.str.contains(search_name, case=False, na=False).any(), axis=1)
                    results = results[mask]
            if search_id:
                cert_cols = [col for col in results.columns if 'CERT' in str(col).upper() or 'ID' in str(col).upper()]
                if cert_cols:
                    mask = results[cert_cols].apply(lambda row: row.str.contains(search_id, case=False, na=False).any(), axis=1)
                    results = results[mask]
            if not results.empty:
                st.success(f"Found {len(results)} matching records! 🎯")
                st.dataframe(results, use_container_width=True)
                st.session_state.filtered_data = results
                if len(results) == 1:
                    transp = results.T.reset_index()
                    transp.columns = ['Field', 'Value']
                    st.markdown(f'<h4 style="color: {secondary_color};">Individual Record</h4>', unsafe_allow_html=True)
                    st.table(transp)
                    with io.BytesIO() as buf:
                        transp.to_excel(buf, index=False)
                        buf.seek(0)
                        st.download_button("📥 Download Record", buf, "delegate_record.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with io.BytesIO() as buf:
                    results.to_excel(buf, index=False)
                    buf.seek(0)
                    st.download_button("📥 Download Matches", buf, "matching_records.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No matching records found. Try different search terms.")

    with tabs[1]:
        st.markdown(f'<h3 style="color: {tertiary_color};">🎚️ Multi-Filter</h3>', unsafe_allow_html=True)
        f_col1, f_col2 = st.columns(2)

        company_col = next((col for col in st.session_state.df.columns if 'COMPANY' in str(col).upper()), None)
        if company_col:
            with f_col1:
                companies = ['All'] + sorted(st.session_state.df[company_col].dropna().unique().tolist())
                sel_comp = st.multiselect("Select Companies 🏢", companies, default=['All'])
                if 'All' in sel_comp:
                    sel_comp = st.session_state.df[company_col].unique().tolist()

        batch_col = next((col for col in st.session_state.df.columns if 'BATCH' in str(col).upper()), None)
        if batch_col:
            with f_col1:
                batches = ['All'] + sorted(st.session_state.df[batch_col].dropna().unique().tolist())
                sel_batch = st.multiselect("Select Batches 🗂️", batches, default=['All'])
                if 'All' in sel_batch:
                    sel_batch = st.session_state.df[batch_col].unique().tolist()

        date_col = next((col for col in st.session_state.df.columns if 'DATE' in str(col).upper()), None)
        if date_col:
            with f_col2:
                dates = ['All'] + sorted(st.session_state.df[date_col].dropna().unique().tolist())
                sel_date = st.multiselect("Select Dates 📅", dates, default=['All'])
                if 'All' in sel_date:
                    sel_date = st.session_state.df[date_col].unique().tolist()

        gender_col = next((col for col in st.session_state.df.columns if 'GENDER' in str(col).upper()), None)
        if gender_col:
            with f_col2:
                genders = ['All'] + sorted(st.session_state.df[gender_col].dropna().unique().tolist())
                sel_gender = st.multiselect("Select Genders 👥", genders, default=['All'])
                if 'All' in sel_gender:
                    sel_gender = st.session_state.df[gender_col].unique().tolist()

        if st.button("Apply Filters 🔍"):
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
            st.success(f"Filter applied! Found {len(filtered)} records. 📊")

        if st.session_state.filtered_data is not None:
            st.markdown(f'<h4 style="color: {secondary_color};">Filtered Results</h4>', unsafe_allow_html=True)
            st.dataframe(st.session_state.filtered_data, use_container_width=True)

    with tabs[2]:
        st.markdown(f'<h3 style="color: {tertiary_color};">📊 Data Analysis & Visualizations</h3>', unsafe_allow_html=True)
        # Use colorful schemes for charts
        color_seq = px.colors.sequential.Rainbow if chart_style == "Colorful" else px.colors.qualitative.Set3

        if gender_col:
            gender_cnt = st.session_state.df[gender_col].value_counts()
            fig_gender = px.pie(values=gender_cnt.values, names=gender_cnt.index, title="Gender Distribution 👥", color_discrete_sequence=color_seq)
            fig_gender.update_traces(textposition='inside', textinfo='percent+label')
            fig_gender.update_layout(showlegend=True, legend_title_text='Gender')
            st.plotly_chart(fig_gender, use_container_width=True)

        if 'Age' in st.session_state.df.columns:
            age_data = st.session_state.df['Age'].dropna().astype(float)
            if not age_data.empty:
                bins = [0, 20, 30, 40, 50, 60, np.inf]
                labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
                age_groups = pd.cut(age_data, bins, labels=labels)
                age_cnt = age_groups.value_counts().sort_index()
                fig_age = px.bar(x=age_cnt.index, y=age_cnt.values, title="Age Group Distribution 📈", color=age_cnt.index, color_discrete_sequence=color_seq)
                fig_age.update_layout(xaxis_title="Age Group", yaxis_title="Count")
                st.plotly_chart(fig_age, use_container_width=True)

        if batch_col:
            batch_cnt = st.session_state.df[batch_col].value_counts()
            fig_batch = px.bar(x=batch_cnt.index, y=batch_cnt.values, title="Delegates by Batch 🗂️", color=batch_cnt.index, color_discrete_sequence=color_seq)
            fig_batch.update_layout(xaxis_title="Batch", yaxis_title="Count")
            st.plotly_chart(fig_batch, use_container_width=True)

        if company_col:
            comp_cnt = st.session_state.df[company_col].value_counts().head(10)
            fig_comp = px.bar(y=comp_cnt.index, x=comp_cnt.values, orientation='h', title="Top 10 Companies 🏢", color=comp_cnt.index, color_discrete_sequence=color_seq)
            fig_comp.update_layout(yaxis_title="Company", xaxis_title="Count")
            st.plotly_chart(fig_comp, use_container_width=True)

    with tabs[3]:
        st.markdown(f'<h3 style="color: {tertiary_color};">💾 Export Data</h3>', unsafe_allow_html=True)
        if st.session_state.filtered_data is not None:
            st.metric("Records to Export", len(st.session_state.filtered_data))
            with io.BytesIO() as buf:
                st.session_state.filtered_data.to_excel(buf, index=False)
                buf.seek(0)
                st.download_button("📊 Download Filtered (Excel)", buf, "filtered_delegates.xlsx")
            csv = st.session_state.filtered_data.to_csv(index=False)
            st.download_button("📝 Download Filtered (CSV)", csv, "filtered_delegates.csv")

        st.markdown(f'<h4 style="color: {secondary_color};">Full Dataset Export</h4>', unsafe_allow_html=True)
        with io.BytesIO() as buf:
            st.session_state.df.to_excel(buf, index=False)
            buf.seek(0)
            st.download_button("📊 Download Full (Excel)", buf, "full_delegates.xlsx")
        csv_full = st.session_state.df.to_csv(index=False)
        st.download_button("📝 Download Full (CSV)", csv_full, "full_delegates.csv")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    if show_sample:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="subheader">📋 Sample Data Format</h2>', unsafe_allow_html=True)
        # Randomized sample
        first_names = ["John", "Jane", "Alex", "Emily", "Michael", "Sarah", "David", "Laura", "Robert", "Anna"]
        middle_names = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J."]
        last_names = ["Doe", "Smith", "Johnson", "Brown", "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas"]
        genders = ["Male", "Female"]
        companies = ["Self", "Tech Corp", "Oil Inc", "Marine Ltd", "Services Co"]
        job_titles = ["Officer", "Manager", "Engineer", "Analyst", "Supervisor"]
        emails = ["user@example.com", "admin@company.com", "-", "info@org.net"]
        remarks = ["Great!", "Impressed.", "Satisfied.", "Informative.", "Organized."]
        num_samples = random.randint(5, 15)
        sample_data = pd.DataFrame({
            "S/N": list(range(1, num_samples + 1)),
            "First Name": random.choices(first_names, k=num_samples),
            "Middle Name": random.choices(middle_names, k=num_samples),
            "Last Name": random.choices(last_names, k=num_samples),
            "Gender": random.choices(genders, k=num_samples),
            "DoB": [f"{random.randint(1970, 2000)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(num_samples)],
            "Company Name": random.choices(companies, k=num_samples),
            "Job Title": random.choices(job_titles, k=num_samples),
            "Contact Number": [f"080{random.randint(10000000, 99999999)}" for _ in range(num_samples)],
            "Email Address": random.choices(emails, k=num_samples),
            "NoK Name": [f"{random.choice(['Mr.', 'Mrs.', 'Ms.'])} {random.choice(first_names)} {random.choice(last_names)}" for _ in range(num_samples)],
            "NoK Number": [f"080{random.randint(10000000, 99999999)}" for _ in range(num_samples)],
            "ID Card Number": [f"AENL/A{random.randint(20,25)}/01/{random.randint(1,999):04d}" for _ in range(num_samples)],
            "Certificate Number": [f"EBS2024{random.randint(1000,9999)}" for _ in range(num_samples)],
            "Issued Date": [f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(num_samples)],
            "Expiry Date": [f"2028-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(num_samples)],
            "Course Date": random.choices(["8th March, 2024", "5th June, 2025", "15th July, 2024"], k=num_samples),
            "Batch": random.choices(["1/24", "1/25", "2/24"], k=num_samples),
            "Delegate's Remark": random.choices(remarks, k=num_samples),
            "Resolution": [""] * num_samples
        })
        st.dataframe(sample_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("Please upload a file to get started! 🚀")

# Footer
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; color: {primary_color};">Delegate Management System v2.1 | Powered by Streamlit with Enhanced UI</div>',
    unsafe_allow_html=True
)

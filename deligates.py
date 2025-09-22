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

# Define color schemes
if config == "dark":
    primary_color = "#0A84FF"  # Blue
    secondary_color = "#32D74B"  # Green
    background_color = "#0E1117"
    text_color = "#FAFAFA"
    accent_color = "#FF2B55"
else:
    primary_color = "#1F77B4"  # Blue
    secondary_color = "#2CA02C"  # Green
    background_color = "#FFFFFF"
    text_color = "#000000"
    accent_color = "#D62728"

# Custom CSS with theme support
st.markdown(f"""
<style>
    /* Global styles */
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}

    /* Main headers */
    .main-header {{
        font-size: 3rem;
        color: {primary_color};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {primary_color}, {secondary_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }}

    /* Subheaders */
    .subheader {{
        font-size: 1.8rem;
        color: {primary_color};
        border-bottom: 3px solid {primary_color};
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }}

    /* Buttons */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(135deg, {primary_color}, {secondary_color});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}

    .stDownloadButton>button {{
        width: 100%;
        background: linear-gradient(135deg, {secondary_color}, {primary_color});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stDownloadButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}

    /* Info boxes */
    .info-box {{
        background: linear-gradient(135deg, rgba(240,242,246,0.8), rgba(230,247,255,0.8));
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid {primary_color};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    /* Filter boxes */
    .filter-box {{
        background: linear-gradient(135deg, rgba(230,247,255,0.8), rgba(240,242,246,0.8));
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid {secondary_color};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    /* Metrics */
    .stMetric {{
        background-color: rgba(248,249,250,0.8);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid {primary_color};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: rgba(240,242,246,0.8);
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: 600;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {primary_color};
        color: white;
        border-bottom: 3px solid {secondary_color};
    }}

    /* Dataframes */
    .dataframe {{
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(135deg, rgba(248,249,250,0.8), rgba(230,247,255,0.8));
    }}

    /* File uploader */
    .stFileUploader {{
        border: 2px dashed {primary_color};
        border-radius: 12px;
        padding: 20px;
        background-color: rgba(248,249,250,0.8);
    }}

    /* Success messages */
    .stSuccess {{
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(212,237,218,0.8), rgba(195,230,203,0.8));
    }}

    /* Warning messages */
    .stWarning {{
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(255,243,205,0.8), rgba(255,234,167,0.8));
    }}

    /* Info messages */
    .stInfo {{
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(209,236,241,0.8), rgba(190,229,235,0.8));
    }}

    /* Expander */
    .stExpander {{
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 8px;
    }}

    /* Selectbox, Multiselect */
    .stSelectbox div[role="combobox"], .stMultiselect div[role="combobox"] {{
        background-color: {background_color};
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 4px;
    }}

    /* Adjust for dark mode */
    [data-testid="stAppViewContainer"] {{
        background-color: {background_color};
    }}
</style>
""", unsafe_allow_html=True)

# App title with emoji
st.markdown('<h1 class="main-header">👥 Delegate Management System</h1>', unsafe_allow_html=True)

# Sidebar for additional options
with st.sidebar:
    st.markdown(f'<h3 style="color: {primary_color};">⚙️ Settings</h3>', unsafe_allow_html=True)
    theme = st.selectbox("Color Theme Override", ["Auto", "Blue", "Green", "Purple"], index=0)
    if theme != "Auto":
        # Override colors based on theme
        if theme == "Blue":
            primary_color = "#1F77B4"
            secondary_color = "#AEC7E8"
        elif theme == "Green":
            primary_color = "#2CA02C"
            secondary_color = "#98DF8A"
        elif theme == "Purple":
            primary_color = "#9467BD"
            secondary_color = "#C5B0D5"
        # Note: Full CSS re-application would be needed for dynamic theme change, but for now, it's static
    st.markdown("---")
    st.markdown(f'<h3 style="color: {primary_color};">📊 Data Options</h3>', unsafe_allow_html=True)
    show_sample = st.checkbox("Show Sample Data", value=True)
    auto_ffill = st.checkbox("Auto Forward-Fill Merged Cells", value=True)
    st.markdown("---")
    st.markdown(f'<h3 style="color: {primary_color};">📈 Visualization Options</h3>', unsafe_allow_html=True)
    chart_style = st.selectbox("Chart Style", ["Default", "Minimal", "Detailed"])
    show_charts = st.checkbox("Show Advanced Charts", value=True)
    st.markdown("---")
    st.markdown(f'<h3 style="color: {primary_color};">🔗 Quick Links</h3>', unsafe_allow_html=True)
    st.button("View Documentation")
    st.button("Contact Support")
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit*")

# File uploader
st.markdown(f'<h2 class="subheader">📤 Upload Your Delegate Data</h2>', unsafe_allow_html=True)
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

    # Show preview of first few rows
    with st.expander("🔍 Raw Data Preview (First 10 Rows)", expanded=True):
        st.dataframe(temp_df.head(10), use_container_width=True)

    # Let user choose header row, default to 1 assuming first row is title
    header_options = list(range(len(temp_df.head(10))))
    header_row = st.selectbox(
        "Select Header Row (0-based index)",
        options=header_options,
        index=1,  # Default to second row (index 1)
        help="Choose which row contains the column headers. Inspect the preview to decide. Often the second row if first is title."
    )

    # Read with selected header
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=header_row, dtype=str)  # Read all as str to preserve numbers
    else:
        df = pd.read_excel(uploaded_file, header=header_row, dtype=str)  # Read all as str

    # Clean up column names
    df.columns = [str(col).strip().replace('\n', ' ').title() for col in df.columns]

    # Forward-fill merged cells if enabled
    if auto_ffill:
        ffill_cols = [col for col in df.columns if any(x in str(col).upper() for x in ['BATCH', 'COURSE DATE', 'ISSUED DATE', 'EXPIRY DATE', 'DATE'])]
        if ffill_cols:
            df[ffill_cols] = df[ffill_cols].ffill()

    # Preserve phone numbers
    for col in df.columns:
        if any(x in str(col).upper() for x in ['NUMBER', 'PHONE', 'NOK', 'CONTACT', 'TEL']):
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.replace('^0', '0', regex=False)

    # Store original columns
    st.session_state.original_columns = df.columns.tolist()

    # Create full name
    first_name_col = next((col for col in df.columns if 'FIRST NAME' in str(col).upper()), None)
    last_name_col = next((col for col in df.columns if 'LAST NAME' in str(col).upper()), None)
    if first_name_col and last_name_col:
        df['Full Name'] = df[first_name_col].fillna('') + ' ' + df[last_name_col].fillna('')

    # Calculate age
    dob_col = next((col for col in df.columns if 'DOB' in str(col).upper() or 'BIRTH' in str(col).upper()), None)
    if dob_col:
        try:
            df[dob_col] = pd.to_datetime(df[dob_col], errors='coerce')
            current_date = pd.to_datetime('today')
            df['Age'] = ((current_date - df[dob_col]).dt.days // 365.25).astype('Int64')
        except:
            st.warning("Unable to calculate ages from DoB column.")

    st.session_state.df = df.copy()

    # Dataset overview
    st.markdown('<h2 class="subheader">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Unique Values (Avg)", int(df.nunique().mean()))

    with st.expander("📝 Original Column Names"):
        st.write(st.session_state.original_columns)

    with st.expander("📊 Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)

    # Preprocessing
    st.markdown('<h2 class="subheader">⚙️ Data Preprocessing</h2>', unsafe_allow_html=True)
    prep_col1, prep_col2 = st.columns(2)

    with prep_col1:
        st.markdown("### 🧹 Cleaning Options")
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
        st.markdown("### 📝 Transformation Options")
        if st.button("Standardize Text"):
            text_cols = st.session_state.df.select_dtypes('object').columns
            for col in text_cols:
                if col != 'Full Name':
                    st.session_state.df[col] = st.session_state.df[col].str.title().str.strip()
            st.success("Standardized text")

        if st.button("Reset to Original"):
            # Reload logic similar to above
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
        with st.container():
            st.success("Data Processed!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records", len(st.session_state.df))
            with col2:
                st.metric("Memory", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        with st.expander("🔍 Processed Data"):
            st.dataframe(st.session_state.df, use_container_width=True)

    # Search and Filter
    st.markdown('<h2 class="subheader">🔍 Search & Filter</h2>', unsafe_allow_html=True)

    tabs = st.tabs(["Individual Search", "Multi-Filter", "Analysis", "Export"])

    with tabs[0]:
        st.markdown("### 👤 Individual Search")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            search_name = st.text_input("By Name")
        with s_col2:
            search_id = st.text_input("By Cert No.")

        if search_name or search_id:
            results = st.session_state.df.copy()
            if search_name:
                name_cols = [col for col in results.columns if 'NAME' in str(col).upper()]
                mask = results[name_cols].apply(lambda row: row.str.contains(search_name, case=False, na=False).any(), axis=1)
                results = results[mask]
            if search_id:
                cert_cols = [col for col in results.columns if 'CERT' in str(col).upper() or 'ID' in str(col).upper()]
                mask = results[cert_cols].apply(lambda row: row.str.contains(search_id, case=False, na=False).any(), axis=1)
                results = results[mask]
            if not results.empty:
                st.success(f"{len(results)} matches")
                st.dataframe(results, use_container_width=True)
                st.session_state.filtered_data = results
                if len(results) == 1:
                    transp = results.T.reset_index()
                    transp.columns = ['Field', 'Value']
                    st.table(transp)
                    with io.BytesIO() as buf:
                        transp.to_excel(buf, index=False)
                        buf.seek(0)
                        st.download_button("Download Record", buf, "record.xlsx")
                with io.BytesIO() as buf:
                    results.to_excel(buf, index=False)
                    buf.seek(0)
                    st.download_button("Download Matches", buf, "matches.xlsx")
            else:
                st.warning("No matches")

    with tabs[1]:
        st.markdown("### 🎚️ Multi-Filter")
        f_col1, f_col2 = st.columns(2)

        company_col = next((col for col in st.session_state.df.columns if 'COMPANY' in str(col).upper()), None)
        if company_col:
            with f_col1:
                companies = ['All'] + sorted(st.session_state.df[company_col].dropna().unique().tolist())
                sel_comp = st.multiselect("Companies", companies, default=['All'])
                if 'All' in sel_comp:
                    sel_comp = st.session_state.df[company_col].unique().tolist()

        batch_col = next((col for col in st.session_state.df.columns if 'BATCH' in str(col).upper()), None)
        if batch_col:
            with f_col1:
                batches = ['All'] + sorted(st.session_state.df[batch_col].dropna().unique().tolist())
                sel_batch = st.multiselect("Batches", batches, default=['All'])
                if 'All' in sel_batch:
                    sel_batch = st.session_state.df[batch_col].unique().tolist()

        date_col = next((col for col in st.session_state.df.columns if 'DATE' in str(col).upper()), None)
        if date_col:
            with f_col2:
                dates = ['All'] + sorted(st.session_state.df[date_col].dropna().unique().tolist())
                sel_date = st.multiselect("Dates", dates, default=['All'])
                if 'All' in sel_date:
                    sel_date = st.session_state.df[date_col].unique().tolist()

        gender_col = next((col for col in st.session_state.df.columns if 'GENDER' in str(col).upper()), None)
        if gender_col:
            with f_col2:
                genders = ['All'] + sorted(st.session_state.df[gender_col].dropna().unique().tolist())
                sel_gender = st.multiselect("Genders", genders, default=['All'])
                if 'All' in sel_gender:
                    sel_gender = st.session_state.df[gender_col].unique().tolist()

        if st.button("Filter"):
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
            st.success(f"{len(filtered)} records")

        if st.session_state.filtered_data is not None:
            st.dataframe(st.session_state.filtered_data, use_container_width=True)

    with tabs[2]:
        st.markdown("### 📊 Analysis & Visualizations")
        if gender_col:
            gender_cnt = st.session_state.df[gender_col].value_counts()
            fig_gender = px.pie(values=gender_cnt.values, names=gender_cnt.index, title="Gender Distribution")
            st.plotly_chart(fig_gender)

        if 'Age' in st.session_state.df.columns:
            age_data = st.session_state.df['Age'].dropna().astype(float)
            if not age_data.empty:
                bins = [0, 20, 30, 40, 50, 60, np.inf]
                labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
                age_groups = pd.cut(age_data, bins, labels=labels)
                age_cnt = age_groups.value_counts().sort_index()
                fig_age = px.bar(x=age_cnt.index, y=age_cnt.values, title="Age Distribution")
                st.plotly_chart(fig_age)

        if batch_col:
            batch_cnt = st.session_state.df[batch_col].value_counts()
            fig_batch = px.bar(x=batch_cnt.index, y=batch_cnt.values, title="Batch Distribution")
            st.plotly_chart(fig_batch)

        if company_col:
            comp_cnt = st.session_state.df[company_col].value_counts().head(10)
            fig_comp = px.bar(y=comp_cnt.index, x=comp_cnt.values, orientation='h', title="Top Companies")
            st.plotly_chart(fig_comp)

    with tabs[3]:
        st.markdown("### 💾 Export Options")
        if st.session_state.filtered_data is not None:
            st.metric("Records to Export", len(st.session_state.filtered_data))
            with io.BytesIO() as buf:
                st.session_state.filtered_data.to_excel(buf, index=False)
                buf.seek(0)
                st.download_button("Filtered Excel", buf, "filtered.xlsx")
            csv = st.session_state.filtered_data.to_csv(index=False)
            st.download_button("Filtered CSV", csv, "filtered.csv")

        st.markdown("### Full Dataset")
        with io.BytesIO() as buf:
            st.session_state.df.to_excel(buf, index=False)
            buf.seek(0)
            st.download_button("Full Excel", buf, "full.xlsx")
        csv_full = st.session_state.df.to_csv(index=False)
        st.download_button("Full CSV", csv_full, "full.csv")

else:
    if show_sample:
        st.markdown("### 📋 Sample Data")
        # Randomized sample data
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
        st.dataframe(sample_data)

    st.info("Upload a file to begin")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center;">Delegate Management System v2.0 | Powered by Streamlit</div>', unsafe_allow_html=True)

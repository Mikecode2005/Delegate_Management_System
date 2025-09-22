import streamlit as st
import pandas as pd
import io
import plotly.express as px
from datetime import datetime
import numpy as np
import random
import uuid

# Set page configuration for modern dashboard
st.set_page_config(
    page_title="Delegate Management Dashboard",
    layout="wide",
    page_icon="👥",
    initial_sidebar_state="expanded"
)

# Detect theme
config = st._config.get_option("theme.base")
if config == "dark":
    primary_color = "#0A84FF"
    secondary_color = "#32D74B"
    tertiary_color = "#FF9500"
    background_color = "#0E1117"
    text_color = "#FAFAFA"
    card_bg = "#1E1F25"
    accent_color = "#FF2B55"
else:
    primary_color = "#1F77B4"
    secondary_color = "#2CA02C"
    tertiary_color = "#FF7F0E"
    background_color = "#FFFFFF"
    text_color = "#000000"
    card_bg = "#F8F9FA"
    accent_color = "#D62728"

# Modern, compact, and vibrant CSS
st.markdown(f"""
<style>
    /* Global styles */
    .stApp {{
        background-color: {background_color};
        color: {text_color};
        font-family: 'Inter', sans-serif;
    }}

    /* Dashboard container */
    .dashboard-container {{
        max-width: 1600px;
        margin: 0 auto;
        padding: 1rem;
    }}

    /* Header */
    .main-header {{
        font-size: 2.5rem;
        color: {primary_color};
        text-align: center;
        font-weight: 700;
        background: linear-gradient(135deg, {primary_color}, {secondary_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem;
        animation: slideIn 0.5s ease-in-out;
    }}

    @keyframes slideIn {{
        from {{ transform: translateY(-20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}

    /* Subheader */
    .subheader {{
        font-size: 1.5rem;
        color: {primary_color};
        font-weight: 600;
        margin: 0.5rem 0;
    }}

    /* Cards */
    .card {{
        background-color: {card_bg};
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {primary_color}, {secondary_color});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        background: linear-gradient(135deg, {secondary_color}, {primary_color});
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}

    .stDownloadButton>button {{
        background: linear-gradient(135deg, {tertiary_color}, {accent_color});
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stDownloadButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}

    /* Metrics */
    .stMetric {{
        background-color: {card_bg};
        border-radius: 8px;
        padding: 0.75rem;
        border-left: 4px solid {primary_color};
        margin-bottom: 0.5rem;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {card_bg};
        border-radius: 8px;
        padding: 0.5rem;
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: linear-gradient(135deg, {primary_color}10, {secondary_color}10);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: {text_color};
        transition: all 0.3s ease;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {primary_color}, {secondary_color});
        color: white;
    }}

    /* Dataframe */
    .dataframe {{
        border-radius: 8px;
        overflow: auto;
        max-height: 400px;
    }}

    .dataframe tr:hover {{
        background-color: {primary_color}10;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {primary_color}10, {secondary_color}10);
        padding: 1rem;
    }}

    /* File uploader */
    .stFileUploader {{
        border: 2px dashed {primary_color};
        border-radius: 12px;
        padding: 1rem;
        background-color: {card_bg};
        text-align: center;
    }}

    /* Messages */
    .stSuccess {{
        background: linear-gradient(135deg, {secondary_color}20, {secondary_color}40);
        border-radius: 8px;
        color: {text_color};
    }}

    .stWarning {{
        background: linear-gradient(135deg, {accent_color}20, {accent_color}40);
        border-radius: 8px;
        color: {text_color};
    }}

    .stInfo {{
        background: linear-gradient(135deg, {primary_color}20, {primary_color}40);
        border-radius: 8px;
        color: {text_color};
    }}

    /* Expander */
    .stExpander {{
        border: 1px solid {primary_color}20;
        border-radius: 8px;
    }}

    .stExpander summary {{
        font-weight: 600;
        color: {primary_color};
    }}

    /* Grid layout */
    .grid-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .main-header {{
            font-size: 2rem;
        }}
        .subheader {{
            font-size: 1.2rem;
        }}
        .grid-container {{
            grid-template-columns: 1fr;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'original_columns' not in st.session_state:
    st.session_state.original_columns = None

# Dashboard container
st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Delegate Management Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f'<h3 style="color: {primary_color};">⚙️ Control Panel</h3>', unsafe_allow_html=True)
    theme = st.selectbox("Theme", ["Auto", "Blue", "Green", "Purple", "Vibrant"], index=0)
    if theme != "Auto":
        if theme == "Blue":
            primary_color, secondary_color, tertiary_color = "#1F77B4", "#AEC7E8", "#17BECF"
        elif theme == "Green":
            primary_color, secondary_color, tertiary_color = "#2CA02C", "#98DF8A", "#2E9945"
        elif theme == "Purple":
            primary_color, secondary_color, tertiary_color = "#9467BD", "#C5B0D5", "#756BB1"
        elif theme == "Vibrant":
            primary_color, secondary_color, tertiary_color = "#FF9500", "#FF2D55", "#5856D6"
    show_sample = st.checkbox("Show Sample Data", value=True)
    auto_ffill = st.checkbox("Auto Forward-Fill", value=True)
    chart_style = st.selectbox("Chart Style", ["Default", "Minimal", "Detailed", "Colorful"], index=3)
    st.markdown("---")
    st.button("Documentation")
    st.button("Support")
    st.markdown(f'<p style="text-align: center; color: {text_color};">Built with Streamlit</p>', unsafe_allow_html=True)

# Main content
if st.session_state.df is None and not show_sample:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="subheader">📤 Upload Data</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])
    if uploaded_file:
        # Read and process file
        if uploaded_file.name.endswith(".csv"):
            temp_df = pd.read_csv(uploaded_file, header=None)
        else:
            temp_df = pd.read_excel(uploaded_file, header=None)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.expander("🔍 Preview Raw Data", expanded=True):
            st.dataframe(temp_df.head(10), use_container_width=True)

        header_row = st.selectbox("Header Row (0-based)", list(range(len(temp_df.head(10)))), index=1)

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=header_row, dtype=str)
        else:
            df = pd.read_excel(uploaded_file, header=header_row, dtype=str)

        df.columns = [str(col).strip().replace('\n', ' ').title() for col in df.columns]

        if auto_ffill:
            ffill_cols = [col for col in df.columns if any(x in str(col).upper() for x in ['BATCH', 'COURSE DATE', 'ISSUED DATE', 'EXPIRY DATE', 'DATE'])]
            if ffill_cols:
                df[ffill_cols] = df[ffill_cols].ffill()

        for col in df.columns:
            if any(x in str(col).upper() for x in ['NUMBER', 'PHONE', 'NOK', 'CONTACT', 'TEL']):
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
            except:
                st.warning("Unable to calculate ages.")

        st.session_state.df = df.copy()
        st.session_state.original_columns = df.columns.tolist()
        st.session_state.processed = True
        st.success("Data uploaded successfully! 🎉")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Upload a file to start managing delegates.")
        if show_sample:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="subheader">📋 Sample Data</h2>', unsafe_allow_html=True)
            sample_data = pd.DataFrame({
                "S/N": list(range(1, 11)),
                "First Name": random.choices(["John", "Jane", "Alex", "Emily", "Michael"], k=10),
                "Last Name": random.choices(["Doe", "Smith", "Johnson", "Brown", "Davis"], k=10),
                "Gender": random.choices(["Male", "Female"], k=10),
                "DoB": [f"199{random.randint(0,9)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(10)],
                "Company Name": random.choices(["Tech Corp", "Oil Inc", "Marine Ltd"], k=10),
                "Contact Number": [f"080{random.randint(10000000, 99999999)}" for _ in range(10)],
                "Certificate Number": [f"EBS2024{random.randint(1000,9999)}" for _ in range(10)],
                "Course Date": random.choices(["8th March, 2024", "5th June, 2025"], k=10),
                "Batch": random.choices(["1/24", "2/24"], k=10)
            })
            st.dataframe(sample_data, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Modern dashboard layout with grid
    st.markdown('<div class="grid-container">', unsafe_allow_html=True)

    # Overview Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">📊 Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", len(st.session_state.df), delta_color="normal")
    with col2:
        st.metric("Columns", len(st.session_state.df.columns))
    with col3:
        st.metric("Missing", st.session_state.df.isnull().sum().sum(), delta_color="inverse")
    with col4:
        st.metric("Unique (Avg)", int(st.session_state.df.nunique().mean()))
    with st.expander("Columns"):
        st.write(st.session_state.original_columns)
    st.markdown('</div>', unsafe_allow_html=True)

    # Data Preview Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">🔍 Data Preview</h2>', unsafe_allow_html=True)
    st.dataframe(st.session_state.df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocessing Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">⚙️ Preprocessing</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Remove Duplicates"):
            old_len = len(st.session_state.df)
            st.session_state.df.drop_duplicates(inplace=True)
            st.success(f"Removed {old_len - len(st.session_state.df)} duplicates")
        fill_opt = st.radio("Missing Values", ["Keep", "Fill", "Drop"], key="fill_opt")
        if fill_opt == "Fill":
            fill_val = st.text_input("Fill Value", "N/A")
            if st.button("Apply Fill"):
                st.session_state.df.fillna(fill_val, inplace=True)
                st.success("Filled missing values")
        elif fill_opt == "Drop":
            if st.button("Apply Drop"):
                old_len = len(st.session_state.df)
                st.session_state.df.dropna(inplace=True)
                st.success(f"Dropped {old_len - len(st.session_state.df)} rows")
    with col2:
        col_remove = st.selectbox("Remove Column", ['None'] + list(st.session_state.df.columns))
        if col_remove != 'None' and st.button("Remove"):
            st.session_state.df.drop(columns=[col_remove], inplace=True)
            if st.session_state.filtered_data is not None:
                st.session_state.filtered_data.drop(columns=[col_remove], errors='ignore', inplace=True)
            st.success(f"Removed {col_remove}")
        if st.button("Standardize Text"):
            text_cols = st.session_state.df.select_dtypes('object').columns
            for col in text_cols:
                if col != 'Full Name':
                    st.session_state.df[col] = st.session_state.df[col].str.title().str.strip()
            st.success("Standardized text")
        if st.button("Reset Data"):
            if uploaded_file:
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
    st.markdown('</div>', unsafe_allow_html=True)

    # Search & Filter Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">🔍 Search & Filter</h2>', unsafe_allow_html=True)
    tabs = st.tabs(["Individual", "Multi-Filter", "Analysis", "Export"])

    with tabs[0]:
        st.markdown(f'<h3 style="color: {tertiary_color};">👤 Individual Search</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            search_name = st.text_input("Name")
        with col2:
            search_id = st.text_input("Cert No.")
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
        st.markdown(f'<h3 style="color: {tertiary_color};">🎚️ Multi-Filter</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        company_col = next((col for col in st.session_state.df.columns if 'COMPANY' in str(col).upper()), None)
        if company_col:
            with col1:
                companies = ['All'] + sorted(st.session_state.df[company_col].dropna().unique().tolist())
                sel_comp = st.multiselect("Companies 🏢", companies, default=['All'])
                if 'All' in sel_comp:
                    sel_comp = st.session_state.df[company_col].unique().tolist()

        batch_col = next((col for col in st.session_state.df.columns if 'BATCH' in str(col).upper()), None)
        if batch_col:
            with col1:
                batches = ['All'] + sorted(st.session_state.df[batch_col].dropna().unique().tolist())
                sel_batch = st.multiselect("Batches 🗂️", batches, default=['All'])
                if 'All' in sel_batch:
                    sel_batch = st.session_state.df[batch_col].unique().tolist()

        date_col = next((col for col in st.session_state.df.columns if 'DATE' in str(col).upper()), None)
        if date_col:
            with col2:
                dates = ['All'] + sorted(st.session_state.df[date_col].dropna().unique().tolist())
                sel_date = st.multiselect("Dates 📅", dates, default=['All'])
                if 'All' in sel_date:
                    sel_date = st.session_state.df[date_col].unique().tolist()

        gender_col = next((col for col in st.session_state.df.columns if 'GENDER' in str(col).upper()), None)
        if gender_col:
            with col2:
                genders = ['All'] + sorted(st.session_state.df[gender_col].dropna().unique().tolist())
                sel_gender = st.multiselect("Genders 👥", genders, default=['All'])
                if 'All' in sel_gender:
                    sel_gender = st.session_state.df[gender_col].unique().tolist()

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
            st.success(f"{len(filtered)} records")

        if st.session_state.filtered_data is not None:
            st.dataframe(st.session_state.filtered_data, use_container_width=True)

    with tabs[2]:
        st.markdown(f'<h3 style="color: {tertiary_color};">📊 Analysis</h3>', unsafe_allow_html=True)
        color_seq = px.colors.sequential.Rainbow if chart_style == "Colorful" else px.colors.qualitative.Set3
        col1, col2 = st.columns(2)
        if gender_col:
            with col1:
                gender_cnt = st.session_state.df[gender_col].value_counts()
                fig_gender = px.pie(values=gender_cnt.values, names=gender_cnt.index, title="Gender", color_discrete_sequence=color_seq)
                fig_gender.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_gender, use_container_width=True)
        if 'Age' in st.session_state.df.columns:
            with col2:
                age_data = st.session_state.df['Age'].dropna().astype(float)
                if not age_data.empty:
                    bins = [0, 20, 30, 40, 50, 60, np.inf]
                    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
                    age_groups = pd.cut(age_data, bins, labels=labels)
                    age_cnt = age_groups.value_counts().sort_index()
                    fig_age = px.bar(x=age_cnt.index, y=age_cnt.values, title="Age Groups", color=age_cnt.index, color_discrete_sequence=color_seq)
                    st.plotly_chart(fig_age, use_container_width=True)
        col3, col4 = st.columns(2)
        if batch_col:
            with col3:
                batch_cnt = st.session_state.df[batch_col].value_counts()
                fig_batch = px.bar(x=batch_cnt.index, y=batch_cnt.values, title="Batches", color=batch_cnt.index, color_discrete_sequence=color_seq)
                st.plotly_chart(fig_batch, use_container_width=True)
        if company_col:
            with col4:
                comp_cnt = st.session_state.df[company_col].value_counts().head(10)
                fig_comp = px.bar(y=comp_cnt.index, x=comp_cnt.values, orientation='h', title="Top Companies", color=comp_cnt.index, color_discrete_sequence=color_seq)
                st.plotly_chart(fig_comp, use_container_width=True)

    with tabs[3]:
        st.markdown(f'<h3 style="color: {tertiary_color};">💾 Export</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        if st.session_state.filtered_data is not None:
            with col1:
                st.metric("Records to Export", len(st.session_state.filtered_data))
                with io.BytesIO() as buf:
                    st.session_state.filtered_data.to_excel(buf, index=False)
                    buf.seek(0)
                    st.download_button("Filtered Excel", buf, "filtered.xlsx")
                csv = st.session_state.filtered_data.to_csv(index=False)
                st.download_button("Filtered CSV", csv, "filtered.csv")
        with col2:
            st.markdown("Full Dataset")
            with io.BytesIO() as buf:
                st.session_state.df.to_excel(buf, index=False)
                buf.seek(0)
                st.download_button("Full Excel", buf, "full.xlsx")
            csv_full = st.session_state.df.to_csv(index=False)
            st.download_button("Full CSV", csv_full, "full.csv")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f'<div style="text-align: center; color: {primary_color}; padding: 1rem;">Delegate Management Dashboard v3.0 | Powered by Streamlit</div>', unsafe_allow_html=True)


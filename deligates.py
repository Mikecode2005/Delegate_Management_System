import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import plotly.express as px
import plotly.graph_objects as go
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

# Custom CSS with better dark mode compatibility
# Use relative colors and avoid fixed light backgrounds
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        padding: 10px;
    }

    /* Subheaders */
    .subheader {
        font-size: 1.8rem;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .stDownloadButton>button {
        width: 100%;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Info boxes */
    .info-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Filter boxes */
    .filter-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Metrics */
    .stMetric {
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: 600;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }

    /* Dataframes */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Sidebar */
    .css-1d391kg {
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed;
        border-radius: 12px;
        padding: 20px;
    }

    /* Success messages */
    .stSuccess {
        border-radius: 8px;
    }

    /* Warning messages */
    .stWarning {
        border-radius: 8px;
    }

    /* Info messages */
    .stInfo {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# App title with emoji
st.markdown('<h1 class="main-header">👥 Delegate Management System</h1>', unsafe_allow_html=True)

# Sidebar for additional options
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme = st.selectbox("Color Theme", ["Blue", "Green", "Purple"])
    st.markdown("---")
    st.markdown("### 📊 Data Options")
    show_sample = st.checkbox("Show Sample Data", value=True)
    st.markdown("---")
    st.markdown("### 📈 Visualization Options")
    chart_style = st.selectbox("Chart Style", ["Default", "Minimal", "Detailed"])
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.button("View Documentation")
    st.button("Contact Support")
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit*")

# File uploader with improved styling
st.markdown("### 📤 Upload Your Delegate Data")
uploaded_file = st.file_uploader("", type=["xlsx", "csv"], help="Upload Excel or CSV file with delegate information")

# Initialize session state for data
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
    st.markdown("### 🔍 Raw Data Preview (First 5 Rows)")
    st.dataframe(temp_df.head(5), use_container_width=True)

    # Let user choose header row
    header_options = list(range(len(temp_df.head(5))))
    header_row = st.selectbox(
        "Select Header Row (0-based index)",
        options=header_options,
        index=1 if "COMPRESSED AIR EMERGENCY BREATHING SYSTEM TRAINING" in ' '.join(str(x) for x in temp_df.iloc[0]) else 0,
        help="Choose which row contains the column headers"
    )

    # Read with selected header
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=header_row, dtype=str)  # Read all as str to preserve numbers
    else:
        df = pd.read_excel(uploaded_file, header=header_row, dtype=str)  # Read all as str

    # Clean up any leading/trailing whitespace in column names
    df.columns = df.columns.str.strip()

    # Forward-fill merged cells for specific columns
    ffill_cols = []
    for col in df.columns:
        if any(x in col.upper() for x in ['BATCH', 'COURSE DATE', 'ISSUED DATE', 'EXPIRY DATE']):
            ffill_cols.append(col)
    if ffill_cols:
        df[ffill_cols] = df[ffill_cols].ffill()

    # Convert phone number columns to string and preserve full values
    for col in df.columns:
        if any(x in col.upper() for x in ['NUMBER', 'PHONE', 'NOK', 'CONTACT']):
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    # Store original column names
    st.session_state.original_columns = df.columns.tolist()

    # Create a full name column if first and last name columns exist
    first_name_col = None
    last_name_col = None
    for col in df.columns:
        if 'FIRST NAME' in str(col).upper():
            first_name_col = col
        if 'LAST NAME' in str(col).upper():
            last_name_col = col

    if first_name_col and last_name_col:
        df['Full Name'] = df[first_name_col].astype(str).fillna('') + ' ' + df[last_name_col].astype(str).fillna('')

    # Calculate age from DoB if available
    dob_col = None
    for col in df.columns:
        if 'DOB' in str(col).upper() or 'BIRTH' in str(col).upper():
            dob_col = col
            break

    if dob_col:
        try:
            # Convert to datetime, coercing errors
            df[dob_col] = pd.to_datetime(df[dob_col], errors='coerce')
            current_date = pd.to_datetime('today')
            df['Age'] = ((current_date - df[dob_col]).dt.days // 365.25).astype('Int64')
        except Exception as e:
            st.warning(f"Could not calculate ages: {e}")

    # Store in session state
    st.session_state.df = df.copy()

    # Display raw data info with metrics
    st.markdown('<h2 class="subheader">📋 Dataset Overview</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df), help="Number of records in the dataset")
    with col2:
        st.metric("Columns", len(df.columns), help="Number of data columns")
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum(), help="Total missing values in the dataset")
    with col4:
        st.metric("Data Types", len(df.dtypes.unique()), help="Unique data types in the dataset")

    # Show original column names
    with st.expander("📝 Original Column Names", expanded=False):
        st.write("Original columns detected in your file:")
        for col in st.session_state.original_columns:
            st.write(f"- {col}")

    # Show data preview with expander
    with st.expander("📊 Preview Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)  # Show head for preview, full in processed view

    # Data preprocessing options
    st.markdown('<h2 class="subheader">⚙️ Data Preprocessing</h2>', unsafe_allow_html=True)

    preprocessing_col1, preprocessing_col2 = st.columns(2)

    with preprocessing_col1:
        st.markdown("### 🧹 Data Cleaning")
        if st.button("Remove Duplicates", help="Remove duplicate records based on all columns"):
            initial_count = len(st.session_state.df)
            st.session_state.df = st.session_state.df.drop_duplicates()
            removed = initial_count - len(st.session_state.df)
            st.success(f"Removed {removed} duplicates")

        fill_option = st.radio("Handle Missing Values",
                               ["Keep as is", "Fill with value", "Remove rows"],
                               help="Choose how to handle missing values in the dataset")

        if fill_option == "Fill with value":
            fill_value = st.text_input("Value to fill missing data", "N/A")
            if st.button("Apply Fill"):
                st.session_state.df = st.session_state.df.fillna(fill_value)
                st.success("Filled missing values")
        elif fill_option == "Remove rows":
            if st.button("Remove Rows"):
                initial_count = len(st.session_state.df)
                st.session_state.df = st.session_state.df.dropna()
                removed = initial_count - len(st.session_state.df)
                st.success(f"Removed {removed} rows with missing values")

        # New feature: Remove Column
        st.markdown("#### 🗑️ Remove Column")
        if st.session_state.df is not None:
            column_to_remove = st.selectbox(
                "Select Column to Remove",
                options=['None'] + list(st.session_state.df.columns),
                help="Select a column to remove from the dataset"
            )
            if column_to_remove != 'None' and st.button("Remove Selected Column", help="Remove the selected column"):
                if column_to_remove in st.session_state.df.columns:
                    st.session_state.df = st.session_state.df.drop(columns=[column_to_remove])
                    st.success(f"Removed column: {column_to_remove}")
                    # Update filtered data if it exists
                    if st.session_state.filtered_data is not None:
                        if column_to_remove in st.session_state.filtered_data.columns:
                            st.session_state.filtered_data = st.session_state.filtered_data.drop(columns=[column_to_remove])
                else:
                    st.error(f"Column {column_to_remove} not found in the dataset.")
        else:
            st.info("Please upload a dataset to remove columns.")

    with preprocessing_col2:
        st.markdown("### 📝 Data Transformation")
        if st.button("Standardize Text Columns", help="Convert text columns to proper case and remove extra spaces"):
            text_cols = st.session_state.df.select_dtypes(include=['object']).columns
            for col in text_cols:
                if col != 'Full Name':  # Skip computed column
                    st.session_state.df.loc[:, col] = st.session_state.df[col].astype(str).str.title().str.strip()
            st.success("Standardized text columns")

        if st.button("Reset Changes", help="Revert to original data"):
            # Reload from uploaded file to reset
            if uploaded_file.name.endswith(".csv"):
                reset_df = pd.read_csv(uploaded_file, header=header_row, dtype=str)
            else:
                reset_df = pd.read_excel(uploaded_file, header=header_row, dtype=str)
            reset_df.columns = reset_df.columns.str.strip()
            # Re-apply ffill
            reset_df[ffill_cols] = reset_df[ffill_cols].ffill()
            # Re-apply phone fixes
            for col in reset_df.columns:
                if any(x in col.upper() for x in ['NUMBER', 'PHONE', 'NOK', 'CONTACT']):
                    reset_df[col] = reset_df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            st.session_state.df = reset_df.copy()
            st.success("Reset to original data")

    st.session_state.processed = True

    if st.session_state.processed:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.success("✅ Data Processing Complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processed Records", len(st.session_state.df))
        with col2:
            st.metric("Memory Usage", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        st.markdown("</div>", unsafe_allow_html=True)

        # Display processed data with expander
        with st.expander("🔍 View Processed Data", expanded=False):
            st.dataframe(st.session_state.df, use_container_width=True)

    # Search and filter functionality
    st.markdown('<h2 class="subheader">🔍 Advanced Search & Filter</h2>', unsafe_allow_html=True)

    search_tab1, search_tab2, search_tab3, search_tab4 = st.tabs(
        ["🔎 Individual Search", "🎚️ Multi-Filter", "📊 Analysis", "💾 Export Data"])

    with search_tab1:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown("### 👤 Search for Individual Delegates")

        search_col1, search_col2 = st.columns(2)
        with search_col1:
            search_name = st.text_input("Search by Name", help="Search for delegates by name")
        with search_col2:
            search_id = st.text_input("Search by Certificate No.", help="Search for delegates by certificate number")

        if search_name or search_id:
            # Use the preprocessed data for search
            results = st.session_state.df.copy()
            if search_name:
                # Search in name columns
                name_cols = [col for col in results.columns if
                             any(x in str(col).upper() for x in ['NAME', 'FIRST', 'LAST', 'FULL'])]
                mask = pd.Series([False] * len(results))
                for col in name_cols:
                    mask = mask | results[col].astype(str).str.contains(search_name, case=False, na=False)
                results = results[mask]

            if search_id:
                # Search in certificate columns
                cert_cols = [col for col in results.columns if
                             any(x in str(col).upper() for x in ['CERTIFICATE', 'ID CARD', 'S/N'])]
                mask = pd.Series([False] * len(results))
                for col in cert_cols:
                    mask = mask | results[col].astype(str).str.contains(search_id, case=False, na=False)
                results = results[mask]

            if not results.empty:
                st.success(f"Found {len(results)} matching records:")
                st.dataframe(results, use_container_width=True)

                # Store filtered data for download
                st.session_state.filtered_data = results

                # Download individual record as a formatted table
                if len(results) == 1:
                    # Transpose the single record for better readability
                    transposed_data = results.T.reset_index()
                    transposed_data.columns = ['Field', 'Value']

                    st.markdown("#### 👤 Individual Delegate Record")
                    st.table(transposed_data)

                    # Download as formatted individual record
                    towrite = io.BytesIO()
                    transposed_data.to_excel(towrite, index=False, sheet_name="Delegate Record")
                    towrite.seek(0)
                    st.download_button(
                        label="📥 Download Individual Record",
                        data=towrite,
                        file_name="delegate_record.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # Download all searched records
                towrite = io.BytesIO()
                results.to_excel(towrite, index=False, sheet_name="Delegates")
                towrite.seek(0)
                st.download_button(
                    label="📥 Download All Matching Records",
                    data=towrite,
                    file_name="delegate_records.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No matching delegate found!")
                st.session_state.filtered_data = None
        else:
            st.info("Enter a name or certificate number to search for delegates")
        st.markdown("</div>", unsafe_allow_html=True)

    with search_tab2:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown("### 🎚️ Filter Delegates by Multiple Criteria")

        col1, col2 = st.columns(2)

        with col1:
            # Filter by Company
            company_col = None
            for col in st.session_state.df.columns:
                if any(x in str(col).upper() for x in ['COMPANY', 'ORGANIZATION']):
                    company_col = col
                    break

            if company_col:
                st.markdown("#### 🏢 Company")
                company_options = ['All'] + sorted(list(st.session_state.df[company_col].dropna().unique()))
                selected_company = st.multiselect(
                    "Select Companies",
                    options=company_options,
                    default=['All'],
                    help="Filter by company"
                )
                if 'All' in selected_company and len(selected_company) == 1:
                    selected_company = list(st.session_state.df[company_col].dropna().unique())

            # Filter by Batch
            batch_col = None
            for col in st.session_state.df.columns:
                if any(x in str(col).upper() for x in ['BATCH']):
                    batch_col = col
                    break

            if batch_col:
                st.markdown("#### 🗂️ Batch")
                batch_options = ['All'] + sorted(list(st.session_state.df[batch_col].dropna().unique()))
                selected_batch = st.multiselect(
                    "Select Batches",
                    options=batch_options,
                    default=['All'],
                    help="Filter by batch"
                )
                if 'All' in selected_batch and len(selected_batch) == 1:
                    selected_batch = list(st.session_state.df[batch_col].dropna().unique())

        with col2:
            # Filter by Course Date
            course_date_col = None
            for col in st.session_state.df.columns:
                if any(x in str(col).upper() for x in ['COURSE DATE', 'DATE']):
                    course_date_col = col
                    break

            if course_date_col:
                st.markdown("#### 📅 Course Date")
                date_options = ['All'] + sorted(list(st.session_state.df[course_date_col].dropna().unique()))
                selected_date = st.multiselect(
                    "Select Course Dates",
                    options=date_options,
                    default=['All'],
                    help="Filter by course date"
                )
                if 'All' in selected_date and len(selected_date) == 1:
                    selected_date = list(st.session_state.df[course_date_col].dropna().unique())

            # Filter by Gender
            gender_col = None
            for col in st.session_state.df.columns:
                if any(x in str(col).upper() for x in ['GENDER']):
                    gender_col = col
                    break

            if gender_col:
                st.markdown("#### 👥 Gender")
                gender_options = ['All'] + sorted(list(st.session_state.df[gender_col].dropna().unique()))
                selected_gender = st.multiselect(
                    "Select Genders",
                    options=gender_options,
                    default=['All'],
                    help="Filter by gender"
                )
                if 'All' in selected_gender and len(selected_gender) == 1:
                    selected_gender = list(st.session_state.df[gender_col].dropna().unique())

        # Apply filters button
        if st.button("Apply Filters", key="apply_filters"):
            # Use the preprocessed data for filtering
            filtered_df = st.session_state.df.copy()

            # Company filter
            if company_col and 'selected_company' in locals() and selected_company:
                filtered_df = filtered_df[filtered_df[company_col].isin(selected_company) | filtered_df[company_col].isna()]

            # Batch filter
            if batch_col and 'selected_batch' in locals() and selected_batch:
                filtered_df = filtered_df[filtered_df[batch_col].isin(selected_batch) | filtered_df[batch_col].isna()]

            # Date filter
            if course_date_col and 'selected_date' in locals() and selected_date:
                filtered_df = filtered_df[filtered_df[course_date_col].isin(selected_date) | filtered_df[course_date_col].isna()]

            # Gender filter
            if gender_col and 'selected_gender' in locals() and selected_gender:
                filtered_df = filtered_df[filtered_df[gender_col].isin(selected_gender) | filtered_df[gender_col].isna()]

            st.session_state.filtered_data = filtered_df
            st.success(f"Filter applied! Found {len(filtered_df)} records.")

        # Show filtered results
        if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
            st.markdown("#### 📋 Filtered Results")
            st.dataframe(st.session_state.filtered_data, use_container_width=True)
        elif st.session_state.filtered_data is not None:
            st.warning("No records match the applied filters.")
        st.markdown("</div>", unsafe_allow_html=True)

    with search_tab3:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Analysis")

        # Gender analysis
        gender_col = None
        for col in st.session_state.df.columns:
            if any(x in str(col).upper() for x in ['GENDER']):
                gender_col = col
                break

        if gender_col:
            st.markdown("#### 👥 Gender Distribution")
            gender_counts = st.session_state.df[gender_col].value_counts()
            if not gender_counts.empty:
                fig = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Gender Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No gender data available for visualization.")

        # Age analysis
        age_col = 'Age' if 'Age' in st.session_state.df.columns else None

        if age_col:
            st.markdown("#### 📊 Age Distribution")
            age_data = st.session_state.df[age_col].dropna().astype(float)  # Ensure numeric
            if not age_data.empty and len(age_data) > 0:
                # Create age groups
                bins = [0, 20, 30, 40, 50, 60, np.inf]
                labels = ['Under 20', '20-29', '30-39', '40-49', '50-59', '60+']
                age_groups = pd.cut(age_data, bins=bins, labels=labels, right=False, include_lowest=True)
                age_group_counts = age_groups.value_counts(sort=False).reset_index()
                age_group_counts.columns = ['Age Group', 'Count']

                fig = px.bar(
                    age_group_counts,
                    x='Age Group',
                    y='Count',
                    title='Age Group Distribution',
                    color='Age Group',
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid age data available for visualization.")

        # Batch analysis
        batch_col = None
        for col in st.session_state.df.columns:
            if any(x in str(col).upper() for x in ['BATCH']):
                batch_col = col
                break

        if batch_col:
            st.markdown("#### 🗂️ Batch Distribution")
            batch_counts = st.session_state.df[batch_col].value_counts().reset_index()
            batch_counts.columns = ['Batch', 'Count']
            if not batch_counts.empty:
                fig = px.bar(
                    batch_counts,
                    x='Batch',
                    y='Count',
                    title='Delegates by Batch',
                    color='Batch',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No batch data available for visualization.")

        # Company analysis if applicable
        company_col = None
        for col in st.session_state.df.columns:
            if any(x in str(col).upper() for x in ['COMPANY']):
                company_col = col
                break

        if company_col:
            st.markdown("#### 🏢 Company Distribution (Top 10)")
            company_counts = st.session_state.df[company_col].value_counts().head(10).reset_index()
            company_counts.columns = ['Company', 'Count']
            if not company_counts.empty:
                fig = px.bar(
                    company_counts,
                    x='Count',
                    y='Company',
                    orientation='h',
                    title='Top 10 Companies by Delegate Count',
                    color='Company',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No company data available for visualization.")
        st.markdown("</div>", unsafe_allow_html=True)

    with search_tab4:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown("### 💾 Export Data")

        if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
            st.success(f"Data ready for export: {len(st.session_state.filtered_data)} records")

            # Show summary of filtered data
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records to Export", len(st.session_state.filtered_data))

            st.dataframe(st.session_state.filtered_data.head(), use_container_width=True)

            export_col1, export_col2 = st.columns(2)

            with export_col1:
                # Excel export
                towrite = io.BytesIO()
                st.session_state.filtered_data.to_excel(towrite, index=False, sheet_name="Filtered_Data")
                towrite.seek(0)
                st.download_button(
                    label="📊 Download Filtered Data (Excel)",
                    data=towrite,
                    file_name="filtered_delegates.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download the filtered data as an Excel file"
                )

            with export_col2:
                # CSV export
                csv_data = st.session_state.filtered_data.to_csv(index=False)
                st.download_button(
                    label="📝 Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name="filtered_delegates.csv",
                    mime="text/csv",
                    help="Download the filtered data as a CSV file"
                )
        else:
            st.info("No filtered data to export. Please use the search/filter tabs first.")

        st.markdown("---")
        st.markdown("### 📦 Export Full Dataset")
        st.info("Export the complete processed dataset")

        full_export_col1, full_export_col2 = st.columns(2)

        with full_export_col1:
            # Excel export
            towrite = io.BytesIO()
            st.session_state.df.to_excel(towrite, index=False, sheet_name="Cleaned_Data")
            towrite.seek(0)
            st.download_button(
                label="📊 Download Full Dataset (Excel)",
                data=towrite,
                file_name="cleaned_delegates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with full_export_col2:
            # CSV export
            csv_data = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="📝 Download Full Dataset (CSV)",
                data=csv_data,
                file_name="cleaned_delegates.csv",
                mime="text/csv"
            )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Show randomized sample data if no file uploaded
    if show_sample:
        st.markdown("### 📋 Sample Data Format")
        
        # Generate randomized sample data
        first_names = ["EMMANUEL", "EMEM", "DENNIS", "ALICE", "BOB", "CHARLIE", "DAVID", "EVA", "FRANK", "GRACE"]
        middle_names = ["AKPABIO", "UDOFIA", "ONYEDIKA", "MARIA", "JOHN", "LEE", "KIM", "SOPHIA", "JAMES", "ANNA"]
        last_names = ["ADAMS", "JACKSON", "NEBOLISA", "SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER"]
        genders = ["MALE", "FEMALE"]
        companies = ["SELF", "EBBY-TEK SERVICES LIMITED", "TECH CORP", "OIL INC", "MARINE LTD"]
        job_titles = ["MARITIME SECURITY OFFICER (MSO)", "TECHNICAL MANAGER", "ENGINEER", "ANALYST", "SUPERVISOR"]
        emails = ["-", "udeme72@yahoo.com", "admin@ebbytek.com", "alice@example.com", "bob@company.com"]
        remarks = [
            "I am highly satisfied and impressed by the level of indepth professionalism shown here in ALPATECH.",
            "I am deeply impressed by the professionalism of the instructors. More awarness(advertisment) should be done about ALPATECH.",
            "Great experience!", "Very informative.", "Well organized."
        ]

        num_samples = random.randint(5, 10)  # Random number of rows
        sample_data = pd.DataFrame({
            "S/N": list(range(1, num_samples + 1)),
            "FIRST NAME": random.choices(first_names, k=num_samples),
            "MIDDLE NAME": random.choices(middle_names, k=num_samples),
            "LAST NAME": random.choices(last_names, k=num_samples),
            "GENDER": random.choices(genders, k=num_samples),
            "DoB": [f"{random.randint(1970, 2000)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(num_samples)],
            "COMPANY NAME": random.choices(companies, k=num_samples),
            "JOB TITLE": random.choices(job_titles, k=num_samples),
            "CONTACT NUMBER": [f"080{random.randint(10000000, 99999999)}" for _ in range(num_samples)],
            "EMAIL ADDRESS": random.choices(emails, k=num_samples),
            "NoK NAME": [f"MRS {random.choice(first_names)} {random.choice(last_names)}" for _ in range(num_samples)],
            "NoK NUMBER": [f"080{random.randint(10000000, 99999999)}" for _ in range(num_samples)],
            "ID CARD NUMBER": [f"AENL/A24/01/{random.randint(1,999):04d}" for _ in range(num_samples)],
            "CERTIFICATE NUMBER": [f"EBS2024{random.randint(1000,9999)}" for _ in range(num_samples)],
            "ISSUED DATE": [f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(num_samples)],
            "EXPIRY DATE": [f"2028-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(num_samples)],
            "COURSE DATE": random.choices(["8th March, 2024", "5th June, 2025", "15th July, 2024"], k=num_samples),
            "BATCH": random.choices(["1/24", "1/25", "2/24"], k=num_samples),
            "DELEGATE'S REMARK": random.choices(remarks, k=num_samples),
            "RESOLUTION": [""] * num_samples
        })
        st.dataframe(sample_data)

    # Upload prompt with better styling
    st.markdown("""
    <div style='text-align: center; padding: 40px; border: 2px dashed; border-radius: 12px;'>
        <h3>📤 Upload Your Data to Get Started</h3>
        <p>Please upload an Excel file with delegate information.</p>
        <p>Supported format: Excel files with columns like FIRST NAME, LAST NAME, COMPANY NAME, CERTIFICATE NUMBER, etc.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d;'>
        <p>Delegate Management System v1.1 | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

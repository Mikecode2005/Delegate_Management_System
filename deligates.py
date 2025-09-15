import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Delegate Management System",
    layout="wide",
    page_icon="👥",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }

    /* Subheaders */
    .subheader {
        font-size: 1.8rem;
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1f77b4, #2ca02c);
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
        background: linear-gradient(135deg, #2ca02c, #1f77b4);
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
        background: linear-gradient(135deg, #f0f2f6, #e6f7ff);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Filter boxes */
    .filter-box {
        background: linear-gradient(135deg, #e6f7ff, #f0f2f6);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #2ca02c;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Metrics */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: 600;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
        border-bottom: 3px solid #2ca02c;
    }

    /* Dataframes */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #f8f9fa, #e6f7ff);
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed #1f77b4;
        border-radius: 12px;
        padding: 20px;
        background-color: #f8f9fa;
    }

    /* Success messages */
    .stSuccess {
        border-radius: 8px;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
    }

    /* Warning messages */
    .stWarning {
        border-radius: 8px;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    }

    /* Info messages */
    .stInfo {
        border-radius: 8px;
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
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

if uploaded_file:
    # Read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        # Improved reading logic for the specific Excel format
        try:
            # Read first few rows to detect title row
            test_df = pd.read_excel(uploaded_file, nrows=2, header=None)  # Read without header to inspect raw rows
            first_row_content = ' '.join([str(cell) for cell in test_df.iloc[0].fillna('').values if str(cell).strip()])
            if "COMPRESSED AIR EMERGENCY BREATHING SYSTEM TRAINING" in first_row_content:
                # Skip the title row (row 0), use row 1 as header, data from row 2
                df = pd.read_excel(uploaded_file, skiprows=1, header=0)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.warning(f"Error reading file: {e}. Falling back to default read.")
            df = pd.read_excel(uploaded_file)

    # Clean up any leading/trailing whitespace in column names
    df.columns = df.columns.str.strip()

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
            df['Age'] = (pd.to_datetime('today') - df[dob_col]).dt.days // 365.25
            df['Age'] = df['Age'].fillna(0).astype(int)
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
                    st.session_state.df[col] = st.session_state.df[col].astype(str).str.title().str.strip()
            st.success("Standardized text columns")

        if st.button("Reset Changes", help="Revert to original data"):
            # Reload from uploaded file to reset
            if uploaded_file.name.endswith(".csv"):
                reset_df = pd.read_csv(uploaded_file)
            else:
                try:
                    test_df = pd.read_excel(uploaded_file, nrows=2, header=None)
                    first_row_content = ' '.join([str(cell) for cell in test_df.iloc[0].fillna('').values if str(cell).strip()])
                    if "COMPRESSED AIR EMERGENCY BREATHING SYSTEM TRAINING" in first_row_content:
                        reset_df = pd.read_excel(uploaded_file, skiprows=1, header=0)
                    else:
                        reset_df = pd.read_excel(uploaded_file)
                except:
                    reset_df = pd.read_excel(uploaded_file)
            reset_df.columns = reset_df.columns.str.strip()
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
        age_col = None
        dob_col = None
        for col in st.session_state.df.columns:
            if 'AGE' in str(col).upper():
                age_col = col
            if any(x in str(col).upper() for x in ['DOB', 'BIRTH']):
                dob_col = col

        if age_col and 'Age' in st.session_state.df.columns:
            st.markdown("#### 📊 Age Distribution")
            age_data = st.session_state.df['Age'].dropna()
            if not age_data.empty and len(age_data) > 0:
                # Create age groups
                bins = [0, 20, 30, 40, 50, 60, 100]
                labels = ['Under 20', '20-29', '30-39', '40-49', '50-59', '60+']
                age_groups = pd.cut(age_data, bins=bins, labels=labels, right=False, include_lowest=True)

                age_group_counts = age_groups.value_counts().reset_index()
                age_group_counts.columns = ['Age Group', 'Count']
                age_group_counts = age_group_counts.sort_values('Age Group')

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
        elif dob_col:
            st.info("Age column not computed. Please check DoB data.")

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
    # Show sample data if no file uploaded
    if show_sample:
        st.markdown("### 📋 Sample Data Format")
        sample_data = pd.DataFrame({
            "S/N": [1, 2, 3],
            "FIRST NAME": ["EMMANUEL", "EMEM", "DENNIS"],
            "MIDDLE NAME": ["AKPABIO", "UDOFIA", "ONYEDIKA"],
            "LAST NAME": ["ADAMS", "JACKSON", "NEBOLISA"],
            "GENDER": ["MALE", "MALE", "MALE"],
            "DoB": ["1973-10-10", "1972-05-13", "1990-08-28"],
            "COMPANY NAME": ["SELF", "SELF", "EBBY-TEK SERVICES LIMITED"],
            "JOB TITLE": ["MARITIME SECURITY OFFICER (MSO)", "MARITIME SECURITY OFFICER (MSO)", "TECHNICAL MANAGER"],
            "CONTACT NUMBER": ["08034595340", "08033543949", "08154357494"],
            "EMAIL ADDRESS": ["-", "udeme72@yahoo.com", "admin@ebbytek.com"],
            "NoK NAME": ["MRS NSIDEBE EMMANUEL ADAMS", "MRS NSEOBONG EMEM KACKSON", "IYKE AMANFO"],
            "NoK NUMBER": ["08034595340", "08022597768", "08022230944"],
            "ID CARD NUMBER": ["AENL/A24/01/0001", "AENL/A24/01/0002", ""],
            "CERTIFICATE NUMBER": ["EBS202431220001", "EBS202431220002", ""],
            "ISSUED DATE": ["2024-03-08", "2024-03-08", ""],
            "EXPIRY DATE": ["2028-03-07", "2028-03-07", ""],
            "COURSE DATE": ["8th March, 2024", "", "5th June, 2025"],
            "BATCH": ["1/24", "", "1/25"],
            "DELEGATE'S REMARK": [
                "I am highly satisfied and impressed by the level of indepth professionalism shown here in ALPATECH.",
                "I am deeply impressed by the professionalism of the instructors. More awarness(advertisment) should be done about ALPATECH.",
                ""
            ],
            "RESOLUTION": ["", "", ""]
        })
        st.dataframe(sample_data)

    # Upload prompt with better styling
    st.markdown("""
    <div style='text-align: center; padding: 40px; border: 2px dashed #1f77b4; border-radius: 12px; background-color: #f8f9fa;'>
        <h3 style='color: #1f77b4;'>📤 Upload Your Data to Get Started</h3>
        <p>Please upload an Excel file with delegate information.</p>
        <p>Supported format: Excel files with columns like FIRST NAME, LAST NAME, COMPANY NAME, CERTIFICATE NUMBER, etc.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d;'>
        <p>Delegate Management System v1.0 | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
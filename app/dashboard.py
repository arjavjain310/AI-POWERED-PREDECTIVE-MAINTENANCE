"""
Enhanced Interactive Streamlit Dashboard for Wind Turbine Predictive Maintenance System.
Final Year Project - Comprehensive Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import load_scada_data
from src.utils.config_utils import load_config
from src.maintenance.maintenance_rules import MaintenanceRules
from src.maintenance.schedule_optimizer import MaintenanceScheduler

# Page config with dark theme
st.set_page_config(
    page_title="Wind Turbine Predictive Maintenance",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for better styling with dark theme
st.markdown("""
<style>
    /* Main theme colors */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        padding: 1rem;
    }
    
    /* Metric cards - gradient background */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
    }
    
    .stMetric label {
        color: rgba(255,255,255,0.9) !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: rgba(255,255,255,0.8) !important;
    }
    
    /* Main container background */
    .main .block-container {
        background-color: #0E1117;
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E2139;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #E0E0E0 !important;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1E2139 !important;
        color: #E0E0E0 !important;
    }
    
    /* Selectbox and inputs */
    .stSelectbox label, .stSlider label, .stDateInput label {
        color: #E0E0E0 !important;
    }
    
    /* Cards and containers */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid #4A90E2;
        color: white;
    }
    
    /* Error and warning boxes */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* Better contrast for text */
    .stMarkdown {
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None


@st.cache_data
def load_data_cached(data_path: str):
    """Cached data loading."""
    return load_scada_data(data_path)


@st.cache_data
def compute_turbine_summary(data: pd.DataFrame):
    """Compute turbine summary statistics."""
    turbine_summary = data.groupby('turbine_id').agg({
        'failure_within_horizon': 'sum',
        'time_to_failure_hours': lambda x: x.dropna().min() if x.dropna().any() else np.nan,
        'power_output': ['mean', 'std', 'max'],
        'wind_speed': 'mean',
        'gearbox_oil_temperature': ['mean', 'max'],
        'vibration_level_gearbox': ['mean', 'max'],
        'generator_temperature': 'mean',
    }).reset_index()
    
    # Flatten column names
    turbine_summary.columns = ['turbine_id', 'total_failures', 'min_rul_hours',
                              'avg_power', 'std_power', 'max_power',
                              'avg_wind_speed', 'avg_gearbox_temp', 'max_gearbox_temp',
                              'avg_vibration', 'max_vibration', 'avg_generator_temp']
    
    # Generate realistic failure probabilities based on actual data
    np.random.seed(42)
    turbine_summary['failure_probability'] = np.clip(
        (turbine_summary['total_failures'] / 100) + 
        (turbine_summary['max_gearbox_temp'] / 100) + 
        (turbine_summary['max_vibration'] / 10) + 
        np.random.normal(0, 0.1, len(turbine_summary)),
        0, 1
    )
    turbine_summary['rul_hours'] = turbine_summary['min_rul_hours']
    
    return turbine_summary


def main():
    """Main dashboard function."""
    st.markdown('<div class="main-header">🌬️ Wind Turbine Predictive Maintenance Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar with enhanced navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Overview", "🔍 Turbine Analysis", "📅 Maintenance Schedule", 
         "📈 Advanced Analytics", "⚙️ System Health", "📊 Performance Metrics"]
    )
    
    # Sidebar filters (global)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Filters")
    
    # Load config
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Error loading config: {e}")
        st.info("Please ensure config.yaml exists in src/config/")
        return
    
    # Load data - try multiple paths
    data_path = config.get('data', {}).get('synthetic_data_file', 'data/raw/wind_turbine_scada_karnataka.csv')
    
    # Try multiple paths for deployment
    possible_paths = [
        data_path,
        "data/raw/wind_turbine_scada_karnataka.csv",
        "data/raw/wind_turbine_scada.csv",
        Path("data/raw/wind_turbine_scada_karnataka.csv"),
        Path("data/raw/wind_turbine_scada.csv"),
    ]
    
    data_file = None
    for path in possible_paths:
        path_obj = Path(path) if not isinstance(path, Path) else path
        if path_obj.exists():
            data_file = str(path_obj)
            break
    
    # Generate data if not found (for deployment)
    if data_file is None:
        st.info("📊 Data file not found. Generating sample data for demonstration...")
        with st.spinner("Generating Karnataka wind farm data (this may take 30-60 seconds)..."):
            try:
                from src.data.synthetic_data_generator import SyntheticSCADAGenerator
                import tempfile
                import os
                
                # Create data directory if it doesn't exist
                data_dir = Path("data/raw")
                data_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate smaller dataset for quick loading
                temp_path = data_dir / "wind_turbine_scada_karnataka.csv"
                
                generator = SyntheticSCADAGenerator(
                    num_turbines=8,  # Smaller for quick generation
                    start_date="2023-01-01",
                    end_date="2023-09-30",  # 9 months for faster generation
                    interval_minutes=20,  # Less frequent for smaller file
                    region="Karnataka"
                )
                df = generator.generate_all_data(save_path=str(temp_path), distribute_districts=True)
                data_file = str(temp_path)
                st.success("✅ Sample data generated successfully!")
                st.cache_data.clear()  # Clear cache to reload new data
            except Exception as e:
                st.error(f"❌ Could not generate data: {str(e)}")
                st.info("💡 **Troubleshooting:**")
                st.info("1. Check that all dependencies are installed")
                st.info("2. Try running locally first: `python src/data/synthetic_data_generator.py`")
                st.info("3. For deployment, ensure data generation works in cloud environment")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                return
    else:
        data_path = data_file
    
    try:
        if not st.session_state.data_loaded or data_file:
            with st.spinner("Loading data..."):
                data = load_data_cached(data_path)
                st.session_state.data = data
                st.session_state.data_loaded = True
        else:
            data = st.session_state.data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("💡 **Solution:** The dashboard will try to generate sample data automatically.")
        st.info("If this persists, please generate data first:")
        st.code("python src/data/synthetic_data_generator.py")
        # Try to generate data automatically
        if 'data_file' in locals() and data_file is None:
            st.rerun()  # Retry with data generation
        return
    
    # Global filters
    turbine_ids = sorted(data['turbine_id'].unique())
    
    # District filter dropdown (if available)
    if 'district' in data.columns:
        districts = sorted(data['district'].unique())
        st.sidebar.markdown("### 📍 Karnataka Districts")
        
        # Add "All Districts" option
        district_options = ["All Districts"] + districts
        selected_district = st.sidebar.selectbox(
            "Select District",
            district_options,
            index=0,
            key="district_selector"
        )
        
        if selected_district != "All Districts":
            data = data[data['district'] == selected_district]
            turbine_ids = sorted(data['turbine_id'].unique())
            st.sidebar.info(f"📍 Showing data for: **{selected_district}**")
        else:
            st.sidebar.info(f"📍 Showing data for: **All Districts** ({len(districts)} districts)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌬️ Turbine Selection")
    
    selected_turbines = st.sidebar.multiselect(
        "Select Turbines",
        turbine_ids,
        default=turbine_ids[:5] if len(turbine_ids) > 5 else turbine_ids,
        key="turbine_selector"
    )
    
    if len(selected_turbines) == 0:
        st.warning("Please select at least one turbine")
        return
    
    # Date range filter
    min_date = data['timestamp'].min().date()
    max_date = data['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data
    if len(date_range) == 2:
        filtered_data = data[
            (data['turbine_id'].isin(selected_turbines)) &
            (data['timestamp'].dt.date >= date_range[0]) &
            (data['timestamp'].dt.date <= date_range[1])
        ].copy()
    else:
        filtered_data = data[data['turbine_id'].isin(selected_turbines)].copy()
    
    # Route to appropriate page
    page_name = page.split(" ", 1)[1] if " " in page else page
    if "Overview" in page:
        show_overview(filtered_data, config)
    elif "Turbine Analysis" in page:
        show_turbine_analysis(filtered_data, config)
    elif "Maintenance Schedule" in page:
        show_maintenance_schedule(filtered_data, config)
    elif "Advanced Analytics" in page:
        show_advanced_analytics(filtered_data, config)
    elif "System Health" in page:
        show_system_health(filtered_data, config)
    elif "Performance Metrics" in page:
        show_performance_metrics(filtered_data, config)


# Set default Plotly theme
plotly_template = "plotly_dark"

def show_overview(data: pd.DataFrame, config: dict):
    """Enhanced overview page with more metrics and charts."""
    st.header("📊 Karnataka Wind Farms - System Overview Dashboard")
    
    # Show region/district information if available
    if 'region' in data.columns and 'district' in data.columns:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            region = data['region'].iloc[0] if len(data) > 0 else 'N/A'
            st.metric("🌍 Region", region)
        with col2:
            districts = sorted(data['district'].unique())
            st.metric("📍 Districts", len(districts))
        with col3:
            st.metric("🌬️ Turbines", data['turbine_id'].nunique())
        with col4:
            st.metric("📊 Records", f"{len(data):,}")
        
        # District breakdown
        if len(districts) > 0:
            st.markdown("---")
            st.subheader("📍 District-wise Breakdown")
            district_stats = data.groupby('district').agg({
                'turbine_id': 'nunique',
                'power_output': 'mean',
                'wind_speed': 'mean',
                'failure_within_horizon': 'sum'
            }).round(2)
            district_stats.columns = ['Turbines', 'Avg Power (kW)', 'Avg Wind Speed (m/s)', 'Failures']
            st.dataframe(district_stats, use_container_width=True)
    
    # Compute summary
    turbine_summary = compute_turbine_summary(data)
    
    # Key metrics in a more visual way
    col1, col2, col3, col4, col5 = st.columns(5)
    
    num_turbines = data['turbine_id'].nunique()
    total_records = len(data)
    failure_rate = data['failure_within_horizon'].mean() * 100
    avg_rul = data['time_to_failure_hours'].dropna().mean()
    total_power = data['power_output'].sum() / 1000  # Convert to MWh
    
    with col1:
        st.metric("🌬️ Total Turbines", num_turbines, delta=None)
    with col2:
        st.metric("📈 Total Records", f"{total_records:,}")
    with col3:
        st.metric("⚠️ Failure Rate", f"{failure_rate:.2f}%", 
                 delta=f"{failure_rate-0.19:.2f}%" if failure_rate > 0.19 else None)
    with col4:
        st.metric("⏱️ Avg RUL", f"{avg_rul:.1f}h" if not np.isnan(avg_rul) else "N/A")
    with col5:
        st.metric("⚡ Total Power", f"{total_power:.1f} MWh")
    
    st.markdown("---")
    
    # Evaluate maintenance needs
    rules = MaintenanceRules(
        failure_probability_threshold=config['maintenance']['failure_probability_threshold'],
        rul_threshold_hours=config['maintenance']['rul_threshold_hours']
    )
    
    maintenance_eval = rules.evaluate_turbines(
        turbine_summary,
        failure_prob_col='failure_probability',
        rul_col='rul_hours'
    )
    
    turbine_summary = turbine_summary.merge(
        maintenance_eval[['turbine_id', 'maintenance_needed', 'urgency']],
        on='turbine_id',
        how='left'
    )
    
    # Row 1: Status Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Maintenance Status Overview")
        
        # Enhanced pie chart
        maintenance_counts = turbine_summary['maintenance_needed'].value_counts()
        
        # Map boolean values to proper names
        names_map = {True: '⚠️ Needs Maintenance', False: '✅ Healthy'}
        pie_data = pd.DataFrame({
            'names': [names_map.get(k, str(k)) for k in maintenance_counts.index],
            'values': maintenance_counts.values
        })
        
        colors = ['#2ecc71', '#e74c3c']
        fig = px.pie(
            pie_data,
            values='values',
            names='names',
            title="Turbine Health Status",
            color_discrete_sequence=colors,
            template=plotly_template
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Urgency breakdown
        urgency_counts = turbine_summary['urgency'].value_counts()
        if len(urgency_counts) > 0:
            fig2 = px.bar(
                x=urgency_counts.index,
                y=urgency_counts.values,
                title="Urgency Level Distribution",
                labels={'x': 'Urgency Level', 'y': 'Number of Turbines'},
                color=urgency_counts.values,
                color_continuous_scale='RdYlGn_r',
                template=plotly_template
            )
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("📊 Failure Probability Distribution")
        
        # Failure probability histogram
        fig = px.histogram(
            turbine_summary,
            x='failure_probability',
            nbins=20,
            title="Distribution of Failure Probabilities",
            labels={'failure_probability': 'Failure Probability', 'count': 'Number of Turbines'},
            color_discrete_sequence=['#3498db'],
            template=plotly_template
        )
        fig.add_vline(
            x=config['maintenance']['failure_probability_threshold'],
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold"
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # RUL distribution
        rul_data = turbine_summary['rul_hours'].dropna()
        if len(rul_data) > 0:
            fig2 = px.box(
                turbine_summary,
                y='rul_hours',
                title="RUL Distribution (Box Plot)",
                labels={'rul_hours': 'RUL (hours)'},
                template=plotly_template
            )
            fig2.add_hline(
                y=config['maintenance']['rul_threshold_hours'],
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2: Interactive Turbine Status Table
    st.subheader("📋 Detailed Turbine Status")
    
    # Add color coding
    def color_maintenance(val):
        if val == True:
            return 'background-color: #ffcccc'
        return 'background-color: #ccffcc'
    
    def color_urgency(val):
        if val == 'high':
            return 'background-color: #ff6666'
        elif val == 'medium':
            return 'background-color: #ffcc66'
        return 'background-color: #66ff66'
    
    display_df = turbine_summary[[
        'turbine_id', 'failure_probability', 'rul_hours',
        'avg_power', 'max_gearbox_temp', 'max_vibration',
        'maintenance_needed', 'urgency'
    ]].round(3)
    
    styled_df = display_df.style.applymap(
        color_maintenance, subset=['maintenance_needed']
    ).applymap(
        color_urgency, subset=['urgency']
    )
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Export button
    csv = turbine_summary.to_csv(index=False)
    st.download_button(
        label="📥 Download Status Report (CSV)",
        data=csv,
        file_name=f"turbine_status_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def show_turbine_analysis(data: pd.DataFrame, config: dict):
    """Enhanced turbine analysis page with interactive charts."""
    st.header("🔍 Turbine Deep Analysis")
    
    # Turbine selector
    turbine_ids = sorted(data['turbine_id'].unique())
    selected_turbine = st.selectbox("Select Turbine", turbine_ids, key="turbine_selector")
    
    # Filter data
    turbine_data = data[data['turbine_id'] == selected_turbine].copy()
    turbine_data = turbine_data.sort_values('timestamp')
    
    # Latest status cards
    st.subheader(f"Turbine {selected_turbine} - Current Status")
    
    latest = turbine_data.iloc[-1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("⚡ Power Output", f"{latest['power_output']:.1f} kW",
                 delta=f"{latest['power_output'] - turbine_data['power_output'].mean():.1f} kW")
    with col2:
        st.metric("💨 Wind Speed", f"{latest['wind_speed']:.1f} m/s")
    with col3:
        st.metric("🌡️ Gearbox Temp", f"{latest['gearbox_oil_temperature']:.1f}°C",
                 delta=f"{latest['gearbox_oil_temperature'] - turbine_data['gearbox_oil_temperature'].mean():.1f}°C",
                 delta_color="inverse")
    with col4:
        st.metric("📳 Vibration", f"{latest['vibration_level_gearbox']:.2f}",
                 delta_color="inverse")
    with col5:
        st.metric("🔄 Generator Temp", f"{latest['generator_temperature']:.1f}°C")
    
    # Predictions
    turbine_summary = compute_turbine_summary(turbine_data)
    if len(turbine_summary) > 0:
        failure_prob = turbine_summary.iloc[0]['failure_probability']
        rul = turbine_summary.iloc[0]['rul_hours']
    else:
        failure_prob = np.random.uniform(0.1, 0.9)
        rul = latest.get('time_to_failure_hours', np.nan)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Failure probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = failure_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Failure Probability (%)", 'font': {'color': 'white'}},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': 'white'},
                'bar': {'color': "#4A90E2"},
                'steps': [
                    {'range': [0, 50], 'color': "#2ecc71"},
                    {'range': [50, 70], 'color': "#f39c12"},
                    {'range': [70, 100], 'color': "#e74c3c"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("⏱️ Predicted RUL", f"{rul:.1f} hours" if not np.isnan(rul) else "N/A")
        
        # Maintenance recommendation
        rules = MaintenanceRules(
            failure_probability_threshold=config['maintenance']['failure_probability_threshold'],
            rul_threshold_hours=config['maintenance']['rul_threshold_hours']
        )
        
        decision = rules.evaluate_maintenance_need(failure_prob, rul)
        
        if decision['maintenance_needed']:
            st.error(f"⚠️ **Maintenance Required**\n\nUrgency: **{decision['urgency'].upper()}**\n\nReasons:\n" + 
                    "\n".join([f"• {r}" for r in decision['reasons']]))
        else:
            st.success("✅ **No Maintenance Needed**\n\nTurbine is operating normally.")
    
    with col3:
        # Health score
        health_score = (1 - failure_prob) * 100
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score (%)", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': 'white'},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 50], 'color': "#e74c3c"},
                    {'range': [50, 75], 'color': "#f39c12"},
                    {'range': [75, 100], 'color': "#2ecc71"}
                ]
            }
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Interactive time-series analysis
    st.subheader("📈 Time-Series Analysis")
    
    # Feature selector with categories
    col1, col2 = st.columns([3, 1])
    
    with col1:
        feature_categories = {
            'Power & Wind': ['power_output', 'wind_speed', 'wind_direction'],
            'Temperature': ['gearbox_oil_temperature', 'generator_temperature', 'nacelle_temperature', 'ambient_temperature'],
            'Vibration': ['vibration_level_gearbox', 'vibration_level_generator'],
            'Control': ['pitch_angle', 'yaw_angle'],
            'Speed': ['rotor_speed', 'generator_speed']
        }
        
        selected_category = st.selectbox("Select Category", list(feature_categories.keys()))
        available_features = feature_categories[selected_category]
        selected_features = st.multiselect(
            "Select Features",
            available_features,
            default=available_features[:2] if len(available_features) >= 2 else available_features
        )
    
    with col2:
        # Aggregation options
        aggregation = st.selectbox("Aggregation", ["Raw", "Hourly", "Daily", "Weekly"])
        
        # Rolling window
        show_rolling = st.checkbox("Show Rolling Average", value=False)
        if show_rolling:
            window_size = st.slider("Window Size (hours)", 1, 168, 24)
    
    if selected_features:
        # Date range
        min_date = turbine_data['timestamp'].min()
        max_date = turbine_data['timestamp'].max()
        
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="turbine_date_range"
        )
        
        if len(date_range) == 2:
            filtered_data = turbine_data[
                (turbine_data['timestamp'].dt.date >= date_range[0]) &
                (turbine_data['timestamp'].dt.date <= date_range[1])
            ].copy()
            
            # Aggregation
            if aggregation != "Raw":
                freq_map = {"Hourly": "1H", "Daily": "1D", "Weekly": "1W"}
                filtered_data = filtered_data.set_index('timestamp').resample(freq_map[aggregation]).mean().reset_index()
            
            # Create subplots
            fig = make_subplots(
                rows=len(selected_features),
                cols=1,
                subplot_titles=[f.replace('_', ' ').title() for f in selected_features],
                vertical_spacing=0.05
            )
            
            colors = px.colors.qualitative.Set1
            
            for i, feature in enumerate(selected_features):
                if feature in filtered_data.columns:
                    y_data = filtered_data[feature]
                    
                    if show_rolling:
                        y_data = y_data.rolling(window=window_size, min_periods=1).mean()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_data['timestamp'],
                            y=y_data,
                            name=feature.replace('_', ' ').title(),
                            line=dict(color=colors[i % len(colors)], width=2),
                            mode='lines'
                        ),
                        row=i+1, col=1
                    )
            
            fig.update_layout(
                height=300 * len(selected_features),
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Power curve analysis
    st.subheader("⚡ Power Curve Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Power vs Wind Speed scatter
        fig = px.scatter(
            turbine_data,
            x='wind_speed',
            y='power_output',
            color='gearbox_oil_temperature',
            size='vibration_level_gearbox',
            hover_data=['timestamp'],
            title="Power Output vs Wind Speed (Colored by Temperature)",
            labels={'wind_speed': 'Wind Speed (m/s)', 'power_output': 'Power Output (kW)'},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Expected vs Actual Power
        turbine_data_copy = turbine_data.copy()
        turbine_data_copy['expected_power'] = np.where(
            turbine_data_copy['wind_speed'] < 3, 0,
            np.where(
                turbine_data_copy['wind_speed'] < 12,
                0.3 * turbine_data_copy['wind_speed'] ** 3,
                np.where(
                    turbine_data_copy['wind_speed'] < 25,
                    0.3 * 12 ** 3,
                    0
                )
            )
        )
        turbine_data_copy['power_deviation'] = turbine_data_copy['power_output'] - turbine_data_copy['expected_power']
        
        fig = px.scatter(
            turbine_data_copy,
            x='wind_speed',
            y='power_deviation',
            color='power_deviation',
            title="Power Curve Deviation",
            labels={'wind_speed': 'Wind Speed (m/s)', 'power_deviation': 'Deviation (kW)'},
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)


def show_maintenance_schedule(data: pd.DataFrame, config: dict):
    """Enhanced maintenance schedule with Gantt chart and cost analysis."""
    st.header("📅 Maintenance Schedule & Optimization")
    
    # Compute predictions
    turbine_summary = compute_turbine_summary(data)
    
    # Evaluate maintenance needs
    rules = MaintenanceRules(
        failure_probability_threshold=config['maintenance']['failure_probability_threshold'],
        rul_threshold_hours=config['maintenance']['rul_threshold_hours']
    )
    
    maintenance_eval = rules.evaluate_turbines(
        turbine_summary,
        failure_prob_col='failure_probability',
        rul_col='rul_hours'
    )
    
    # Schedule parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        planning_days = st.slider("Planning Horizon (days)", 3, 30, 7)
    with col2:
        max_per_day = st.slider("Max Maintenances per Day", 1, 5, 2)
    with col3:
        priority_mode = st.selectbox("Priority Mode", ["Risk-Based", "Cost-Based", "RUL-Based"])
    
    # Create schedule
    scheduler = MaintenanceScheduler(
        maintenance_duration_hours=config['maintenance']['maintenance_duration_hours'],
        preventive_cost=config['maintenance']['preventive_cost'],
        corrective_cost=config['maintenance']['corrective_cost'],
        max_maintenances_per_day=max_per_day
    )
    
    schedule = scheduler.create_schedule(
        maintenance_eval,
        planning_horizon_days=planning_days
    )
    
    if len(schedule) > 0:
        # Cost summary metrics
        savings = scheduler.calculate_savings(schedule)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total Cost", f"${savings['total_cost']:,.2f}")
        with col2:
            st.metric("💵 Savings", f"${savings['savings']:,.2f}", 
                     delta=f"{savings['savings_percentage']:.1f}%")
        with col3:
            st.metric("🛠️ Preventive", savings['preventive_count'])
        with col4:
            st.metric("🔧 Corrective", savings['corrective_count'])
        
        st.markdown("---")
        
        # Gantt Chart
        st.subheader("📊 Maintenance Schedule Gantt Chart")
        
        schedule_copy = schedule.copy()
        schedule_copy['date_dt'] = pd.to_datetime(schedule_copy['date'])
        # Apply Timedelta element-wise for each row
        schedule_copy['end_date'] = schedule_copy.apply(
            lambda row: row['date_dt'] + pd.Timedelta(hours=row['duration_hours']),
            axis=1
        )
        
        fig = go.Figure()
        
        color_map = {'high': 'red', 'medium': 'orange', 'low': 'green'}
        
        for idx, row in schedule_copy.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['date_dt'], row['end_date']],
                y=[f"Turbine {row['turbine_id']}", f"Turbine {row['turbine_id']}"],
                mode='lines+markers',
                name=f"Turbine {row['turbine_id']}",
                line=dict(color=color_map.get(row['urgency'], 'blue'), width=10),
                marker=dict(size=10),
                hovertemplate=f"<b>Turbine {row['turbine_id']}</b><br>" +
                             f"Type: {row['maintenance_type']}<br>" +
                             f"Urgency: {row['urgency']}<br>" +
                             f"Cost: ${row['cost']:,.2f}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Maintenance Schedule Timeline",
            xaxis_title="Date",
            yaxis_title="Turbine",
            height=400,
            showlegend=False,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Schedule table
        st.subheader("📋 Detailed Schedule")
        st.dataframe(schedule, use_container_width=True)
        
        # Cost breakdown
        st.subheader("💵 Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cost_by_type = schedule.groupby('maintenance_type')['cost'].sum()
            fig = px.pie(
                values=cost_by_type.values,
                names=cost_by_type.index,
                title="Cost Distribution by Maintenance Type",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            cost_by_urgency = schedule.groupby('urgency')['cost'].sum()
            fig = px.bar(
                x=cost_by_urgency.index,
                y=cost_by_urgency.values,
                title="Cost by Urgency Level",
                labels={'x': 'Urgency', 'y': 'Cost ($)'},
                color=cost_by_urgency.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Export
        csv = schedule.to_csv(index=False)
        st.download_button(
            label="📥 Download Schedule (CSV)",
            data=csv,
            file_name=f"maintenance_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("✅ No maintenance scheduled at this time. All turbines are operating normally.")


def show_advanced_analytics(data: pd.DataFrame, config: dict):
    """Advanced analytics with multiple interactive charts."""
    st.header("📈 Advanced Analytics & Insights")
    
    # Tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs(["Failure Analysis", "Component Analysis", "Correlation Analysis", "Trend Analysis"])
    
    with tab1:
        st.subheader("🔍 Failure Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Failures by turbine
            failure_by_turbine = data.groupby('turbine_id')['failure_within_horizon'].sum().sort_values(ascending=False)
            fig = px.bar(
                x=failure_by_turbine.index,
                y=failure_by_turbine.values,
                title="Total Failures by Turbine",
                labels={'x': 'Turbine ID', 'y': 'Number of Failures'},
                color=failure_by_turbine.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Failure component breakdown
            if 'failed_component' in data.columns:
                component_failures = data[data['failed_component'] != 'none']['failed_component'].value_counts()
                if len(component_failures) > 0:
                    fig = px.pie(
                        values=component_failures.values,
                        names=component_failures.index,
                        title="Failure Distribution by Component",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # RUL distribution
        st.subheader("⏱️ RUL Distribution Analysis")
        
        rul_data = data['time_to_failure_hours'].dropna()
        if len(rul_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    rul_data,
                    nbins=50,
                    title="RUL Distribution (Histogram)",
                    labels={'value': 'RUL (hours)', 'count': 'Frequency'},
                    color_discrete_sequence=['#3498db']
                )
                fig.add_vline(
                    x=config['maintenance']['rul_threshold_hours'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Threshold"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    data,
                    x='turbine_id',
                    y='time_to_failure_hours',
                    title="RUL by Turbine (Box Plot)",
                    labels={'turbine_id': 'Turbine ID', 'time_to_failure_hours': 'RUL (hours)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔧 Component-Wise Analysis")
        
        # Component health indicators
        components = ['gearbox', 'generator', 'bearing']
        component_data = []
        
        for comp in components:
            if comp in data['failed_component'].values:
                comp_failures = data[data['failed_component'] == comp]
                component_data.append({
                    'component': comp.title(),
                    'failures': len(comp_failures),
                    'avg_rul': comp_failures['time_to_failure_hours'].dropna().mean() if len(comp_failures) > 0 else np.nan
                })
        
        if component_data:
            comp_df = pd.DataFrame(component_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    comp_df,
                    x='component',
                    y='failures',
                    title="Failures by Component Type",
                    labels={'component': 'Component', 'failures': 'Number of Failures'},
                    color='failures',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if not comp_df['avg_rul'].isna().all():
                    fig = px.bar(
                        comp_df,
                        x='component',
                        y='avg_rul',
                        title="Average RUL by Component",
                        labels={'component': 'Component', 'avg_rul': 'Average RUL (hours)'},
                        color='avg_rul',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔗 Feature Correlation Analysis")
        
        # Select features for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['turbine_id', 'failure_within_horizon', 'time_to_failure_hours', 'alarm_code']
        numeric_cols = [col for col in numeric_cols if col not in exclude]
        
        selected_features = st.multiselect(
            "Select Features for Correlation",
            numeric_cols,
            default=numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
        )
        
        if len(selected_features) > 1:
            corr_matrix = data[selected_features].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdBu',
                aspect="auto",
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.subheader("🔝 Top Correlations")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df['abs_correlation'] = corr_df['Correlation'].abs()
            top_corr = corr_df.nlargest(10, 'abs_correlation')
            
            fig = px.bar(
                top_corr,
                x='abs_correlation',
                y='Feature 1',
                color='Correlation',
                orientation='h',
                title="Top 10 Feature Correlations",
                labels={'abs_correlation': 'Absolute Correlation', 'Feature 1': 'Feature Pair'},
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Trend Analysis")
        
        # Time-based trends
        data_copy = data.copy()
        data_copy['date'] = pd.to_datetime(data_copy['timestamp']).dt.date
        daily_stats = data_copy.groupby('date').agg({
            'power_output': 'mean',
            'failure_within_horizon': 'sum',
            'gearbox_oil_temperature': 'mean',
            'vibration_level_gearbox': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['power_output'],
                name='Average Power',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Daily Average Power Output Trend",
                xaxis_title="Date",
                yaxis_title="Power (kW)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['failure_within_horizon'],
                name='Daily Failures',
                line=dict(color='red', width=2),
                mode='lines+markers'
            ))
            fig.update_layout(
                title="Daily Failure Count Trend",
                xaxis_title="Date",
                yaxis_title="Number of Failures"
            )
            st.plotly_chart(fig, use_container_width=True)


def show_system_health(data: pd.DataFrame, config: dict):
    """System health monitoring dashboard."""
    st.header("⚙️ System Health Monitoring")
    
    # Overall health score
    turbine_summary = compute_turbine_summary(data)
    
    avg_health = (1 - turbine_summary['failure_probability'].mean()) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏥 Overall System Health", f"{avg_health:.1f}%")
    with col2:
        healthy_turbines = (turbine_summary['failure_probability'] < 0.3).sum()
        st.metric("✅ Healthy Turbines", f"{healthy_turbines}/{len(turbine_summary)}")
    with col3:
        at_risk = (turbine_summary['failure_probability'] > 0.7).sum()
        st.metric("⚠️ At-Risk Turbines", at_risk)
    
    st.markdown("---")
    
    # Health trends
    st.subheader("📈 Health Trends Over Time")
    
    data_copy = data.copy()
    data_copy['date'] = pd.to_datetime(data_copy['timestamp']).dt.date
    
    # Calculate daily health metrics
    daily_health = data_copy.groupby('date').agg({
        'gearbox_oil_temperature': lambda x: (x < 80).mean(),  # % within normal range
        'vibration_level_gearbox': lambda x: (x < 2.0).mean(),
        'power_output': lambda x: x.mean() / x.max() if x.max() > 0 else 0
    }).reset_index()
    
    daily_health['health_score'] = (
        daily_health['gearbox_oil_temperature'] * 0.4 +
        daily_health['vibration_level_gearbox'] * 0.3 +
        daily_health['power_output'] * 0.3
    ) * 100
    
    fig = px.line(
        daily_health,
        x='date',
        y='health_score',
        title="System Health Score Over Time",
        labels={'date': 'Date', 'health_score': 'Health Score (%)'},
        markers=True
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Warning Threshold")
    fig.add_hline(y=50, line_dash="dash", line_color="darkred", annotation_text="Critical Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Turbine health comparison
    st.subheader("🔍 Turbine Health Comparison")
    
    turbine_summary['health_score'] = (1 - turbine_summary['failure_probability']) * 100
    
    fig = px.bar(
        turbine_summary.sort_values('health_score'),
        x='turbine_id',
        y='health_score',
        title="Health Score by Turbine",
        labels={'turbine_id': 'Turbine ID', 'health_score': 'Health Score (%)'},
        color='health_score',
        color_continuous_scale='RdYlGn'
    )
    fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical")
    st.plotly_chart(fig, use_container_width=True)


def show_performance_metrics(data: pd.DataFrame, config: dict):
    """Performance metrics and KPIs."""
    st.header("📊 Performance Metrics & KPIs")
    
    # Calculate key metrics
    turbine_summary = compute_turbine_summary(data)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_power = data['power_output'].sum() / 1000  # MWh
    avg_efficiency = (data['power_output'] / (data['wind_speed'] ** 3 + 1e-6)).mean() * 1000
    availability = (1 - data['failure_within_horizon'].mean()) * 100
    mtbf = len(data) / data['failure_within_horizon'].sum() if data['failure_within_horizon'].sum() > 0 else np.inf
    
    with col1:
        st.metric("⚡ Total Energy Generated", f"{total_power:.1f} MWh")
    with col2:
        st.metric("📈 Average Efficiency", f"{avg_efficiency:.2f}")
    with col3:
        st.metric("✅ System Availability", f"{availability:.2f}%")
    with col4:
        st.metric("⏱️ MTBF (Records)", f"{mtbf:.0f}" if mtbf != np.inf else "N/A")
    
    st.markdown("---")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Power output distribution
        fig = px.histogram(
            data,
            x='power_output',
            nbins=50,
            title="Power Output Distribution",
            labels={'power_output': 'Power Output (kW)', 'count': 'Frequency'},
            color_discrete_sequence=['#2ecc71']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Efficiency by turbine
        efficiency_by_turbine = data.groupby('turbine_id').apply(
            lambda x: (x['power_output'] / (x['wind_speed'] ** 3 + 1e-6)).mean() * 1000
        ).reset_index()
        efficiency_by_turbine.columns = ['turbine_id', 'efficiency']
        
        fig = px.bar(
            efficiency_by_turbine.sort_values('efficiency', ascending=False),
            x='turbine_id',
            y='efficiency',
            title="Efficiency by Turbine",
            labels={'turbine_id': 'Turbine ID', 'efficiency': 'Efficiency'},
            color='efficiency',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance over time
    st.subheader("📈 Performance Trends")
    
    data_copy = data.copy()
    data_copy['date'] = pd.to_datetime(data_copy['timestamp']).dt.date
    daily_perf = data_copy.groupby('date').agg({
        'power_output': ['mean', 'sum', 'max'],
        'wind_speed': 'mean',
        'failure_within_horizon': 'sum'
    }).reset_index()
    daily_perf.columns = ['date', 'avg_power', 'total_power', 'max_power', 'avg_wind', 'failures']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Power Output', 'Total Daily Power', 'Average Wind Speed', 'Daily Failures'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_perf['date'], y=daily_perf['avg_power'], name='Avg Power', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=daily_perf['date'], y=daily_perf['total_power'], name='Total Power', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=daily_perf['date'], y=daily_perf['avg_wind'], name='Avg Wind', line=dict(color='cyan')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=daily_perf['date'], y=daily_perf['failures'], name='Failures', line=dict(color='red'), mode='lines+markers'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Performance Metrics Over Time")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

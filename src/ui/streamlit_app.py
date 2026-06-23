"""
Streamlit UI for Vietnamese LPR System
Provides web interface for real-time monitoring and history
"""

import streamlit as st
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modules.database_manager import DatabaseManager, decode_base64_to_image
from src.modules.rule_engine import get_province_from_plate

# Page configuration
st.set_page_config(
    page_title="Vietnamese LPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .plate-display {
        font-size: 2rem;
        font-weight: bold;
        font-family: monospace;
        background-color: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
    }
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    .status-inactive {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state"""
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'selected_camera' not in st.session_state:
        st.session_state.selected_camera = "cam_01"


def show_sidebar():
    """Render sidebar navigation"""
    st.sidebar.markdown("## 🚗 Vietnamese LPR")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Live Detection", "📋 History", "📈 Statistics"],
        index=0,
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    
    min_confidence = st.sidebar.slider(
        "Min Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
    )
    
    show_plates = st.sidebar.checkbox("Show Plate Thumbnails", value=True)
    
    return page, min_confidence, show_plates


def show_dashboard():
    """Show main dashboard"""
    st.markdown('<p class="main-header">Vietnamese License Plate Recognition</p>', unsafe_allow_html=True)
    
    # Load stats
    stats = st.session_state.db.get_statistics()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Recognitions",
            f"{stats.get('total_recognitions', 0):,}",
            delta="Last 24h: " + str(stats.get('last_24h', 0)),
        )
    
    with col2:
        st.metric(
            "Unique Plates",
            f"{stats.get('unique_plates', 0):,}",
        )
    
    with col3:
        avg_conf = stats.get('avg_confidence', 0)
        st.metric(
            "Avg Confidence",
            f"{avg_conf:.1%}" if avg_conf else "N/A",
        )
    
    with col4:
        plate_types = stats.get('by_plate_type', {})
        if plate_types:
            most_common = max(plate_types.items(), key=lambda x: x[1])
            st.metric("Most Common Type", most_common[0], delta=f"{most_common[1]} records")
        else:
            st.metric("Most Common Type", "N/A")
    
    st.markdown("---")
    
    # Recent recognitions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Recent Recognitions")
        recent = st.session_state.db.get_recent_recognitions(limit=20)
        
        if recent:
            # Convert to DataFrame
            df = pd.DataFrame(recent)
            df = df[['timestamp', 'plate_text', 'confidence', 'plate_type', 'province']]
            df.columns = ['Time', 'Plate', 'Confidence', 'Type', 'Province']
            df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No recognitions recorded yet. Start live detection to see results.")
    
    with col2:
        st.markdown("### Quick Stats")
        
        st.markdown("#### By Plate Type")
        plate_types = stats.get('by_plate_type', {})
        if plate_types:
            for ptype, count in sorted(plate_types.items(), key=lambda x: -x[1]):
                st.write(f"- **{ptype}**: {count}")
        else:
            st.write("No data")
        
        st.markdown("#### By Camera")
        cameras = stats.get('by_camera', {})
        if cameras:
            for cam, count in sorted(cameras.items(), key=lambda x: -x[1]):
                st.write(f"- **{cam}**: {count}")
        else:
            st.write("No data")


def show_live_detection(min_confidence: float):
    """Show live detection interface"""
    st.markdown("## 🔍 Live Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        camera_options = {
            "Camera 0 (Webcam)": 0,
            "Camera 1": 1,
            "Camera 2": 2,
        }
        
        selected = st.selectbox("Select Camera", list(camera_options.keys()))
        camera_id = camera_options[selected]
    
    with col2:
        st.markdown("### Camera Status")
        if st.session_state.camera_active:
            st.markdown('<p class="status-active">● ACTIVE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-inactive">○ INACTIVE</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Start/Stop buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Detection", type="primary", disabled=st.session_state.camera_active):
            st.session_state.camera_active = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Detection", disabled=not st.session_state.camera_active):
            st.session_state.camera_active = False
            st.rerun()
    
    with col3:
        if st.button("📸 Capture Frame"):
            st.info("Frame captured! (In real implementation, this would save the frame)")
    
    st.markdown("---")
    
    # Video display area
    st.markdown("### Live Feed")
    
    if st.session_state.camera_active:
        try:
            cap = cv2.VideoCapture(camera_id)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", width=600)
            else:
                st.error("Could not read from camera")
        except Exception as e:
            st.error(f"Camera error: {e}")
    else:
        # Placeholder
        st.image("https://via.placeholder.com/800x400?text=Camera+Feed+Preview", width=600)
        st.info("Click 'Start Detection' to begin live processing")
    
    # Results panel
    st.markdown("### Current Results")
    
    recent = st.session_state.db.get_recent_recognitions(limit=10)
    if recent:
        for rec in recent[:5]:
            if rec['confidence'] >= min_confidence:
                province = rec.get('province', '') or ''
                province_str = f" ({province})" if province else ""
                
                st.markdown(
                    f"<div class='plate-display'>{rec['plate_text']}{province_str}</div>",
                    unsafe_allow_html=True
                )
                st.write(f"Confidence: {rec['confidence']:.2f} | Type: {rec['plate_type']} | Time: {rec['timestamp']}")
                st.markdown("---")
    else:
        st.write("No results yet")


def show_history():
    """Show recognition history"""
    st.markdown("## 📋 Recognition History")
    
    # Search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_plate = st.text_input("Search by Plate", placeholder="e.g., 30A-1234")
    
    with col2:
        date_from = st.date_input("From Date", value=datetime.now() - timedelta(days=7))
    
    with col3:
        date_to = st.date_input("To Date", value=datetime.now())
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        plate_type_filter = st.multiselect(
            "Plate Type",
            ["private_car", "motorcycle", "police", "army", "commercial"],
        )
    
    with col2:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.1)
    
    # Fetch data
    if search_plate:
        records = st.session_state.db.search_by_plate(search_plate, limit=500)
    else:
        records = st.session_state.db.get_recent_recognitions(limit=500)
    
    # Apply filters
    filtered = []
    for rec in records:
        if rec['confidence'] < min_conf:
            continue
        if plate_type_filter and rec.get('plate_type', '') not in plate_type_filter:
            continue
        
        # Date filter
        try:
            rec_date = datetime.fromisoformat(rec['timestamp']).date()
            if rec_date < date_from or rec_date > date_to:
                continue
        except:
            pass
        
        filtered.append(rec)
    
    st.markdown(f"### Found {len(filtered)} records")
    
    # Display results
    if filtered:
        # Create DataFrame
        df = pd.DataFrame(filtered)
        display_cols = ['timestamp', 'plate_text', 'confidence', 'plate_type', 'province', 'camera_id']
        df = df[[c for c in display_cols if c in df.columns]]
        df.columns = ['Time', 'Plate', 'Confidence', 'Type', 'Province', 'Camera']
        df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export option
        if st.button("📥 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"lpr_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
    else:
        st.info("No records match your search criteria")


def show_statistics():
    """Show detailed statistics"""
    st.markdown("## 📈 Statistics & Analytics")
    
    stats = st.session_state.db.get_statistics()
    
    # Overview
    st.markdown("### Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", stats.get('total_recognitions', 0))
    with col2:
        st.metric("Unique Plates", stats.get('unique_plates', 0))
    with col3:
        st.metric("Last 24 Hours", stats.get('last_24h', 0))
    with col4:
        avg = stats.get('avg_confidence', 0)
        st.metric("Avg Confidence", f"{avg:.1%}" if avg else "N/A")
    
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### By Plate Type")
        plate_types = stats.get('by_plate_type', {})
        if plate_types:
            fig_data = pd.DataFrame({
                'Type': list(plate_types.keys()),
                'Count': list(plate_types.values()),
            })
            st.bar_chart(fig_data.set_index('Type'))
        else:
            st.write("No data available")
    
    with col2:
        st.markdown("### By Camera")
        cameras = stats.get('by_camera', {})
        if cameras:
            fig_data = pd.DataFrame({
                'Camera': list(cameras.keys()),
                'Count': list(cameras.values()),
            })
            st.bar_chart(fig_data.set_index('Camera'))
        else:
            st.write("No data available")
    
    st.markdown("---")
    
    # Database management
    st.markdown("### Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Delete records older than:"):
            days = st.number_input("Days", min_value=1, max_value=365, value=30)
            if st.button("Confirm Delete"):
                deleted = st.session_state.db.delete_old_records(days)
                st.success(f"Deleted {deleted} records")
                st.rerun()
    
    with col2:
        db_size = Path(st.session_state.db.db_path).stat().st_size / 1024 / 1024
        st.metric("Database Size", f"{db_size:.2f} MB")
        
        if st.button("📊 Refresh Statistics"):
            st.rerun()


def main():
    """Main application"""
    init_session_state()
    
    page, min_conf, show_thumbs = show_sidebar()
    
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🔍 Live Detection":
        show_live_detection(min_conf)
    elif page == "📋 History":
        show_history()
    elif page == "📈 Statistics":
        show_statistics()


if __name__ == "__main__":
    main()

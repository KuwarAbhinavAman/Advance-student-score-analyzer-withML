import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
from streamlit.components.v1 import html
from uuid import uuid4

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ======================
# DATA VALIDATION & PREPROCESSING
# ======================

REQUIRED_COLUMNS = {
    'Student_Name': 'object',
    'Previous Scores': 'object',
    'Current Percentage': 'float64',
    'Attendance': 'float64',
    'Reading_Score': 'float64',
    'Writing_Score': 'float64',
    'Grade': 'category'
}

def safe_convert_to_list(value):
    """
    Safely convert string or list to a list of floats.

    Args:
        value: Input value (string or list).

    Returns:
        list: List of floats, or [0.0] if conversion fails.
    """
    try:
        if isinstance(value, str):
            cleaned = value.strip('[]').split(',')
            result = [float(x.strip()) for x in cleaned if x.strip()]
            if not result:
                raise ValueError("Empty list")
            return result
        elif isinstance(value, list):
            result = [float(x) for x in value]
            if not result:
                raise ValueError("Empty list")
            return result
        raise ValueError("Invalid input")
    except (ValueError, TypeError):
        st.warning(f"Invalid Previous Scores format: {value}. Using default [0.0].")
        return [0.0]

def preprocess_data(data):
    """
    Preprocess student data by creating additional features.

    Args:
        data (pd.DataFrame): Input DataFrame with student data.

    Returns:
        pd.DataFrame: Processed DataFrame with new features.
    """
    # Work on a copy to avoid SettingWithCopyWarning
    data = data.copy()
    
    # Use .loc for safe assignment
    data.loc[:, 'Previous_Avg'] = data['Previous Scores'].apply(lambda x: np.mean(x))
    data.loc[:, 'Previous_Max'] = data['Previous Scores'].apply(lambda x: max(x))
    data.loc[:, 'Previous_Min'] = data['Previous Scores'].apply(lambda x: min(x))
    data.loc[:, 'Previous_Std'] = data['Previous Scores'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    data.loc[:, 'Score_Trend'] = data['Previous Scores'].apply(lambda x: calculate_trend(x))
    data.loc[:, 'Literacy_Score'] = (data['Reading_Score'] + data['Writing_Score']) / 2
    grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0}
    data.loc[:, 'Grade_Num'] = data['Grade'].map(grade_map)
    data.loc[:, 'Performance_Category'] = pd.cut(data['Current Percentage'],
                                                bins=[0, 60, 75, 85, 100],
                                                labels=['Low', 'Medium', 'Good', 'Excellent'])
    return data

@st.cache_data
def load_and_validate_data(_file):
    """
    Load and validate student data from a CSV file.

    Args:
        _file: Uploaded file object or file path.

    Returns:
        pd.DataFrame or None: Validated DataFrame or None if validation fails.
    """
    try:
        if isinstance(_file, str):
            data = pd.read_csv(_file)
        else:
            data = pd.read_csv(_file)
        
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns in CSV: {missing_cols}")
            return None
        
        for col, dtype in REQUIRED_COLUMNS.items():
            if col == 'Previous Scores':
                data.loc[:, col] = data[col].apply(safe_convert_to_list)
            else:
                data.loc[:, col] = data[col].astype(dtype, errors='ignore')
        
        # Explicitly create a copy after dropna to avoid SettingWithCopyWarning
        data = data.dropna().copy()
        data = preprocess_data(data)
        
        if data.empty:
            st.error("No valid data available after preprocessing.")
            return None
        
        return data
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None

def save_new_student(data, new_student_data, output_file='student_data.csv'):
    """
    Save new student data to a CSV file.

    Args:
        data (pd.DataFrame): Existing student data.
        new_student_data (dict): New student data.
        output_file (str): Path to save the CSV.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        new_student_df = pd.DataFrame([new_student_data])
        if data is not None:
            updated_data = pd.concat([data, new_student_df], ignore_index=True)
        else:
            updated_data = new_student_df
        updated_data.to_csv(output_file, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving student data: {str(e)}")
        return False

# ======================
# MACHINE LEARNING MODEL
# ======================

@st.cache_data
def train_prediction_model(_data):
    """
    Train a RandomForestRegressor model on student data.

    Args:
        _data (pd.DataFrame): Preprocessed student data.

    Returns:
        tuple: (model, scaler, features, mae, r2, feature_importance)
    """
    if _data.empty:
        st.error("No valid data available for training.")
        return None, None, None, None, None, None
    
    try:
        features = ['Previous_Avg', 'Previous_Max', 'Previous_Min', 'Previous_Std', 
                   'Score_Trend', 'Attendance', 'Reading_Score', 'Writing_Score', 
                   'Literacy_Score', 'Grade_Num']
        X = _data[features]
        y = _data['Current Percentage']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return model, scaler, features, mae, r2, feature_importance
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, None, None, None, None, None

# ======================
# TREND ANALYSIS & PREDICTION
# ======================

def calculate_trend(scores):
    """
    Calculate the trend (slope) of scores over time.

    Args:
        scores (list): List of historical scores.

    Returns:
        float: Slope of the trend line or 0.0 if invalid.
    """
    if len(scores) < 2:
        return 0.0
    x = np.arange(len(scores)).reshape(-1, 1)
    y = np.array(scores)
    try:
        return LinearRegression().fit(x, y).coef_[0]
    except:
        return 0.0

def predict_future_score(model, scaler, features, student_data, current_perc):
    """
    Predict future score using the trained model.

    Args:
        model: Trained RandomForestRegressor.
        scaler: Fitted StandardScaler.
        features (list): List of feature names.
        student_data (dict): Student data.
        current_perc (float): Current percentage.

    Returns:
        float: Predicted score, clipped between 0 and 100.
    """
    try:
        student_features = {
            'Previous_Avg': np.mean(student_data.get('Previous Scores', [])),
            'Previous_Max': max(student_data.get('Previous Scores', [0])),
            'Previous_Min': min(student_data.get('Previous Scores', [100])),
            'Previous_Std': np.std(student_data.get('Previous Scores', [0])),
            'Score_Trend': calculate_trend(student_data.get('Previous Scores', [])),
            'Attendance': student_data.get('Attendance', 0),
            'Reading_Score': student_data.get('Reading_Score', 0),
            'Writing_Score': student_data.get('Writing_Score', 0),
            'Literacy_Score': (student_data.get('Reading_Score', 0) + student_data.get('Writing_Score', 0)) / 2,
            'Grade_Num': {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0}.get(student_data.get('Grade', 'E'), 0)
        }
        
        X_pred = pd.DataFrame([student_features])[features]
        X_pred_scaled = scaler.transform(X_pred)
        prediction = model.predict(X_pred_scaled)[0]
        return max(0, min(100, prediction))
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return current_perc

# ======================
# VISUALIZATION FUNCTIONS
# ======================

def plot_performance_trend(student_name, historical, current, prediction):
    """
    Create an interactive performance trend plot.

    Args:
        student_name (str): Name of the student.
        historical (list): Historical scores.
        current (float): Current score.
        prediction (float): Predicted score.

    Returns:
        go.Figure: Plotly figure object.
    """
    history = historical + [current]
    terms = [f"Term {i+1}" for i in range(len(history))] + ["Next Term"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=terms[:-1],
        y=history,
        mode='lines+markers',
        name='Historical Performance',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10, color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=terms[-2:],
        y=[current, prediction],
        mode='lines+markers',
        name='Current & Prediction',
        line=dict(color='#ff7f0e', width=3, dash='dot'),
        marker=dict(size=12, color='#ff7f0e', symbol='diamond')
    ))
    fig.add_hline(y=85, line_dash="dash", line_color="green", 
                 annotation_text="Target", annotation_position="bottom right")
    
    fig.update_layout(
        title=f'Performance Trend for {student_name}',
        xaxis_title='Term',
        yaxis_title='Score (%)',
        yaxis_range=[0, 110],
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    return fig

def plot_attendance_impact(data, current_attendance, current_percentage):
    """
    Plot attendance impact on performance.

    Args:
        data (pd.DataFrame): Student data.
        current_attendance (float): Current student's attendance.
        current_percentage (float): Current student's percentage.

    Returns:
        px.Figure: Plotly figure object.
    """
    fig = px.scatter(data, x='Attendance', y='Current Percentage',
                    color='Performance_Category',
                    title='Attendance Impact on Performance',
                    labels={'Current Percentage': 'Score (%)', 'Attendance': 'Attendance (%)'},
                    hover_name='Student_Name')
    fig.add_trace(go.Scatter(
        x=[current_attendance],
        y=[current_percentage],
        mode='markers',
        marker=dict(size=15, color='red', symbol='x'),
        name='Your Attendance'
    ))
    fig.update_layout(height=500, template='plotly_white')
    return fig

def plot_literacy_impact(data, current_literacy, current_percentage):
    """
    Plot literacy impact on performance.

    Args:
        data (pd.DataFrame): Student data.
        current_literacy (float): Current student's literacy score.
        current_percentage (float): Current student's percentage.

    Returns:
        px.Figure: Plotly figure object.
    """
    fig = px.scatter(data, x='Literacy_Score', y='Current Percentage',
                    color='Grade',
                    title='Literacy Skills Impact on Performance',
                    labels={'Current Percentage': 'Score (%)', 'Literacy_Score': 'Literacy Score (0-10)'},
                    hover_name='Student_Name')
    fig.add_trace(go.Scatter(
        x=[current_literacy],
        y=[current_percentage],
        mode='markers',
        marker=dict(size=15, color='red', symbol='x'),
        name='Your Literacy'
    ))
    fig.update_layout(height=500, template='plotly_white')
    return fig

def plot_feature_importance(feature_importance):
    """
    Plot feature importance from the model.

    Args:
        feature_importance (pd.DataFrame): DataFrame with feature importance.

    Returns:
        px.Figure: Plotly figure object.
    """
    fig = px.bar(feature_importance, x='Importance', y='Feature',
                orientation='h', title='Feature Importance in Predictions',
                color='Importance', color_continuous_scale='Blues')
    fig.update_layout(height=500, template='plotly_white', coloraxis_showscale=False)
    return fig

def styled_html(content, bg_color="#e6f3ff", border_color="#1e88e5"):
    """
    Create styled HTML content for recommendations.

    Args:
        content (str): HTML content.
        bg_color (str): Background color.
        border_color (str): Border color.

    Returns:
        str: Styled HTML string.
    """
    return f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0;
        min-height: 100px;
        overflow: visible;
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease;
    ">
        {content}
    </div>
    """

def display_recommendations(attendance, literacy, trend, prediction, current_perc):
    """
    Display personalized recommendations based on student data.

    Args:
        attendance (float): Student's attendance percentage.
        literacy (float): Student's literacy score.
        trend (float): Performance trend.
        prediction (float): Predicted score.
        current_perc (float): Current percentage.

    Returns:
        list: List of recommendation strings for export.
    """
    st.subheader("🎯 Personalized Recommendations")
    recommendations = []
    
    if attendance < 70:
        content = """
        <h4 style="color: #d32f2f;">🚨 Critical Attendance</h4>
        <p>Below 70% attendance will significantly impact performance. Consider:</p>
        <ul>
            <li>Meeting with student to understand barriers</li>
            <li>Implementing attendance improvement plan</li>
            <li>Connecting with parents/guardians</li>
        </ul>
        """
        html(styled_html(content, bg_color="#ffebee", border_color="#d32f2f"))
        recommendations.append("Critical Attendance: Implement improvement plan.")
    elif attendance < 85:
        content = """
        <h4 style="color: #ffa000;">📉 Attendance Warning</h4>
        <p>Below optimal attendance level (85%). Suggestions:</p>
        <ul>
            <li>Monitor attendance closely</li>
            <li>Provide incentives for improved attendance</li>
            <li>Check for patterns in absences</li>
        </ul>
        """
        html(styled_html(content, bg_color="#fff3e0", border_color="#ffa000"))
        recommendations.append("Attendance Warning: Monitor and incentivize.")
    else:
        content = """
        <h4 style="color: #388e3c;">✅ Excellent Attendance</h4>
        <p>Maintaining good attendance. To maintain this:</p>
        <ul>
            <li>Recognize good attendance</li>
            <li>Continue positive reinforcement</li>
            <li>Monitor for any sudden changes</li>
        </ul>
        """
        html(styled_html(content, bg_color="#e8f5e9", border_color="#388e3c"))
        recommendations.append("Excellent Attendance: Maintain with reinforcement.")
    
    if literacy < 5:
        content = """
        <h4 style="color: #d32f2f;">📚 Literacy Deficiency</h4>
        <p>Immediate remediation needed in reading/writing skills:</p>
        <ul>
            <li>Implement targeted literacy interventions</li>
            <li>Provide additional reading materials</li>
            <li>Consider specialized tutoring</li>
        </ul>
        """
        html(styled_html(content, bg_color="#ffebee", border_color="#d32f2f"))
        recommendations.append("Literacy Deficiency: Implement interventions.")
    elif literacy < 7:
        content = """
        <h4 style="color: #ffa000;">📖 Literacy Improvement</h4>
        <p>Focus on language skills development:</p>
        <ul>
            <li>Incorporate daily reading exercises</li>
            <li>Provide writing prompts</li>
            <li>Encourage journaling</li>
        </ul>
        """
        html(styled_html(content, bg_color="#fff3e0", border_color="#ffa000"))
        recommendations.append("Literacy Improvement: Focus on reading and writing.")
    else:
        content = """
        <h4 style="color: #388e3c;">🎯 Strong Literacy</h4>
        <p>Solid foundation in reading/writing. To enhance further:</p>
        <ul>
            <li>Challenge with advanced materials</li>
            <li>Encourage creative writing</li>
            <li>Introduce critical analysis exercises</li>
        </ul>
        """
        html(styled_html(content, bg_color="#e8f5e9", border_color="#388e3c"))
        recommendations.append("Strong Literacy: Enhance with advanced materials.")
    
    if trend > 7:
        content = f"""
        <h4 style="color: #388e3c;">🌟 Outstanding Progress!</h4>
        <p>+{trend:.1f}% improvement from historical average. To maintain momentum:</p>
        <ul>
            <li>Provide challenging material</li>
            <li>Recognize achievements</li>
            <li>Set stretch goals</li>
        </ul>
        """
        html(styled_html(content, bg_color="#e8f5e9", border_color="#388e3c"))
        recommendations.append(f"Outstanding Progress (+{trend:.1f}%): Maintain momentum.")
        st.balloons()
    elif trend > 3:
        content = f"""
        <h4 style="color: #388e3c;">📈 Positive Trend</h4>
        <p>+{trend:.1f}% improvement from historical average. Suggestions:</p>
        <ul>
            <li>Identify what's working well</li>
            <li>Reinforce successful strategies</li>
            <li>Set incremental goals</li>
        </ul>
        """
        html(styled_html(content, bg_color="#e8f5e9", border_color="#388e3c"))
        recommendations.append(f"Positive Trend (+{trend:.1f}%): Reinforce strategies.")
    elif trend < -5:
        content = f"""
        <h4 style="color: #d32f2f;">📉 Concerning Trend</h4>
        <p>{abs(trend):.1f}% drop from historical average. Immediate actions:</p>
        <ul>
            <li>Review recent assessments</li>
            <li>Schedule student conference</li>
            <li>Implement targeted support</li>
        </ul>
        """
        html(styled_html(content, bg_color="#ffebee", border_color="#d32f2f"))
        recommendations.append(f"Concerning Trend (-{abs(trend):.1f}%): Implement support.")
    elif trend < 0:
        content = f"""
        <h4 style="color: #ffa000;">⚠️ Slight Decline</h4>
        <p>{abs(trend):.1f}% drop from historical average. Considerations:</p>
        <ul>
            <li>Monitor closely</li>
            <li>Check for external factors</li>
            <li>Provide additional support</li>
        </ul>
        """
        html(styled_html(content, bg_color="#fff3e0", border_color="#ffa000"))
        recommendations.append(f"Slight Decline (-{abs(trend):.1f}%): Monitor and support.")
    else:
        content = """
        <h4 style="color: #1976d2;">🔄 Consistent Performance</h4>
        <p>Maintaining similar performance level. To improve:</p>
        <ul>
            <li>Identify areas for growth</li>
            <li>Set specific improvement goals</li>
            <li>Provide targeted challenges</li>
        </ul>
        """
        html(styled_html(content, bg_color="#e3f2fd", border_color="#1976d2"))
        recommendations.append("Consistent Performance: Set improvement goals.")
    
    st.subheader("📝 Action Plan")
    if prediction < current_perc:
        content = """
        <h4 style="color: #ffa000;">Performance Decline Predicted</h4>
        <p>Consider these targeted interventions:</p>
        <ul>
            <li><strong>Academic Support:</strong> Schedule weekly check-ins to review material</li>
            <li><strong>Study Skills:</strong> Provide training on effective study techniques</li>
            <li><strong>Parent Communication:</strong> Inform parents of concerns and plan</li>
            <li><strong>Progress Monitoring:</strong> Implement bi-weekly progress assessments</li>
        </ul>
        """
        html(styled_html(content, bg_color="#fff3e0", border_color="#ffa000"))
        recommendations.append("Performance Decline Predicted: Implement interventions.")
    else:
        content = """
        <h4 style="color: #388e3c;">Performance Improvement Predicted</h4>
        <p>Strategies to maintain positive trajectory:</p>
        <ul>
            <li><strong>Challenge:</strong> Introduce advanced material in strong areas</li>
            <li><strong>Goal Setting:</strong> Work with student to set ambitious but achievable goals</li>
            <li><strong>Positive Reinforcement:</strong> Recognize and reward progress</li>
            <li><strong>Peer Support:</strong> Consider peer tutoring opportunities</li>
        </ul>
        """
        html(styled_html(content, bg_color="#e8f5e9", border_color="#388e3c"))
        recommendations.append("Performance Improvement Predicted: Maintain trajectory.")
    
    return recommendations

# ======================
# REPORT EXPORT
# ======================

def export_report(student_name, current_perc, prediction, trend, literacy, recommendations):
    """
    Export analysis report as a CSV.

    Args:
        student_name (str): Student name.
        current_perc (float): Current percentage.
        prediction (float): Predicted score.
        trend (float): Performance trend.
        literacy (float): Literacy score.
        recommendations (list): List of recommendation strings.

    Returns:
        pd.DataFrame: DataFrame for export.
    """
    report_data = {
        'Student_Name': [student_name],
        'Current_Percentage': [current_perc],
        'Predicted_Score': [prediction],
        'Trend': [trend],
        'Literacy_Score': [literacy],
        'Recommendations': ['; '.join(recommendations)]
    }
    return pd.DataFrame(report_data)

# ======================
# ANALYSIS DISPLAY FUNCTION
# ======================

def display_analysis(student_db, analysis):
    """
    Display the analysis results for a student.

    Args:
        student_db (pd.DataFrame): Student database.
        analysis (dict): Analysis results containing student_name, historical, current_perc, etc.
    """
    student_name = analysis['student_name']
    historical = analysis['historical']
    current_perc = analysis['current_perc']
    prediction = analysis['prediction']
    attendance = analysis['attendance']
    literacy = analysis['literacy']
    trend = analysis['trend']
    
    st.subheader("📈 Performance Analysis Report")
    st.plotly_chart(plot_performance_trend(
        student_name, historical, current_perc, prediction
    ), use_container_width=True)
    
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Score</h4>
            <h2 style="color: #1f77b4;">{current_perc:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Predicted Score</h4>
            <h2 style="color: #ff7f0e;">{prediction:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        trend_color = "green" if trend >= 0 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Trend</h4>
            <h2 style="color: {trend_color};">{trend:+.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Literacy</h4>
            <h2 style="color: #2ca02c;">{literacy:.1f}/10</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("📊 Comparative Analysis")
    tab1, tab2, tab3 = st.tabs(["Attendance Impact", "Literacy Impact", "Feature Importance"])
    with tab1:
        st.plotly_chart(plot_attendance_impact(
            student_db, attendance, current_perc
        ), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_literacy_impact(
            student_db, literacy, current_perc
        ), use_container_width=True)
    with tab3:
        if st.session_state.feature_importance is not None:
            st.plotly_chart(plot_feature_importance(
                st.session_state.feature_importance
            ), use_container_width=True)
        else:
            st.warning("Feature importance data not available")
    
    recommendations = display_recommendations(attendance, literacy, trend, prediction, current_perc)
    
    report_df = export_report(
        student_name, current_perc, prediction, trend, literacy, recommendations
    )
    st.download_button(
        label="📥 Download Report",
        data=report_df.to_csv(index=False),
        file_name=f"{student_name}_report.csv",
        mime="text/csv"
    )

# ======================
# STREAMLIT UI
# ======================

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="🎓 Student Performance Predictor",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #ccc;
    }
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: 8px 8px 0 0;
        transition: background-color 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Advanced Student Performance Predictor")
    
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.features = None
        st.session_state.student_db = None
        st.session_state.feature_importance = None
        st.session_state.mae = None
        st.session_state.r2 = None
        st.session_state.new_student_added = False
        st.session_state.output_file = 'student_data.csv'
        st.session_state.analysis_results = None
        st.session_state.new_student_analysis = None
        st.session_state.form_submitted = False
        st.session_state.new_student_form_submitted = False
    
    with st.sidebar:
        st.header("📂 Data Management")
        uploaded_file = st.file_uploader("Upload Student Database (CSV)", type="csv")
        st.session_state.output_file = st.text_input("Output CSV File", value="student_data.csv")
        
        if uploaded_file:
            with st.spinner("Loading and validating data..."):
                student_db = load_and_validate_data(uploaded_file)
            if student_db is not None:
                st.session_state.student_db = student_db
                with st.spinner("Training prediction model..."):
                    (st.session_state.model, st.session_state.scaler, 
                     st.session_state.features, st.session_state.mae,
                     st.session_state.r2, st.session_state.feature_importance) = train_prediction_model(student_db)
                st.success("Model trained successfully!")
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Model Performance</h4>
                    <p>Mean Absolute Error: <strong>{st.session_state.mae:.2f}%</strong></p>
                    <p>R² Score: <strong>{st.session_state.r2:.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This tool predicts student performance using machine learning, offering:
        - Performance trend analysis
        - Predictive insights
        - Personalized recommendations
        - Interactive visualizations
        - Report export functionality
        """)
    
    if st.session_state.student_db is not None:
        student_db = st.session_state.student_db
        student_names = student_db['Student_Name'].unique().tolist()
        student_names.append("New Student")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.header("👨‍🎓 Student Selection")
            selected_student = st.selectbox("Select Student", student_names, key="student_select")
            
            if selected_student != "New Student":
                student_data = student_db[student_db['Student_Name'] == selected_student].iloc[0].to_dict()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{selected_student}</h3>
                    <p><strong>Current Grade:</strong> {student_data['Grade']}</p>
                    <p><strong>Average Score:</strong> {student_data['Previous_Avg']:.1f}%</p>
                    <p><strong>Attendance:</strong> {student_data['Attendance']}%</p>
                    <p><strong>Literacy Score:</strong> {(student_data['Reading_Score'] + student_data['Writing_Score'])/2:.1f}/10</p>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("📋 View Historical Scores"):
                    st.write(pd.DataFrame({
                        'Term': [f"Term {i+1}" for i in range(len(student_data['Previous Scores']))],
                        'Score': student_data['Previous Scores']
                    }))
            else:
                student_data = None
        
        with col2:
            if selected_student == "New Student":
                st.header("➕ Add New Student")
                with st.form("new_student_form"):
                    new_name = st.text_input("Student Name", "New Student")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        attendance = st.number_input("Attendance (%)", 0.0, 100.0, 85.0)
                    with c2:
                        reading_score = st.number_input("Reading Score (0-10)", 0.0, 10.0, 7.0)
                    with c3:
                        writing_score = st.number_input("Writing Score (0-10)", 0.0, 10.0, 7.0)
                    
                    prev_scores = st.text_input("Previous Scores (comma separated)", "70, 75, 80")
                    try:
                        prev_scores = [float(x.strip()) for x in prev_scores.split(",") if x.strip()]
                        if not prev_scores:
                            raise ValueError("Empty scores")
                    except:
                        st.error("Invalid Previous Scores format. Using default [70, 75, 80].")
                        prev_scores = [70, 75, 80]
                    
                    current_perc = st.number_input("Current Term Percentage (%)", 0.0, 100.0, 80.0)
                    grade = st.selectbox("Current Grade", ['A', 'B', 'C', 'D', 'E'], index=1)
                    
                    submitted = st.form_submit_button("Save & Analyze")
                    if submitted:
                        if not new_name.strip():
                            st.error("Student name cannot be empty.")
                        else:
                            new_student_data = {
                                'Student_Name': new_name.strip(),
                                'Previous Scores': prev_scores,
                                'Current Percentage': current_perc,
                                'Attendance': attendance,
                                'Reading_Score': reading_score,
                                'Writing_Score': writing_score,
                                'Grade': grade
                            }
                            with st.spinner("Saving student data..."):
                                if save_new_student(student_db, new_student_data, st.session_state.output_file):
                                    st.success("Student data saved successfully!")
                                    st.session_state.new_student_added = True
                                    student_db = load_and_validate_data(st.session_state.output_file)
                                    if student_db is not None:
                                        st.session_state.student_db = student_db
                                        with st.spinner("Retraining model..."):
                                            (st.session_state.model, st.session_state.scaler, 
                                             st.session_state.features, st.session_state.mae,
                                             st.session_state.r2, st.session_state.feature_importance) = train_prediction_model(student_db)
                                    # Perform analysis and store results
                                    historical = new_student_data['Previous Scores']
                                    current_perc = new_student_data['Current Percentage']
                                    attendance = new_student_data['Attendance']
                                    reading = new_student_data['Reading_Score']
                                    writing = new_student_data['Writing_Score']
                                    literacy = (reading + writing) / 2
                                    if st.session_state.model:
                                        prediction = predict_future_score(
                                            st.session_state.model,
                                            st.session_state.scaler,
                                            st.session_state.features,
                                            new_student_data,
                                            current_perc
                                        )
                                    else:
                                        prediction = current_perc * 0.9 + np.mean(historical) * 0.1 if historical else current_perc
                                    trend = current_perc - np.mean(historical) if historical else 0
                                    st.session_state.new_student_analysis = {
                                        'student_name': new_name.strip(),
                                        'historical': historical,
                                        'current_perc': current_perc,
                                        'prediction': prediction,
                                        'attendance': attendance,
                                        'literacy': literacy,
                                        'trend': trend
                                    }
                                    st.session_state.new_student_form_submitted = True
                
                # Display analysis results if form was submitted
                if st.session_state.new_student_form_submitted and st.session_state.new_student_analysis:
                    display_analysis(student_db, st.session_state.new_student_analysis)
            
            else:
                st.header("📊 Performance Analysis")
                with st.form("prediction_form"):
                    current_perc = st.number_input(
                        "Current Term Percentage (%)", 0.0, 100.0,
                        value=student_data['Current Percentage'] if student_data else 85.0
                    )
                    submitted = st.form_submit_button("Analyze Performance")
                    
                    if submitted:
                        try:
                            historical = student_data.get('Previous Scores', [])
                            attendance = student_data.get('Attendance', 0)
                            reading = student_data.get('Reading_Score', 0)
                            writing = student_data.get('Writing_Score', 0)
                            literacy = (reading + writing) / 2
                            if st.session_state.model:
                                prediction = predict_future_score(
                                    st.session_state.model,
                                    st.session_state.scaler,
                                    st.session_state.features,
                                    student_data,
                                    current_perc
                                )
                            else:
                                prediction = current_perc * 0.9 + np.mean(historical) * 0.1 if historical else current_perc
                            trend = current_perc - np.mean(historical) if historical else 0
                            st.session_state.analysis_results = {
                                'student_name': selected_student,
                                'historical': historical,
                                'current_perc': current_perc,
                                'prediction': prediction,
                                'attendance': attendance,
                                'literacy': literacy,
                                'trend': trend
                            }
                            st.session_state.form_submitted = True
                        except Exception as e:
                            st.error(f"Analysis error: {str(e)}")
                
                # Display analysis results if form was submitted
                if st.session_state.form_submitted and st.session_state.analysis_results:
                    display_analysis(student_db, st.session_state.analysis_results)
    else:
        st.info("""
        ## Welcome to the Student Performance Predictor
        
        To get started:
        1. Upload a CSV file with student records using the sidebar
        2. Specify an output CSV file name
        3. Select a student or add a new one
        4. Analyze performance and download reports
        
        The system will automatically analyze the data and provide insights.
        """)
        with st.expander("💡 Sample CSV Structure"):
            st.markdown("""
            Your CSV should contain these columns (sample data shown):
            
            | Student_Name | Previous Scores | Current Percentage | Attendance | Reading_Score | Writing_Score | Grade |
            |--------------|------------------|---------------------|------------|---------------|---------------|-------|
            | John Doe     | [75, 80, 78]     | 82.5                | 92.0       | 7.5           | 8.0           | B     |
            | Jane Smith   | [85, 88, 90]     | 91.0                | 98.0       | 9.0           | 9.5           | A     |
            
            **Note:** Previous Scores should be formatted as a list in square brackets or comma-separated values.
            """)

if __name__ == "__main__":
    main()

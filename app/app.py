import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
import time

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DefectForge AI - Steel Defect Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── JetBrains Mono font + Professional theme ───────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── Font stack ─────────────────────────────────────────────────── */
    :root {
        --font-main: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
        --navy-900: #0d1b2a;
        --navy-800: #1b2a4a;
        --navy-700: #1f3461;
        --navy-600: #274479;
        --accent: #2563eb;
        --accent-light: #3b82f6;
        --green: #16a34a;
        --amber: #d97706;
        --red: #dc2626;
        --text-primary: #111827;
        --text-secondary: #4b5563;
        --text-muted: #6b7280;
        --border: #e5e7eb;
        --card-bg: #f9fafb;
    }

    /* Force font everywhere */
    html, body, .stApp, [class*="css"], [class*="emotion"] {
        font-family: var(--font-main) !important;
    }

    .stApp * {
        font-family: var(--font-main) !important;
    }

    /* ── Main area: white background ───────────────────────────────── */
    .stApp {
        background-color: #ffffff !important;
    }

    section.main { background-color: #ffffff !important; }

    div[data-testid="stMainBlock"] {
        background-color: #ffffff !important;
    }

    div[data-testid="stMainBlockContainer"] {
        background-color: #ffffff !important;
    }

    /* ── Sidebar: dark navy blue ────────────────────────────────────── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    div[data-testid="stSidebarContent"],
    div[data-baseweb="sidebar"] {
        background: var(--navy-900) !important;
        background-color: var(--navy-900) !important;
    }

    section[data-testid="stSidebar"] * {
        background-color: transparent !important;
    }

    /* Sidebar text colors */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:hover {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:hover span {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] div:first-child {
        border-color: #60a5fa !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* ── Headings: black text ───────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: var(--font-main) !important;
    }

    /* ── Metric cards ───────────────────────────────────────────────── */
    div[data-testid="stMetric"],
    div[data-testid="stMetric"] > div {
        background: var(--card-bg) !important;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label * {
        color: var(--text-muted) !important;
        font-size: 0.8em;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] * {
        color: var(--navy-800) !important;
        font-weight: 700;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricDelta"],
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] * {
        color: var(--green) !important;
    }

    /* ── Cards ──────────────────────────────────────────────────────── */
    .custom-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .custom-card h3 {
        color: var(--navy-800) !important;
        margin: 0 0 8px 0;
        font-size: 1.05em;
        font-weight: 600;
    }

    .custom-card p, .custom-card li {
        color: var(--text-secondary) !important;
        margin: 2px 0;
        font-size: 0.9em;
        line-height: 1.6;
    }

    /* ── Prediction result ──────────────────────────────────────────── */
    .prediction-box {
        background: #f0fdf4;
        border: 2px solid var(--green);
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }

    .prediction-box h2 {
        color: var(--green) !important;
        margin: 0;
        font-size: 1.6em;
    }

    .prediction-box p {
        color: #166534 !important;
        font-size: 16px;
    }

    /* ── Severity badges ────────────────────────────────────────────── */
    .severity-low {
        background: var(--green);
        color: white;
        padding: 4px 14px;
        border-radius: 4px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.85em;
        letter-spacing: 0.5px;
    }

    .severity-medium {
        background: var(--amber);
        color: white;
        padding: 4px 14px;
        border-radius: 4px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.85em;
        letter-spacing: 0.5px;
    }

    .severity-high {
        background: var(--red);
        color: white;
        padding: 4px 14px;
        border-radius: 4px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.85em;
        letter-spacing: 0.5px;
    }

    /* ── Rejected prediction ────────────────────────────────────────── */
    .prediction-rejected {
        background: #fef2f2;
        border: 2px solid var(--red);
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }

    .prediction-rejected h2 {
        color: var(--red) !important;
        margin: 0;
        font-size: 1.4em;
    }

    .prediction-rejected p {
        color: #991b1b !important;
        font-size: 14px;
    }

    /* ── Sidebar logo ───────────────────────────────────────────────── */
    .logo-container {
        text-align: center;
        padding: 24px 0 20px 0;
        border-bottom: 1px solid #334155;
        margin-bottom: 20px;
    }

    .logo-container h1 {
        color: #ffffff !important;
        font-size: 1.5em;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .logo-container p {
        color: #64748b !important;
        margin: 4px 0 0 0;
        font-size: 0.8em;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── File uploader ──────────────────────────────────────────────── */
    .stFileUploader {
        border: 2px dashed var(--border) !important;
        border-radius: 8px !important;
    }

    /* ── Footer ─────────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: var(--text-muted);
        padding: 24px 0;
        border-top: 1px solid var(--border);
        margin-top: 48px;
        font-size: 0.8em;
    }

    /* ── Horizontal rules ───────────────────────────────────────────── */
    hr {
        border-color: var(--border) !important;
    }

    /* ── Streamlit info/warning boxes ───────────────────────────────── */
    .stAlert {
        font-family: var(--font-mono) !important;
    }

    /* ── Selectbox / input ──────────────────────────────────────────── */
    .stSelectbox div[data-baseweb="select"] {
        font-family: var(--font-mono) !important;
    }

    /* ── Sidebar system info ────────────────────────────────────────── */
    .sys-info {
        font-size: 0.8em;
        padding: 10px 0;
    }
    .sys-info p {
        color: #94a3b8 !important;
        margin: 4px 0;
    }
    .sys-info b {
        color: #60a5fa !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Model & classes ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SteelCNN

CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Confidence threshold -- below this the model is "unsure"
CONFIDENCE_THRESHOLD = 0.70  # 70%

CLASS_LABELS = {
    'crazing': 'Crazing',
    'inclusion': 'Inclusion',
    'patches': 'Patches',
    'pitted_surface': 'Pitted Surface',
    'rolled-in_scale': 'Rolled-in Scale',
    'scratches': 'Scratches',
}

SEVERITY = {
    'crazing': ('Medium', 'severity-medium'),
    'inclusion': ('High', 'severity-high'),
    'patches': ('Low', 'severity-low'),
    'pitted_surface': ('Medium', 'severity-medium'),
    'rolled-in_scale': ('Medium', 'severity-medium'),
    'scratches': ('Low', 'severity-low'),
}

DESCRIPTIONS = {
    'crazing': 'A network of fine cracks on the steel surface, often caused by stress or thermal cycling.',
    'inclusion': 'Non-metallic particles embedded in the steel, typically from slag or refractory material.',
    'patches': 'Localized surface irregularities appearing as discolored or uneven areas on the steel.',
    'pitted_surface': 'Small cavities or pits formed on the surface due to corrosion or chemical attack.',
    'rolled-in_scale': 'Oxide scale pressed into the steel surface during the rolling process.',
    'scratches': 'Linear grooves or marks caused by mechanical contact with tools or handling equipment.',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Chart color palette (professional) ──────────────────────────────────
CHART_NAVY = '#1f3461'
CHART_BLUE = '#2563eb'
CHART_GRAY = '#9ca3af'
CHART_GREEN = '#16a34a'
CHART_AMBER = '#d97706'
CHART_RED = '#dc2626'

def bar_color(v):
    if v >= 90:
        return CHART_GREEN
    elif v >= 70:
        return CHART_AMBER
    return CHART_RED


@st.cache_resource
def load_model():
    model = SteelCNN().to(device)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'steel_cnn.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        all_probs = probs.squeeze().cpu().numpy().tolist()
    pred_class = CLASSES[predicted.item()]
    conf = confidence.item()
    is_reliable = conf >= CONFIDENCE_THRESHOLD
    return pred_class, conf, all_probs, is_reliable


def render_prediction_result(predicted_class, confidence, all_probs, is_reliable):
    """Renders the prediction result card. Handles low-confidence rejection."""
    if not is_reliable:
        st.markdown(f"""
        <div class="prediction-rejected">
            <p style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Detection Rejected</p>
            <h2>Not a Recognized Steel Defect</h2>
            <p>The model confidence is only <b>{confidence*100:.2f}%</b> -- below the {CONFIDENCE_THRESHOLD*100:.0f}% threshold.</p>
            <p>This image may not be a steel surface, or the defect is too ambiguous to classify reliably.</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("Please upload a clear image of a steel surface with a visible defect.")
    else:
        sev, sev_class = SEVERITY[predicted_class]
        st.markdown(f"""
        <div class="prediction-box">
            <p style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Detected Defect</p>
            <h2>{CLASS_LABELS[predicted_class]}</h2>
            <p>Confidence: <b>{confidence*100:.2f}%</b></p>
            <span class="{sev_class}">Severity: {sev}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="custom-card">
            <h3>Description</h3>
            <p>{DESCRIPTIONS[predicted_class]}</p>
        </div>
        """, unsafe_allow_html=True)

    # Probability chart (always shown)
    st.markdown("#### Probability Distribution")
    sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)
    sorted_labels = [CLASS_LABELS[CLASSES[i]] for i in sorted_indices]
    sorted_probs = [all_probs[i] * 100 for i in sorted_indices]
    bar_colors = [CHART_BLUE if i == sorted_indices[0] else '#d1d5db' for i in range(len(sorted_indices))]

    fig = go.Figure(data=[
        go.Bar(
            x=sorted_probs,
            y=sorted_labels,
            orientation='h',
            marker=dict(color=bar_colors),
            text=[f"{p:.1f}%" for p in sorted_probs],
            textposition='outside',
            textfont=dict(color=CHART_NAVY, size=11, family='JetBrains Mono')
        )
    ])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4b5563', family='JetBrains Mono'),
        xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False),
        yaxis=dict(autorange='reversed', showgrid=False),
        margin=dict(l=0, r=60, t=10, b=10),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Load accuracy results ────────────────────────────────────────────────
accuracy_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'accuracy_results.json')
if os.path.exists(accuracy_path):
    with open(accuracy_path) as f:
        accuracy_data = json.load(f)
else:
    accuracy_data = {
        "overall_accuracy": 89.17,
        "total_images": 360,
        "correct_predictions": 321,
        "class_accuracy": {
            "crazing": 100.0, "inclusion": 60.0, "patches": 98.33,
            "pitted_surface": 91.67, "rolled-in_scale": 100.0, "scratches": 85.0
        }
    }

# ── Sidebar ──────────────────────────────────────────────────────────────
NAV_ITEMS = ["Dashboard", "Analyze Defect", "Model Performance", "About"]

with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <h1>DefectForge AI</h1>
        <p>Steel Defect Analyzer</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div class="sys-info">
        <p>Device: <b>{device}</b></p>
        <p>Classes: <b>6</b></p>
        <p>Architecture: <b>CNN</b></p>
        <p>Threshold: <b>{CONFIDENCE_THRESHOLD*100:.0f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

# ── Page: Dashboard ──────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown("# DefectForge AI -- Dashboard")
    st.markdown("Real-time steel surface defect detection powered by deep learning.")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy_data['overall_accuracy']}%", delta="Trained")
    with col2:
        st.metric("Validation Images", f"{accuracy_data['total_images']:,}")
    with col3:
        st.metric("Correct Predictions", f"{accuracy_data['correct_predictions']:,}")
    with col4:
        st.metric("Defect Classes", "6")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Per-Class Accuracy")
        class_acc = accuracy_data['class_accuracy']
        labels = [CLASS_LABELS.get(k, k) for k in class_acc.keys()]
        values = list(class_acc.values())
        colors = [bar_color(v) for v in values]

        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(color=colors, line=dict(color='#e5e7eb', width=1)),
                text=[f"{v}%" for v in values],
                textposition='outside',
                textfont=dict(color=CHART_NAVY, size=12, family='JetBrains Mono')
            )
        ])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4b5563', family='JetBrains Mono'),
            xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=60, t=20, b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Accuracy Distribution")
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textfont=dict(color=CHART_NAVY, size=11, family='JetBrains Mono'),
                hoverinfo='label+value'
            )
        ])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4b5563', family='JetBrains Mono'),
            margin=dict(l=0, r=0, t=20, b=20),
            height=350,
            annotations=[dict(text=f"<b>{accuracy_data['overall_accuracy']}%</b>",
                              x=0.5, y=0.5, font_size=22, font_color=CHART_NAVY,
                              showarrow=False, font=dict(family='JetBrains Mono'))]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("### Defect Class Reference")
    cols = st.columns(3)
    for i, cls in enumerate(CLASSES):
        with cols[i % 3]:
            sev, sev_class = SEVERITY[cls]
            st.markdown(f"""
            <div class="custom-card">
                <h3>{CLASS_LABELS[cls]}</h3>
                <p>{DESCRIPTIONS[cls]}</p>
                <br><span class="{sev_class}">Severity: {sev}</span>
            </div>
            """, unsafe_allow_html=True)


# ── Page: Analyze Defect ─────────────────────────────────────────────────
elif page == "Analyze Defect":
    st.markdown("# Analyze Steel Surface")
    st.markdown("Upload a steel surface image to detect defects in real-time.")
    st.markdown("---")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a steel surface image...",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_result:
        st.markdown("### Detection Result")

        if uploaded_file is not None:
            with st.spinner("Analyzing surface defect..."):
                time.sleep(0.5)
                predicted_class, confidence, all_probs, is_reliable = predict_image(image)

            render_prediction_result(predicted_class, confidence, all_probs, is_reliable)
        else:
            st.info("Upload an image to begin analysis.")

            st.markdown("---")
            st.markdown("#### Quick Test Images")
            st.markdown("Select a sample image from the test dataset:")

            sample_dirs = {
                'crazing': 'Crazing',
                'inclusion': 'Inclusion',
                'patches': 'Patches',
                'pitted_surface': 'Pitted Surface',
                'rolled-in_scale': 'Rolled-in Scale',
                'scratches': 'Scratches',
            }

            test_base = os.path.join(os.path.dirname(__file__), '..', 'data', 'test')
            selected_class = st.selectbox("Defect Class", list(sample_dirs.values()))
            cls_key = [k for k, v in sample_dirs.items() if v == selected_class][0]
            cls_dir = os.path.join(test_base, cls_key)

            if os.path.isdir(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    selected_file = st.selectbox("Sample Image", files[:10])
                    if st.button("Load & Analyze"):
                        img_path = os.path.join(cls_dir, selected_file)
                        image = Image.open(img_path).convert("RGB")
                        st.image(image, caption=f"Sample: {selected_file}", use_container_width=True)

                        predicted_class, confidence, all_probs, is_reliable = predict_image(image)
                        render_prediction_result(predicted_class, confidence, all_probs, is_reliable)


# ── Page: Model Performance ──────────────────────────────────────────────
elif page == "Model Performance":
    st.markdown("# Model Performance Report")
    st.markdown("Detailed evaluation metrics for the DefectForge AI model.")
    st.markdown("---")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy_data['overall_accuracy']}%")
    with col2:
        st.metric("Total Evaluated", f"{accuracy_data['total_images']}")
    with col3:
        st.metric("Correct", f"{accuracy_data['correct_predictions']}")
    with col4:
        st.metric("Misclassified", f"{accuracy_data['total_images'] - accuracy_data['correct_predictions']}")

    st.markdown("---")

    # Detailed charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Accuracy by Defect Class")
        class_acc = accuracy_data['class_accuracy']
        labels = [CLASS_LABELS.get(k, k) for k in class_acc.keys()]
        values = list(class_acc.values())
        colors = [bar_color(v) for v in values]

        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(color=colors, line=dict(color='#e5e7eb', width=1)),
                text=[f"{v}%" for v in values],
                textposition='outside',
                textfont=dict(color=CHART_NAVY, size=12, family='JetBrains Mono')
            )
        ])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4b5563', size=12, family='JetBrains Mono'),
            xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=80, t=20, b=20),
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Class Distribution (Validation)")
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.1)',
            line=dict(color=CHART_BLUE, width=2),
            marker=dict(color=CHART_BLUE, size=6),
            name='Accuracy %'
        ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(color='#6b7280', family='JetBrains Mono'),
                                gridcolor='#e5e7eb'),
                angularaxis=dict(tickfont=dict(color='#4b5563', size=11, family='JetBrains Mono'),
                                 gridcolor='#e5e7eb')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4b5563', family='JetBrains Mono'),
            margin=dict(l=40, r=40, t=30, b=30),
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Model Architecture Info
    st.markdown("### Model Architecture")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="custom-card">
            <h3>Architecture</h3>
            <p><b>Type:</b> Convolutional Neural Network</p>
            <p><b>Conv Layers:</b> 2 (16 + 32 filters)</p>
            <p><b>FC Layers:</b> 2 (128 + 6 neurons)</p>
            <p><b>Activation:</b> ReLU</p>
            <p><b>Pooling:</b> MaxPool 2x2</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="custom-card">
            <h3>Training Config</h3>
            <p><b>Optimizer:</b> Adam (lr=0.001)</p>
            <p><b>Loss:</b> Cross-Entropy</p>
            <p><b>Epochs:</b> 10</p>
            <p><b>Batch Size:</b> 8</p>
            <p><b>Input Size:</b> 224x224 RGB</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="custom-card">
            <h3>Dataset</h3>
            <p><b>Source:</b> NEU-DET</p>
            <p><b>Train Images:</b> 1,440</p>
            <p><b>Val Images:</b> 360</p>
            <p><b>Classes:</b> 6 defect types</p>
            <p><b>Framework:</b> PyTorch</p>
        </div>
        """, unsafe_allow_html=True)

    # Insights
    st.markdown("---")
    st.markdown("### Performance Insights")
    best_class = max(class_acc, key=class_acc.get)
    worst_class = min(class_acc, key=class_acc.get)
    st.markdown(f"""
    <div class="custom-card">
        <p><b>Best performing class:</b> {CLASS_LABELS[best_class]} at {class_acc[best_class]}% accuracy</p>
        <p><b>Needs improvement:</b> {CLASS_LABELS[worst_class]} at {class_acc[worst_class]}% accuracy</p>
        <p><b>Performance spread:</b> {max(class_acc.values()) - min(class_acc.values()):.2f}% between best and worst class</p>
        <p><b>Recommendation:</b> Consider data augmentation for the '{CLASS_LABELS[worst_class]}' class to improve balance</p>
    </div>
    """, unsafe_allow_html=True)


# ── Page: About ──────────────────────────────────────────────────────────
elif page == "About":
    st.markdown("# About DefectForge AI")
    st.markdown("---")

    st.markdown("""
    <div class="custom-card">
        <h3>What is DefectForge AI?</h3>
        <p>DefectForge AI is an intelligent steel surface defect detection system powered by deep learning.
        It uses a Convolutional Neural Network (CNN) trained on the NEU-DET dataset to classify six types
        of steel surface defects with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-card">
        <h3>Defect Types Detected</h3>
        <p>The system identifies six critical steel surface defects:</p>
        <ul>
            <li><b>Crazing</b> -- Network of fine cracks from stress or thermal cycling</li>
            <li><b>Inclusion</b> -- Non-metallic particles embedded in the steel</li>
            <li><b>Patches</b> -- Localized surface irregularities and discoloration</li>
            <li><b>Pitted Surface</b> -- Small cavities from corrosion or chemical attack</li>
            <li><b>Rolled-in Scale</b> -- Oxide scale pressed in during rolling</li>
            <li><b>Scratches</b> -- Linear grooves from mechanical contact</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-card">
        <h3>Technology Stack</h3>
        <p><b>Deep Learning:</b> PyTorch, CNN Architecture</p>
        <p><b>Frontend:</b> Streamlit Dashboard</p>
        <p><b>Visualization:</b> Plotly Interactive Charts</p>
        <p><b>Dataset:</b> NEU-DET (Northeastern University)</p>
        <p><b>Application:</b> Industrial Quality Control</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-card">
        <h3>How to Use</h3>
        <p>1. Navigate to <b>Analyze Defect</b> from the sidebar</p>
        <p>2. Upload a steel surface image (JPG, PNG, BMP)</p>
        <p>3. The model will detect the defect type and confidence level</p>
        <p>4. Review the probability distribution across all defect classes</p>
        <p>5. Check <b>Model Performance</b> for detailed evaluation metrics</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>DefectForge AI -- Steel Surface Defect Analyzer | PyTorch &amp; Streamlit</p>
</div>
""", unsafe_allow_html=True)


"""
FRESHCHECK AI
---------------
Deep Learning : Fruit freshness image classification
Machine Learning: Spoilage-risk demonstration
Generative AI : OpenRouter-powered food assistant
Database       : SQLite food inventory
Analytics      : Inventory reports and CSV export

IMPORTANT:
- CNN confidence is not a food-safety probability.
- Risk model uses synthetic proxy labels, not validated spoilage data.
- Image analysis cannot detect invisible pathogens or toxins.
"""

import csv
import hashlib
import io
import json
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DB_PATH = DATA_DIR / "freshcheck.db"
CNN_PATH = MODEL_DIR / "freshness_cnn.keras"
RISK_PATH = MODEL_DIR / "spoilage_risk.joblib"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="FreshCheck AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. LIGHT AND CLEAN APP DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f5fbf8 0%,
            #f8fafc 55%,
            #fffaf2 100%
        );
        color: #263b35;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #edf6f1;
        border-right: 1px solid #dcebe3;
    }

    section[data-testid="stSidebar"] * {
        color: #29483d;
    }

    /* Headings */
    h1, h2, h3 {
        color: #294d40 !important;
        font-weight: 600 !important;
        letter-spacing: -0.4px;
    }

    /* Paragraphs */
    p, label {
        color: #52665f;
    }

    /* Cards */
    div[data-testid="stMetric"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255,255,255,0.82);
        border: 1px solid #e1ece6;
        border-radius: 14px;
        padding: 12px;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background-color: #dcefe5;
        color: #285340;
        border: 1px solid #c5e1d1;
        border-radius: 10px;
        padding: 0.55rem 1.1rem;
        font-weight: 500;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #cce7d8;
        border-color: #a9d4bd;
        color: #214b38;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #4d9976;
        color: white;
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #3c805f;
        color: white;
    }

    /* Input fields */
    input, textarea {
        border-radius: 8px !important;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.7);
        border: 1px dashed #b9d8c7;
        border-radius: 12px;
        padding: 12px;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Horizontal divider */
    hr {
        border-color: #e0ebe5;
    }

    /* Progress bars */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #6eae8b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. HEADER
# ============================================================

st.markdown(
    """
    <div style="
        padding: 24px 28px;
        background: linear-gradient(110deg, #e5f4eb, #eef5fb);
        border: 1px solid #dcebe4;
        border-radius: 18px;
        margin-bottom: 22px;
    ">
        <div style="font-size: 36px;">🍎 FreshCheck AI</div>
        <div style="
            color: #59736a;
            font-size: 16px;
            margin-top: 6px;
        ">
            Smarter food freshness insights, storage guidance
            and inventory management.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "FreshCheck AI is an educational prototype. It cannot certify "
    "food safety or detect invisible pathogens, toxins or contamination."
)


# ============================================================
# 4. OPENROUTER CONFIGURATION
# ============================================================

def read_secret_or_environment(name, default=""):
    """
    Read configuration from Streamlit secrets.
    If unavailable, try the Windows environment variable.
    """

    value = None

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    if value is None or not str(value).strip():
        value = os.getenv(name, default)

    return str(value).strip() if value is not None else default


def get_openrouter_config():

    api_key = read_secret_or_environment(
        "OPENROUTER_API_KEY"
    )

    model = read_secret_or_environment(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    return api_key, model


# ============================================================
# 5. OPENROUTER API CONNECTION
#    No OpenAI SDK. No jiter. Uses Python urllib.
# ============================================================

def call_openrouter(
    prompt,
    system_prompt=None,
    max_tokens=800
):

    api_key, model = get_openrouter_config()

    if not api_key:
        return None, (
            "OPENROUTER_API_KEY is missing. "
            "Check your .streamlit/secrets.toml file."
        )

    if not model:
        return None, "OPENROUTER_MODEL is missing."

    url = "https://openrouter.ai/api/v1/chat/completions"

    system_message = system_prompt or (
        "You are FreshCheck AI, an intelligent food-storage "
        "and food-waste reduction assistant. "
        "Give accurate, clear and practical explanations. "
        "Use simple language where possible. "
        "Never claim that a CNN image prediction can certify "
        "food safety or detect invisible pathogens. "
        "Explain limitations whenever relevant."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": str(prompt)
            }
        ],
        "temperature": 0.3,
        "max_tokens": int(max_tokens)
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "FreshCheck AI"
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=90
        ) as response:

            response_data = response.read().decode("utf-8")
            result = json.loads(response_data)

        answer = result["choices"][0]["message"]["content"]

        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", "")
                for item in answer
                if isinstance(item, dict)
            )

        if not answer or not str(answer).strip():
            return None, "OpenRouter returned an empty response."

        return str(answer).strip(), None

    except urllib.error.HTTPError as error:

        error_body = error.read().decode(
            "utf-8",
            errors="replace"
        )

        try:
            error_json = json.loads(error_body)
            details = error_json.get(
                "error", {}
            ).get("message", error_body)
        except Exception:
            details = error_body

        if error.code == 401:
            return None, (
                "OpenRouter authentication failed. "
                "Check your API key and generate a replacement "
                "if the old key was revoked."
            )

        if error.code == 402:
            return None, (
                "OpenRouter account has insufficient credits "
                "or a payment issue."
            )

        if error.code == 403:
            return None, (
                "OpenRouter access denied. Check your account "
                "permissions and model access."
            )

        if error.code == 404:
            return None, (
                f"Model not found or unavailable: {model}. "
                "Check the model ID."
            )

        if error.code == 429:
            return None, (
                "OpenRouter rate limit reached. "
                "Wait and try again."
            )

        return None, (
            f"OpenRouter HTTP error {error.code}: {details}"
        )

    except urllib.error.URLError as error:

        return None, f"Network connection failed: {error.reason}"

    except (KeyError, IndexError, TypeError, ValueError) as error:

        return None, f"Invalid OpenRouter response: {error}"

    except Exception as error:

        return None, f"GenAI request failed: {error}"


def test_openrouter_connection():

    return call_openrouter(
        "Reply with exactly: FreshCheck AI connection successful.",
        system_prompt=(
            "You are a connection test assistant. "
            "Reply with the requested short confirmation."
        ),
        max_tokens=40
    )


# ============================================================
# 6. SQLITE DATABASE
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food TEXT NOT NULL,
            quantity REAL NOT NULL,
            unit TEXT NOT NULL,
            purchase_date TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            storage TEXT,
            notes TEXT
        )
        """
    )

    connection.commit()

    return connection


def fetch_inventory():

    with get_db_connection() as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM inventory
            ORDER BY expiry_date ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def add_inventory_item(
    food,
    quantity,
    unit,
    purchase_date,
    expiry_date,
    storage,
    notes
):

    with get_db_connection() as connection:

        connection.execute(
            """
            INSERT INTO inventory (
                food,
                quantity,
                unit,
                purchase_date,
                expiry_date,
                storage,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                food.strip(),
                float(quantity),
                unit,
                purchase_date.isoformat(),
                expiry_date.isoformat(),
                storage,
                notes.strip()
            )
        )

        connection.commit()


def delete_inventory_item(item_id):

    with get_db_connection() as connection:

        connection.execute(
            "DELETE FROM inventory WHERE id = ?",
            (int(item_id),)
        )

        connection.commit()


# ============================================================
# 7. LOAD CNN MODEL
# ============================================================

CLASSES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

CLASS_DETAILS = {
    "freshapples": ("Apples", "Fresh-looking"),
    "freshbanana": ("Bananas", "Fresh-looking"),
    "freshoranges": ("Oranges", "Fresh-looking"),
    "rottenapples": ("Apples", "Rotten-looking"),
    "rottenbanana": ("Bananas", "Rotten-looking"),
    "rottenoranges": ("Oranges", "Rotten-looking")
}


@st.cache_resource
def load_cnn_model():

    if not CNN_PATH.exists():
        return None, f"CNN model not found: {CNN_PATH}"

    try:

        import tensorflow as tf

        model = tf.keras.models.load_model(CNN_PATH)

        if model.output_shape[-1] != len(CLASSES):
            return None, (
                "CNN output does not match the six expected classes."
            )

        return model, None

    except Exception as error:

        return None, f"CNN loading failed: {error}"


@st.cache_resource
def load_risk_model():

    if not RISK_PATH.exists():
        return None, f"Risk model not found: {RISK_PATH}"

    try:

        import joblib

        model = joblib.load(RISK_PATH)

        return model, None

    except Exception as error:

        return None, f"Risk model loading failed: {error}"


cnn, cnn_error = load_cnn_model()
risk_model, risk_error = load_risk_model()


# ============================================================
# 8. CNN IMAGE PREDICTION
# ============================================================

def predict_image(image):

    if cnn is None:
        raise RuntimeError(
            cnn_error or "CNN model is unavailable."
        )

    image = image.convert("RGB").resize((224, 224))

    # IMPORTANT:
    # Your trained model already contains Rescaling(1./255).
    # Therefore, DO NOT divide this input by 255 again.

    array = np.asarray(
        image,
        dtype=np.float32
    )

    predictions = cnn.predict(
        np.expand_dims(array, axis=0),
        verbose=0
    )[0]

    if len(predictions) != len(CLASSES):
        raise ValueError("Unexpected CNN output dimensions.")

    index = int(np.argmax(predictions))

    class_name = CLASSES[index]

    fruit, condition = CLASS_DETAILS[class_name]

    score_list = []

    for i, label in enumerate(CLASSES):

        score_fruit, score_condition = CLASS_DETAILS[label]

        score_list.append({
            "label": f"{score_fruit} — {score_condition}",
            "probability": float(predictions[i])
        })

    return {
        "class_name": class_name,
        "fruit": fruit,
        "condition": condition,
        "confidence": float(predictions[index]),
        "scores": score_list
    }


# ============================================================
# 9. RANDOM FOREST RISK ESTIMATION
# ============================================================

def estimate_spoilage_risk(
    food,
    days,
    temperature,
    humidity
):

    if risk_model is None:
        return None, risk_error or "Risk model unavailable."

    fruit_codes = {
        "apple": 0,
        "banana": 1,
        "orange": 2
    }

    features = [[
        fruit_codes[food.lower()],
        float(days),
        float(temperature),
        float(humidity)
    ]]

    try:

        probabilities = risk_model.predict_proba(features)[0]

        model_classes = list(risk_model.classes_)

        if 1 not in model_classes:
            return None, "Risk model does not contain class 1."

        positive_index = model_classes.index(1)

        score = float(probabilities[positive_index])

        return score, None

    except Exception as error:

        return None, f"Risk prediction failed: {error}"


# ============================================================
# 10. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("## 🍃 FreshCheck AI")

st.sidebar.caption(
    "Food quality insights powered by AI"
)

pages = [
    "Home",
    "Freshness Scanner",
    "Food Inventory",
    "AI Food Assistant",
    "Analytics & Reports",
    "About & Setup"
]

page = st.sidebar.radio(
    "Explore",
    pages
)

st.sidebar.divider()

api_key, model_name = get_openrouter_config()

if api_key:

    st.sidebar.success(
        f"OpenRouter key detected\n\nModel: {model_name}"
    )

else:

    st.sidebar.error(
        "OpenRouter API key not configured."
    )

if st.sidebar.button("Test OpenRouter connection"):

    with st.sidebar:
        with st.spinner("Testing OpenRouter..."):

            test_answer, test_error = test_openrouter_connection()

        if test_error:
            st.error(test_error)
        else:
            st.success("OpenRouter connection successful!")


if cnn is None:
    st.sidebar.warning("CNN model unavailable.")
else:
    st.sidebar.success("CNN model loaded.")

if risk_model is None:
    st.sidebar.warning("Risk model unavailable.")
else:
    st.sidebar.success("Risk model loaded.")


# ============================================================
# 11. HOME PAGE
# ============================================================

if page == "Home":

    st.subheader("Welcome to FreshCheck AI")

    st.write(
        "An intelligent food management application combining "
        "Deep Learning, Machine Learning and Generative AI."
    )

    inventory = fetch_inventory()

    today = date.today()

    expiring_count = 0

    for item in inventory:

        expiry = date.fromisoformat(item["expiry_date"])

        if today <= expiry <= today + timedelta(days=3):
            expiring_count += 1

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Inventory items",
        len(inventory)
    )

    col2.metric(
        "Expiring within 3 days",
        expiring_count
    )

    col3.metric(
        "CNN model",
        "Ready" if cnn is not None else "Unavailable"
    )

    st.divider()

    st.markdown("### Explore the application")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div style="
                background:#edf7f0;
                border:1px solid #d8eadf;
                border-radius:14px;
                padding:20px;
                min-height:155px;
            ">
                <div style="font-size:27px;">📷</div>
                <h3>Freshness Scanner</h3>
                <p>Upload fruit images and receive CNN-based
                visual classification results.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div style="
                background:#f1f5fc;
                border:1px solid #dce5f3;
                border-radius:14px;
                padding:20px;
                min-height:155px;
            ">
                <div style="font-size:27px;">🤖</div>
                <h3>Generative AI Assistant</h3>
                <p>Ask food-storage questions and receive
                AI-generated explanations through OpenRouter.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div style="
                background:#fff7eb;
                border:1px solid #f0e5d2;
                border-radius:14px;
                padding:20px;
                min-height:155px;
            ">
                <div style="font-size:27px;">🧺</div>
                <h3>Food Inventory</h3>
                <p>Track food quantities, purchase dates
                and expiry dates.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div style="
                background:#f5f0fa;
                border:1px solid #e6dcef;
                border-radius:14px;
                padding:20px;
                min-height:155px;
            ">
                <div style="font-size:27px;">📊</div>
                <h3>Analytics & Reports</h3>
                <p>Review inventory information and
                download CSV reports.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 12. FRESHNESS SCANNER
# ============================================================

elif page == "Freshness Scanner":

    st.subheader("Fruit Freshness Scanner")

    st.write(
        "Upload a clear image containing one apple, banana or orange."
    )

    if cnn is None:

        st.error(cnn_error)

    else:

        uploaded_file = st.file_uploader(
            "Upload fruit image",
            type=["jpg", "jpeg", "png", "webp"],
            key="fruit_image"
        )

        if uploaded_file is not None:

            try:

                image_bytes = uploaded_file.getvalue()

                image_id = hashlib.sha256(
                    image_bytes
                ).hexdigest()

                image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded fruit image",
                    width=350
                )

                if st.session_state.get("scan_image_id") != image_id:

                    st.session_state["scan_image_id"] = image_id

                    st.session_state.pop(
                        "scan_result",
                        None
                    )

                if st.button(
                    "Analyze image",
                    type="primary"
                ):

                    with st.spinner("Analyzing image..."):

                        st.session_state["scan_result"] = predict_image(
                            image
                        )

                result = st.session_state.get("scan_result")

                if result:

                    st.divider()

                    st.markdown(
                        f"### {result['fruit']} — {result['condition']}"
                    )

                    confidence = result["confidence"]

                    st.metric(
                        "Model confidence",
                        f"{confidence * 100:.1f}%"
                    )

                    st.progress(confidence)

                    st.caption(
                        "This confidence represents the model's "
                        "prediction for a dataset class. It is not "
                        "a food-safety probability."
                    )

                    if confidence < 0.55:

                        st.warning(
                            "The model has relatively low confidence. "
                            "Try another clear image with one fruit "
                            "and a simple background."
                        )

                    st.markdown("#### All class probabilities")

                    for score in result["scores"]:

                        st.write(
                            f"{score['label']}: "
                            f"{score['probability'] * 100:.2f}%"
                        )

                        st.progress(
                            score["probability"]
                        )

                    st.divider()

                    if st.button("Explain result with GenAI"):

                        prompt = (
                            f"The image classifier predicted "
                            f"{result['fruit']} — {result['condition']}. "
                            f"The model confidence is "
                            f"{confidence * 100:.2f}%. "
                            f"Explain this result in simple scientific "
                            f"language. Explain possible visible signs, "
                            f"the limitations of image classification, "
                            f"and general storage advice. Do not claim "
                            f"the image proves the fruit is safe or unsafe."
                        )

                        with st.spinner(
                            "Generating explanation with OpenRouter..."
                        ):

                            answer, error = call_openrouter(prompt)

                        if error:

                            st.error(error)

                        else:

                            st.markdown("### AI Explanation")

                            st.markdown(answer)

            except Exception as error:

                st.error(f"Image analysis failed: {error}")

    st.divider()

    st.subheader("ML Spoilage-Risk Demonstration")

    st.warning(
        "This Random Forest uses synthetic proxy labels. "
        "Its output is not a validated real-world spoilage probability."
    )

    col1, col2 = st.columns(2)

    with col1:

        risk_food = st.selectbox(
            "Fruit type",
            ["apple", "banana", "orange"]
        )

        days = st.number_input(
            "Days since purchase",
            min_value=0,
            max_value=60,
            value=2
        )

    with col2:

        temperature = st.number_input(
            "Storage temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=22.0
        )

        humidity = st.number_input(
            "Relative humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

    if st.button("Calculate demonstration score"):

        score, error = estimate_spoilage_risk(
            risk_food,
            days,
            temperature,
            humidity
        )

        if error:

            st.error(error)

        else:

            st.metric(
                "Synthetic proxy-risk score",
                f"{score * 100:.1f}%"
            )

            st.caption(
                "Educational model output only. "
                "Not a food-safety assessment."
            )


# ============================================================
# 13. FOOD INVENTORY
# ============================================================

elif page == "Food Inventory":

    st.subheader("Food Inventory Manager")

    with st.form(
        "inventory_form",
        clear_on_submit=True
    ):

        col1, col2, col3 = st.columns(3)

        food = col1.text_input(
            "Food item",
            placeholder="e.g. Apples"
        )

        quantity = col2.number_input(
            "Quantity",
            min_value=0.1,
            value=1.0,
            step=0.5
        )

        unit = col3.selectbox(
            "Unit",
            ["pieces", "kg", "g", "litres", "packets"]
        )

        col4, col5, col6 = st.columns(3)

        purchase = col4.date_input(
            "Purchase date",
            value=date.today()
        )

        expiry = col5.date_input(
            "Expiry / best-before date",
            value=date.today() + timedelta(days=5)
        )

        storage = col6.selectbox(
            "Storage",
            [
                "Room temperature",
                "Refrigerator",
                "Freezer",
                "Other"
            ]
        )

        notes = st.text_input("Additional notes")

        submitted = st.form_submit_button(
            "Add to inventory",
            type="primary"
        )

    if submitted:

        if not food.strip():

            st.error("Enter a food item name.")

        elif expiry < purchase:

            st.error(
                "Expiry date cannot be earlier than purchase date."
            )

        else:

            add_inventory_item(
                food,
                quantity,
                unit,
                purchase,
                expiry,
                storage,
                notes
            )

            st.success("Food item added successfully.")

            st.rerun()

    st.divider()

    st.markdown("### Your inventory")

    inventory = fetch_inventory()

    if not inventory:

        st.info("Your inventory is empty. Add your first item above.")

    else:

        for item in inventory:

            expiry_date = date.fromisoformat(
                item["expiry_date"]
            )

            days_left = (
                expiry_date - date.today()
            ).days

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.markdown(f"**{item['food']}**")

                    st.write(
                        f"Quantity: {item['quantity']:g} {item['unit']}"
                    )

                    st.write(
                        f"Purchase date: {item['purchase_date']}"
                    )

                    st.write(
                        f"Expiry date: {item['expiry_date']}"
                    )

                    st.write(
                        f"Storage: {item['storage']}"
                    )

                    if item["notes"]:
                        st.caption(item["notes"])

                with col2:

                    if days_left < 0:

                        st.error("Past entered date")

                    elif days_left <= 3:

                        st.warning(
                            f"{days_left} day(s) left"
                        )

                    else:

                        st.success(
                            f"{days_left} days left"
                        )

                    if st.button(
                        "Delete",
                        key=f"delete_{item['id']}"
                    ):

                        delete_inventory_item(item["id"])

                        st.rerun()


# ============================================================
# 14. AI FOOD ASSISTANT
# ============================================================

elif page == "AI Food Assistant":

    st.subheader("🤖 AI Food Assistant")

    st.write(
        "Ask questions about food storage, fruit freshness, "
        "food handling and reducing food waste."
    )

    api_key, model_name = get_openrouter_config()

    if api_key:

        st.success(
            f"Configured OpenRouter model: {model_name}"
        )

    else:

        st.error(
            "OpenRouter API key is missing. "
            "Configure .streamlit/secrets.toml."
        )

    if "chat_history" not in st.session_state:

        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    question = st.chat_input(
        "Ask FreshCheck AI anything about food..."
    )

    if question:

        st.session_state["chat_history"].append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Generating your answer using OpenRouter..."
            ):

                answer, error = call_openrouter(question)

            if error:

                st.error(error)

                displayed_answer = error

            else:

                st.markdown(answer)

                displayed_answer = answer

        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": displayed_answer
            }
        )

    if st.session_state["chat_history"]:

        if st.button("Clear chat history"):

            st.session_state["chat_history"] = []

            st.rerun()


# ============================================================
# 15. ANALYTICS AND REPORTS
# ============================================================

elif page == "Analytics & Reports":

    st.subheader("Inventory Analytics & Reports")

    inventory = fetch_inventory()

    if not inventory:

        st.info(
            "Add food items to your inventory to see analytics."
        )

    else:

        today = date.today()

        expired_count = 0
        expiring_count = 0

        quantity_summary = {}

        report_rows = []

        for item in inventory:

            expiry_date = date.fromisoformat(
                item["expiry_date"]
            )

            days_left = (
                expiry_date - today
            ).days

            if days_left < 0:

                status = "Past entered date"

                expired_count += 1

            elif days_left <= 3:

                status = "Expiring soon"

                expiring_count += 1

            else:

                status = "Within entered date"

            quantity_summary[item["food"]] = (
                quantity_summary.get(item["food"], 0)
                + float(item["quantity"])
            )

            report_rows.append({
                "ID": item["id"],
                "Food": item["food"],
                "Quantity": item["quantity"],
                "Unit": item["unit"],
                "Purchase date": item["purchase_date"],
                "Expiry date": item["expiry_date"],
                "Days remaining": days_left,
                "Storage": item["storage"],
                "Notes": item["notes"],
                "Status": status
            })

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total items",
            len(inventory)
        )

        col2.metric(
            "Expiring soon",
            expiring_count
        )

        col3.metric(
            "Past entered date",
            expired_count
        )

        st.divider()

        st.markdown("### Quantity by food item")

        for food_name, quantity_value in quantity_summary.items():

            st.write(
                f"**{food_name}:** {quantity_value:g}"
            )

        st.divider()

        st.markdown("### Inventory details")

        for row in report_rows:

            with st.container(border=True):

                st.write(
                    f"**{row['Food']}** — "
                    f"{row['Quantity']} {row['Unit']}"
                )

                st.write(
                    f"Expiry: {row['Expiry date']} | "
                    f"Status: {row['Status']}"
                )

        # CSV EXPORT

        csv_buffer = io.StringIO()

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=list(report_rows[0].keys())
        )

        writer.writeheader()

        writer.writerows(report_rows)

        st.download_button(
            "Download inventory CSV report",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="freshcheck_inventory_report.csv",
            mime="text/csv"
        )

        st.divider()

        if st.button("Generate AI food-waste suggestions"):

            summary = "\n".join(
                [
                    f"- {row['Food']}: "
                    f"{row['Quantity']} {row['Unit']}, "
                    f"expiry {row['Expiry date']}, "
                    f"storage {row['Storage']}"
                    for row in report_rows
                ]
            )

            prompt = (
                "Analyze this food inventory and suggest practical "
                "ways to reduce food waste and improve storage. "
                "Do not guarantee shelf life or certify food safety.\n\n"
                + summary
            )

            with st.spinner(
                "Generating suggestions using OpenRouter..."
            ):

                answer, error = call_openrouter(prompt)

            if error:

                st.error(error)

            else:

                st.markdown("### AI Suggestions")

                st.markdown(answer)


# ============================================================
# 16. ABOUT AND SETUP
# ============================================================

elif page == "About & Setup":

    st.subheader("About FreshCheck AI")

    st.markdown(
        """
        FreshCheck AI is a portfolio project integrating
        three major AI technologies with food inventory management.

        ### 1. Deep Learning
        CNN-based image classification of:
        - Fresh-looking apples
        - Fresh-looking bananas
        - Fresh-looking oranges
        - Rotten-looking apples
        - Rotten-looking bananas
        - Rotten-looking oranges

        ### 2. Machine Learning
        Random Forest model using:
        - Fruit type
        - Days since purchase
        - Storage temperature
        - Relative humidity

        ### 3. Generative AI
        OpenRouter-hosted language model for:
        - Food storage questions
        - Image-result explanations
        - Food-waste reduction suggestions

        ### 4. Database
        SQLite-based inventory management.

        ### 5. Analytics
        Inventory summaries and CSV reports.
        """
    )

    st.divider()

    st.markdown("### Model status")

    if cnn is not None:

        st.success("CNN model loaded successfully.")

    else:

        st.error(cnn_error)

    if risk_model is not None:

        st.success("Random Forest model loaded successfully.")

    else:

        st.warning(risk_error)

    api_key, model_name = get_openrouter_config()

    if api_key:

        st.success(
            f"OpenRouter API key detected. Model: {model_name}"
        )

    else:

        st.error("OpenRouter API key not detected.")

    st.divider()

    st.markdown("### OpenRouter configuration")

    st.write(
        "Create `.streamlit/secrets.toml` inside your project folder."
    )

    st.code(
        'OPENROUTER_API_KEY = "YOUR_NEW_API_KEY"\n'
        'OPENROUTER_MODEL = "openai/gpt-4o-mini"',
        language="toml"
    )

    st.info(
        "Keep your API key private. Do not place it in app.py "
        "or upload secrets.toml to GitHub."
    )

    st.divider()

    st.markdown("### Scientific limitations")

    st.warning(
        "CNN image classification detects visual patterns learned "
        "from the training dataset. It cannot detect invisible "
        "pathogens or certify food safety."
    )

    st.warning(
        "The Random Forest uses synthetic proxy labels. "
        "Its predictions are not validated against real-world "
        "spoilage observations."
    )

    st.caption(
        "FreshCheck AI | Educational AI portfolio project"
    )

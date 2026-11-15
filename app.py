
from pathlib import Path
from datetime import date, datetime
import hashlib
import io
import os
import sqlite3
import csv

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models"
CNN_PATH = MODEL_DIR / "freshness_cnn.keras"
RISK_PATH = MODEL_DIR / "spoilage_risk.joblib"
DB_PATH = ROOT / "freshcheck_inventory.db"

IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="FreshCheck AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

FRUIT_CODES = {
    "Apple": 0,
    "Banana": 1,
    "Orange": 2
}


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #176b45;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #5c6670;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .info-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e9ef;
        margin-bottom: 12px;
    }

    .small-note {
        color: #667085;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. MODEL FILE VERIFICATION
# ============================================================

def get_file_hash(path):
    """Return SHA256 hash of a model file."""

    sha = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha.update(chunk)

    return sha.hexdigest()


def get_model_fingerprint(path):
    """
    Create a fingerprint for Streamlit caching.
    If the model file changes, the cache key changes too.
    """

    if not path.exists():
        return "missing"

    return get_file_hash(path)


# ============================================================
# 4. LOAD CNN MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading fruit freshness model...")
def load_cnn_model(model_path, fingerprint):

    if not Path(model_path).exists():
        return None, f"CNN model not found: {model_path}"

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        if model.output_shape[-1] != len(CLASSES):
            return None, (
                "CNN output does not match the six expected classes. "
                f"Expected {len(CLASSES)}, "
                f"received {model.output_shape[-1]}."
            )

        if len(model.input_shape) != 4:
            return None, "Unexpected CNN input shape."

        if model.input_shape[-1] != 3:
            return None, (
                "The CNN model is not configured for RGB images."
            )

        return model, None

    except Exception as error:
        return None, f"CNN loading failed: {error}"


# ============================================================
# 5. LOAD SPOILAGE RISK MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading spoilage-risk model...")
def load_risk_model(model_path, fingerprint):

    if not Path(model_path).exists():
        return None, f"Risk model not found: {model_path}"

    try:
        import joblib

        model = joblib.load(model_path)

        return model, None

    except Exception as error:
        return None, f"Risk model loading failed: {error}"


cnn_fingerprint = get_model_fingerprint(CNN_PATH)
risk_fingerprint = get_model_fingerprint(RISK_PATH)

cnn, cnn_error = load_cnn_model(
    str(CNN_PATH),
    cnn_fingerprint
)

risk_model, risk_error = load_risk_model(
    str(RISK_PATH),
    risk_fingerprint
)


# ============================================================
# 6. CNN IMAGE PREDICTION
# ============================================================

def predict_image(uploaded_image):

    if cnn is None:
        raise RuntimeError(
            cnn_error or "CNN model is unavailable."
        )

    # Correct image orientation and RGB conversion
    image = ImageOps.exif_transpose(uploaded_image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    # IMPORTANT:
    # Your trained CNN already contains Rescaling(1./255).
    # Therefore, do NOT divide this array by 255 again.

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    if image_array.shape != (224, 224, 3):
        raise ValueError(
            f"Unexpected image shape: {image_array.shape}"
        )

    predictions = cnn.predict(
        np.expand_dims(image_array, axis=0),
        verbose=0
    )[0]

    if len(predictions) != len(CLASSES):
        raise ValueError(
            "Unexpected CNN output dimensions."
        )

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASSES[predicted_index]

    fruit, condition = CLASS_DETAILS[predicted_class]

    score_list = []

    for index, class_name in enumerate(CLASSES):

        score_fruit, score_condition = CLASS_DETAILS[class_name]

        score_list.append({
            "label": f"{score_fruit} — {score_condition}",
            "probability": float(predictions[index])
        })

    return {
        "class_name": predicted_class,
        "fruit": fruit,
        "condition": condition,
        "confidence": float(predictions[predicted_index]),
        "scores": score_list
    }


# ============================================================
# 7. SPOILAGE RISK PREDICTION
# ============================================================

def predict_spoilage_risk(
    fruit,
    days_since_purchase,
    temperature,
    humidity
):

    if risk_model is None:
        raise RuntimeError(
            risk_error or "Risk model is unavailable."
        )

    fruit_code = FRUIT_CODES[fruit]

    features = np.array([[
        fruit_code,
        days_since_purchase,
        temperature,
        humidity
    ]], dtype=np.float64)

    probabilities = risk_model.predict_proba(features)[0]
    model_classes = list(risk_model.classes_)

    if 1 in model_classes:
        risk_index = model_classes.index(1)
    else:
        risk_index = int(np.argmax(probabilities))

    risk_probability = float(probabilities[risk_index])

    if risk_probability < 0.35:
        risk_level = "Low"
        explanation = "The model estimates a relatively low risk."
    elif risk_probability < 0.70:
        risk_level = "Moderate"
        explanation = "The model estimates a moderate risk."
    else:
        risk_level = "High"
        explanation = "The model estimates a relatively high risk."

    return {
        "probability": risk_probability,
        "level": risk_level,
        "explanation": explanation
    }


# ============================================================
# 8. SQLITE INVENTORY DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():

    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fruit TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            purchase_date TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            notes TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def add_inventory_item(
    fruit,
    quantity,
    purchase_date,
    expiry_date,
    notes
):

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO inventory
        (fruit, quantity, purchase_date, expiry_date, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            fruit,
            quantity,
            purchase_date.isoformat(),
            expiry_date.isoformat(),
            notes
        )
    )

    connection.commit()
    connection.close()


def get_inventory():

    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT id, fruit, quantity, purchase_date, expiry_date, notes
        FROM inventory
        ORDER BY expiry_date ASC
        """
    )

    rows = cursor.fetchall()
    connection.close()

    columns = [
        "ID",
        "Fruit",
        "Quantity",
        "Purchase Date",
        "Expiry Date",
        "Notes"
    ]

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def delete_inventory_item(item_id):

    connection = get_connection()

    connection.execute(
        "DELETE FROM inventory WHERE id = ?",
        (item_id,)
    )

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# 9. OPENROUTER GENAI ASSISTANT
# ============================================================

def get_openrouter_settings():

    api_key = None
    model_name = "openai/gpt-4o-mini"

    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY")
        model_name = st.secrets.get(
            "OPENROUTER_MODEL",
            model_name
        )
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not os.getenv("OPENROUTER_MODEL"):
        model_name = os.getenv(
            "OPENROUTER_MODEL",
            model_name
        )

    return api_key, model_name


def ask_openrouter(question, chat_history):

    api_key, model_name = get_openrouter_settings()

    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is not configured. "
            "Add OPENROUTER_API_KEY in Streamlit Cloud Secrets."
        )

    system_prompt = """
You are FreshCheck AI, a helpful food freshness and storage assistant.

Your responsibilities:
1. Answer questions about fruit storage and food freshness.
2. Explain common visible signs of fruit spoilage.
3. Give practical food storage recommendations.
4. Explain how temperature and humidity can affect fruit storage.
5. Use simple language suitable for students and general users.

Important safety rules:
- Never claim that an image can prove food is safe to eat.
- Visible appearance cannot detect all harmful microorganisms.
- If food has mold, unusual odor, sliminess, leakage, or other clear
  signs of spoilage, advise the user not to consume it.
- Do not present the application's ML risk estimate as a validated
  food-safety measurement.
- If uncertain, clearly say so.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for message in chat_history[-8:]:
        messages.append(message)

    messages.append({
        "role": "user",
        "content": question
    })

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://freshcheck-ai.streamlit.app",
            "X-Title": "FreshCheck AI"
        },
        json={
            "model": model_name,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 900
        },
        timeout=60
    )

    if response.status_code != 200:
        try:
            error_details = response.json()
        except Exception:
            error_details = response.text

        raise RuntimeError(
            f"OpenRouter error ({response.status_code}): "
            f"{error_details}"
        )

    result = response.json()

    return result["choices"][0]["message"]["content"]


# ============================================================
# 10. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🍎 FreshCheck AI")

st.sidebar.caption(
    "AI-powered fruit freshness and storage assistant"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Freshness Scanner",
        "Spoilage Risk Estimator",
        "GenAI Food Assistant",
        "Fruit Inventory",
        "Analytics",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "FreshCheck AI is an educational prototype. "
    "It does not certify food safety."
)


# ============================================================
# 11. HOME PAGE
# ============================================================

if page == "Home":

    st.markdown(
        '<div class="main-title">🍎 FreshCheck AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'An intelligent fruit freshness and storage assistant '
        'powered by Deep Learning, Machine Learning and Generative AI.'
        '</div>',
        unsafe_allow_html=True
    )

    inventory = get_inventory()

    total_items = sum(
        item["Quantity"] for item in inventory
    )

    today = date.today()

    expiring_items = 0

    for item in inventory:
        try:
            expiry = date.fromisoformat(item["Expiry Date"])

            if expiry <= today:
                expiring_items += 1

        except ValueError:
            pass

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Inventory Entries",
        len(inventory)
    )

    col2.metric(
        "Total Fruit Quantity",
        total_items
    )

    col3.metric(
        "Expired / Due Today",
        expiring_items
    )

    st.divider()

    st.subheader("Explore FreshCheck AI")

    card1, card2 = st.columns(2)

    with card1:
        st.markdown(
            """
            <div class="info-card">
                <h3>📷 Freshness Scanner</h3>
                Upload a fruit image and classify it as
                fresh-looking or rotten-looking using a CNN.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <h3>🤖 GenAI Food Assistant</h3>
                Ask questions about fruit storage,
                freshness and food handling.
            </div>
            """,
            unsafe_allow_html=True
        )

    with card2:
        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Spoilage Risk Estimator</h3>
                Explore an ML-based risk estimate using
                fruit type, storage duration, temperature
                and humidity.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <h3>📦 Fruit Inventory</h3>
                Record fruit purchases and expiry dates
                and monitor your inventory.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("How the system works")

    st.markdown(
        """
        1. **Deep Learning:** A CNN analyses an uploaded fruit image.
        2. **Machine Learning:** A Random Forest model estimates a
           spoilage-risk probability from storage-related inputs.
        3. **Generative AI:** OpenRouter provides conversational
           food-storage guidance.
        4. **Database:** SQLite stores inventory and expiry details.
        """
    )


# ============================================================
# 12. FRESHNESS SCANNER
# ============================================================

elif page == "Freshness Scanner":

    st.title("📷 Fruit Freshness Scanner")

    st.write(
        "Upload an image of an apple, banana or orange. "
        "The CNN will predict its visible freshness category."
    )

    if cnn is None:
        st.error(cnn_error)

        st.info(
            "Check that models/freshness_cnn.keras exists "
            "in your GitHub repository."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload a fruit image",
            type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded_file is not None:

            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded fruit image",
                    width=350
                )

                if st.button(
                    "Analyse Freshness",
                    type="primary"
                ):

                    with st.spinner("Analysing image..."):

                        result = predict_image(image)

                    st.divider()

                    st.subheader("CNN Prediction")

                    st.metric(
                        "Predicted Fruit",
                        result["fruit"]
                    )

                    if result["condition"] == "Fresh-looking":
                        st.success(
                            f"Prediction: {result['condition']}"
                        )
                    else:
                        st.warning(
                            f"Prediction: {result['condition']}"
                        )

                    st.metric(
                        "Model Confidence",
                        f"{result['confidence'] * 100:.2f}%"
                    )

                    if result["confidence"] < 0.60:
                        st.warning(
                            "The model has relatively low confidence. "
                            "Try a clearer image with one fruit in focus."
                        )

                    st.subheader("Class-wise Prediction Scores")

                    for score in result["scores"]:

                        st.write(
                            f"**{score['label']}** — "
                            f"{score['probability'] * 100:.2f}%"
                        )

                        st.progress(
                            max(
                                0.0,
                                min(1.0, score["probability"])
                            )
                        )

                    st.info(
                        "This result describes visible appearance only. "
                        "It cannot establish whether food is safe to eat "
                        "or detect invisible pathogens."
                    )

            except Exception as error:
                st.error(
                    f"Image analysis failed: {error}"
                )


# ============================================================
# 13. SPOILAGE RISK ESTIMATOR
# ============================================================

elif page == "Spoilage Risk Estimator":

    st.title("📊 Spoilage Risk Estimator")

    st.write(
        "Estimate a model-based risk score using storage-related inputs."
    )

    if risk_model is None:
        st.error(risk_error)

    else:

        with st.form("risk_form"):

            fruit = st.selectbox(
                "Select fruit",
                ["Apple", "Banana", "Orange"]
            )

            days = st.number_input(
                "Days since purchase",
                min_value=0,
                max_value=60,
                value=3
            )

            temperature = st.number_input(
                "Storage temperature (°C)",
                min_value=-5.0,
                max_value=50.0,
                value=25.0,
                step=0.5
            )

            humidity = st.slider(
                "Storage humidity (%)",
                min_value=0,
                max_value=100,
                value=60
            )

            submitted = st.form_submit_button(
                "Estimate Risk",
                type="primary"
            )

        if submitted:

            try:

                result = predict_spoilage_risk(
                    fruit,
                    days,
                    temperature,
                    humidity
                )

                st.divider()

                st.subheader("Estimated Risk")

                st.metric(
                    "Risk Probability",
                    f"{result['probability'] * 100:.1f}%"
                )

                if result["level"] == "Low":
                    st.success(
                        f"Risk category: {result['level']}"
                    )

                elif result["level"] == "Moderate":
                    st.warning(
                        f"Risk category: {result['level']}"
                    )

                else:
                    st.error(
                        f"Risk category: {result['level']}"
                    )

                st.write(result["explanation"])

                st.info(
                    "Research limitation: this model was trained using "
                    "synthetic proxy labels, not laboratory-measured "
                    "spoilage observations. Its output is educational "
                    "and is not a validated shelf-life or food-safety "
                    "prediction."
                )

            except Exception as error:
                st.error(
                    f"Risk estimation failed: {error}"
                )


# ============================================================
# 14. GENAI FOOD ASSISTANT
# ============================================================

elif page == "GenAI Food Assistant":

    st.title("🤖 GenAI Food Assistant")

    st.write(
        "Ask questions about fruit freshness, storage conditions "
        "and food handling."
    )

    api_key, selected_model = get_openrouter_settings()

    if not api_key:
        st.warning(
            "OpenRouter API key not found. Add "
            "OPENROUTER_API_KEY to Streamlit Cloud Secrets."
        )

    st.caption(
        f"AI model: {selected_model}"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask FreshCheck AI a question..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("FreshCheck AI is thinking..."):

                try:

                    answer = ask_openrouter(
                        question,
                        st.session_state.chat_history[:-1]
                    )

                    st.markdown(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:

                    st.error(
                        f"GenAI request failed: {error}"
                    )

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================
# 15. FRUIT INVENTORY
# ============================================================

elif page == "Fruit Inventory":

    st.title("📦 Fruit Inventory Manager")

    st.write(
        "Add fruits to your inventory and track their expiry dates."
    )

    with st.form("inventory_form"):

        fruit = st.selectbox(
            "Fruit",
            ["Apple", "Banana", "Orange"]
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=10000,
            value=1
        )

        purchase_date = st.date_input(
            "Purchase date",
            value=date.today()
        )

        expiry_date = st.date_input(
            "Expected expiry date",
            value=date.today()
        )

        notes = st.text_area(
            "Notes (optional)"
        )

        add_item = st.form_submit_button(
            "Add to Inventory",
            type="primary"
        )

    if add_item:

        if expiry_date < purchase_date:
            st.error(
                "Expiry date cannot be earlier than purchase date."
            )

        else:

            add_inventory_item(
                fruit,
                int(quantity),
                purchase_date,
                expiry_date,
                notes
            )

            st.success(
                "Fruit added to inventory."
            )

            st.rerun()

    st.divider()

    st.subheader("Current Inventory")

    inventory = get_inventory()

    if not inventory:
        st.info(
            "Your inventory is empty. Add a fruit item above."
        )

    else:

        st.dataframe(
            inventory,
            use_container_width=True,
            hide_index=True
        )

        item_ids = [
            item["ID"] for item in inventory
        ]

        selected_id = st.selectbox(
            "Select an inventory entry to delete",
            item_ids
        )

        if st.button(
            "Delete Selected Entry",
            type="secondary"
        ):

            delete_inventory_item(
                int(selected_id)
            )

            st.success(
                "Inventory entry deleted."
            )

            st.rerun()

        st.subheader("Expiry Status")

        today = date.today()

        for item in inventory:

            expiry = date.fromisoformat(
                item["Expiry Date"]
            )

            if expiry < today:
                st.error(
                    f"{item['Fruit']} — expired on {expiry}"
                )

            elif expiry == today:
                st.warning(
                    f"{item['Fruit']} — due today"
                )

            else:
                st.success(
                    f"{item['Fruit']} — expected expiry: {expiry}"
                )


# ============================================================
# 16. ANALYTICS AND CSV EXPORT
# ============================================================

elif page == "Analytics":

    st.title("📈 FreshCheck AI Analytics")

    inventory = get_inventory()

    if not inventory:

        st.info(
            "Add fruit entries to the inventory to view analytics."
        )

    else:

        fruit_counts = {}

        total_quantity = 0

        for item in inventory:

            fruit_name = item["Fruit"]

            quantity = item["Quantity"]

            fruit_counts[fruit_name] = (
                fruit_counts.get(fruit_name, 0) + quantity
            )

            total_quantity += quantity

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Inventory Quantity",
            total_quantity
        )

        col2.metric(
            "Number of Inventory Entries",
            len(inventory)
        )

        st.subheader("Fruit Quantity Distribution")

        chart_data = {
            "Fruit": list(fruit_counts.keys()),
            "Quantity": list(fruit_counts.values())
        }

        st.bar_chart(
            chart_data,
            x="Fruit",
            y="Quantity"
        )

        st.subheader("Inventory Data")

        st.dataframe(
            inventory,
            use_container_width=True,
            hide_index=True
        )

        csv_buffer = io.StringIO()

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=inventory[0].keys()
        )

        writer.writeheader()
        writer.writerows(inventory)

        st.download_button(
            label="Download Inventory CSV",
            data=csv_buffer.getvalue(),
            file_name="freshcheck_inventory.csv",
            mime="text/csv"
        )


# ============================================================
# 17. MODEL INFORMATION AND DIAGNOSTICS
# ============================================================

elif page == "Model Information":

    st.title("🔍 Model Information & Diagnostics")

    st.subheader("CNN Fruit Freshness Model")

    if cnn is None:

        st.error(cnn_error)

    else:

        st.success("CNN model loaded successfully.")

        st.write("**Model file:**", str(CNN_PATH))
        st.write("**Input shape:**", str(cnn.input_shape))
        st.write("**Output shape:**", str(cnn.output_shape))

        st.write("**Expected class order:**")

        for index, class_name in enumerate(CLASSES):
            st.write(f"{index}: {class_name}")

        rescaling_layers = []

        for layer in cnn.layers:

            if layer.__class__.__name__ == "Rescaling":
                rescaling_layers.append(layer)

        if rescaling_layers:

            st.success(
                "Rescaling layer detected inside the CNN. "
                "The app passes image pixels in the 0–255 range."
            )

            for layer in rescaling_layers:
                st.write(
                    f"Layer: {layer.name} | "
                    f"Configuration: {layer.get_config()}"
                )

        else:

            st.warning(
                "No top-level Rescaling layer was detected. "
                "Check the model's training preprocessing before "
                "changing inference scaling."
            )

        st.write("**CNN SHA256 fingerprint:**")

        st.code(cnn_fingerprint)

    st.divider()

    st.subheader("Risk Model")

    if risk_model is None:

        st.error(risk_error)

    else:

        st.success("Risk model loaded successfully.")

        st.write("**Model file:**", str(RISK_PATH))

        st.write(
            "**Expected input feature order:** "
            "Fruit code, days since purchase, temperature, humidity"
        )

        if hasattr(risk_model, "classes_"):
            st.write(
                "**Risk model classes:**",
                str(list(risk_model.classes_))
            )

        st.write("**Risk model SHA256 fingerprint:**")

        st.code(risk_fingerprint)

    st.divider()

    st.subheader("Deployment Checks")

    st.write(
        "Confirm that your GitHub repository contains these exact files:"
    )

    st.code(
        """
FreshCheck_AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    ├── freshness_cnn.keras
    └── spoilage_risk.joblib
        """
    )

    st.caption(
        "The CNN class order in this app must match the order "
        "used when the model was trained."
    )


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.caption(
    "FreshCheck AI | Educational prototype combining "
    "CNN, Machine Learning, Generative AI and SQLite."
)

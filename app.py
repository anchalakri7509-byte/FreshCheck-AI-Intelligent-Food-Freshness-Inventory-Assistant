
from pathlib import Path
from datetime import date
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
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models"
CNN_PATH = MODEL_DIR / "freshness_cnn.keras"
RISK_PATH = MODEL_DIR / "spoilage_risk.joblib"
DB_PATH = ROOT / "freshcheck_inventory.db"

IMG_SIZE = (224, 224)

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

st.set_page_config(
    page_title="FreshCheck AI",
    page_icon="🍎",
    layout="wide"
)


# ============================================================
# 2. CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #176b45;
    }

    .subtitle {
        color: #667085;
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
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. MODEL FILE HASH
# ============================================================

def get_file_hash(path):

    sha = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha.update(chunk)

    return sha.hexdigest()


def get_model_fingerprint(path):

    if not path.exists():
        return "missing"

    return get_file_hash(path)


cnn_fingerprint = get_model_fingerprint(CNN_PATH)
risk_fingerprint = get_model_fingerprint(RISK_PATH)


# ============================================================
# 4. LOAD CNN
# ============================================================

@st.cache_resource(show_spinner="Loading CNN model...")
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
                f"Expected {len(CLASSES)} output classes, "
                f"but model has {model.output_shape[-1]}."
            )

        if len(model.input_shape) != 4:
            return None, "Unexpected CNN input dimensions."

        if model.input_shape[-1] != 3:
            return None, "CNN does not expect RGB images."

        return model, None

    except Exception as error:
        return None, f"CNN loading failed: {error}"


# ============================================================
# 5. LOAD RISK MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading risk model...")
def load_risk_model(model_path, fingerprint):

    if not Path(model_path).exists():
        return None, f"Risk model not found: {model_path}"

    try:
        import joblib

        model = joblib.load(model_path)

        return model, None

    except Exception as error:
        return None, f"Risk model loading failed: {error}"


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
            cnn_error or "CNN model unavailable."
        )

    # Correct orientation and convert to RGB
    image = ImageOps.exif_transpose(uploaded_image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    # IMPORTANT:
    # The trained model already contains Rescaling(1./255).
    # Do not divide the image array by 255 here.

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
        raise ValueError("Unexpected CNN output dimensions.")

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASSES[predicted_index]

    fruit, condition = CLASS_DETAILS[predicted_class]

    scores = []

    for index, class_name in enumerate(CLASSES):

        score_fruit, score_condition = CLASS_DETAILS[class_name]

        scores.append({
            "class_name": class_name,
            "label": f"{score_fruit} — {score_condition}",
            "probability": float(predictions[index])
        })

    return {
        "class_name": predicted_class,
        "fruit": fruit,
        "condition": condition,
        "confidence": float(predictions[predicted_index]),
        "scores": scores,
        "image_shape": image_array.shape,
        "pixel_min": float(image_array.min()),
        "pixel_max": float(image_array.max())
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
            risk_error or "Risk model unavailable."
        )

    features = np.array([[
        FRUIT_CODES[fruit],
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

    probability = float(probabilities[risk_index])

    if probability < 0.35:
        level = "Low"
    elif probability < 0.70:
        level = "Moderate"
    else:
        level = "High"

    return {
        "probability": probability,
        "level": level
    }


# ============================================================
# 8. SQLITE INVENTORY
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
# 9. OPENROUTER GENAI
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

    model_name = os.getenv(
        "OPENROUTER_MODEL",
        model_name
    )

    return api_key, model_name


def ask_openrouter(question, history):

    api_key, model_name = get_openrouter_settings()

    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is not configured."
        )

    system_prompt = """
You are FreshCheck AI, a helpful food freshness and storage assistant.

Help users understand:
- Fruit storage
- Visible freshness indicators
- Common spoilage signs
- Temperature and humidity effects
- Practical food handling

Safety instructions:
- An image cannot prove food is safe to eat.
- Visible appearance cannot detect all harmful microorganisms.
- Do not claim that the ML risk estimate is a validated food-safety result.
- If food has mold, unusual odor, sliminess or leakage, advise the
  user not to consume it.
- Clearly communicate uncertainty.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(history[-8:])

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
            details = response.json()
        except Exception:
            details = response.text

        raise RuntimeError(
            f"OpenRouter error {response.status_code}: {details}"
        )

    result = response.json()

    return result["choices"][0]["message"]["content"]


# ============================================================
# 10. SIDEBAR
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
    "Educational prototype — not a food-safety certification tool."
)


# ============================================================
# 11. HOME
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

    total_quantity = sum(
        item["Quantity"] for item in inventory
    )

    due_items = 0

    for item in inventory:

        expiry = date.fromisoformat(item["Expiry Date"])

        if expiry <= date.today():
            due_items += 1

    col1, col2, col3 = st.columns(3)

    col1.metric("Inventory Entries", len(inventory))
    col2.metric("Total Fruit Quantity", total_quantity)
    col3.metric("Expired / Due Today", due_items)

    st.divider()

    st.subheader("Explore FreshCheck AI")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>📷 Freshness Scanner</h3>
                Upload fruit images and classify their visible
                appearance using a CNN.
            </div>

            <div class="info-card">
                <h3>🤖 GenAI Food Assistant</h3>
                Ask questions about fruit storage and freshness.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Spoilage Risk Estimator</h3>
                Explore an ML-based risk estimate from storage inputs.
            </div>

            <div class="info-card">
                <h3>📦 Fruit Inventory</h3>
                Record purchases and monitor expiry dates.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
        1. **Deep Learning:** CNN analyses fruit images.
        2. **Machine Learning:** Random Forest estimates a risk score.
        3. **Generative AI:** OpenRouter answers food-storage questions.
        4. **Database:** SQLite stores fruit inventory information.
        """
    )


# ============================================================
# 12. FRESHNESS SCANNER
# ============================================================

elif page == "Freshness Scanner":

    st.title("📷 Fruit Freshness Scanner")

    st.write(
        "Upload an image of an apple, banana or orange."
    )

    if cnn is None:

        st.error(cnn_error)

    else:

        uploaded_file = st.file_uploader(
            "Upload fruit image",
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
                        "Confidence",
                        f"{result['confidence'] * 100:.2f}%"
                    )

                    if result["confidence"] < 0.60:
                        st.warning(
                            "The model has relatively low confidence. "
                            "Try a clear image containing one fruit."
                        )

                    st.subheader("All Six Class Scores")

                    for score in result["scores"]:

                        st.write(
                            f"**{score['class_name']}** — "
                            f"{score['probability'] * 100:.2f}%"
                        )

                        st.progress(
                            max(
                                0.0,
                                min(1.0, score["probability"])
                            )
                        )

                    # Temporary diagnostics for prediction debugging
                    with st.expander("Image prediction diagnostics"):

                        st.write(
                            "Input image shape:",
                            result["image_shape"]
                        )

                        st.write(
                            "Minimum pixel value:",
                            result["pixel_min"]
                        )

                        st.write(
                            "Maximum pixel value:",
                            result["pixel_max"]
                        )

                        st.write(
                            "Preprocessing: RGB, 224 × 224, "
                            "float32, pixel range 0–255."
                        )

                    st.info(
                        "This model classifies visible appearance only. "
                        "It cannot detect invisible pathogens or certify "
                        "food safety."
                    )

            except Exception as error:

                st.error(
                    f"Image prediction failed: {error}"
                )


# ============================================================
# 13. SPOILAGE RISK ESTIMATOR
# ============================================================

elif page == "Spoilage Risk Estimator":

    st.title("📊 Spoilage Risk Estimator")

    st.write(
        "Estimate a model-based risk score using storage inputs."
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
                "Temperature (°C)",
                min_value=-5.0,
                max_value=50.0,
                value=25.0,
                step=0.5
            )

            humidity = st.slider(
                "Humidity (%)",
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

                st.metric(
                    "Estimated Risk Probability",
                    f"{result['probability'] * 100:.1f}%"
                )

                if result["level"] == "Low":
                    st.success("Risk category: Low")

                elif result["level"] == "Moderate":
                    st.warning("Risk category: Moderate")

                else:
                    st.error("Risk category: High")

                st.info(
                    "This model was trained using synthetic proxy labels, "
                    "not laboratory-measured spoilage observations. "
                    "It is an educational estimate, not a validated "
                    "shelf-life or food-safety prediction."
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
        "Ask questions about fruit freshness, storage and food handling."
    )

    api_key, model_name = get_openrouter_settings()

    if not api_key:
        st.warning(
            "OpenRouter API key not found. Add "
            "OPENROUTER_API_KEY to Streamlit Cloud Secrets."
        )

    st.caption(f"AI model: {model_name}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask FreshCheck AI..."
    )

    if question:

        history = st.session_state.chat_history.copy()

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = ask_openrouter(
                        question,
                        history
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

        submitted = st.form_submit_button(
            "Add to Inventory",
            type="primary"
        )

    if submitted:

        if expiry_date < purchase_date:

            st.error(
                "Expiry date cannot be before purchase date."
            )

        else:

            add_inventory_item(
                fruit,
                int(quantity),
                purchase_date,
                expiry_date,
                notes
            )

            st.success("Fruit added successfully.")
            st.rerun()

    st.divider()

    inventory = get_inventory()

    st.subheader("Current Inventory")

    if not inventory:

        st.info("Your inventory is empty.")

    else:

        st.dataframe(
            inventory,
            use_container_width=True,
            hide_index=True
        )

        selected_id = st.selectbox(
            "Select entry to delete",
            [item["ID"] for item in inventory]
        )

        if st.button("Delete Selected Entry"):

            delete_inventory_item(int(selected_id))

            st.success("Entry deleted.")
            st.rerun()

        st.subheader("Expiry Status")

        for item in inventory:

            expiry = date.fromisoformat(item["Expiry Date"])

            if expiry < date.today():

                st.error(
                    f"{item['Fruit']} — expired on {expiry}"
                )

            elif expiry == date.today():

                st.warning(
                    f"{item['Fruit']} — due today"
                )

            else:

                st.success(
                    f"{item['Fruit']} — expected expiry: {expiry}"
                )


# ============================================================
# 16. ANALYTICS
# ============================================================

elif page == "Analytics":

    st.title("📈 FreshCheck AI Analytics")

    inventory = get_inventory()

    if not inventory:

        st.info(
            "Add fruit entries to view inventory analytics."
        )

    else:

        fruit_counts = {}

        for item in inventory:

            fruit = item["Fruit"]

            fruit_counts[fruit] = (
                fruit_counts.get(fruit, 0) + item["Quantity"]
            )

        total_quantity = sum(fruit_counts.values())

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Fruit Quantity",
            total_quantity
        )

        col2.metric(
            "Inventory Entries",
            len(inventory)
        )

        st.subheader("Fruit Quantity Distribution")

        st.bar_chart(
            {
                "Fruit": list(fruit_counts.keys()),
                "Quantity": list(fruit_counts.values())
            },
            x="Fruit",
            y="Quantity"
        )

        st.subheader("Inventory Records")

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
            "Download Inventory CSV",
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

        # TensorFlow version diagnostic
        try:

            import tensorflow as tf

            st.write(
                "**Deployed TensorFlow version:**",
                tf.__version__
            )

        except Exception as error:

            st.error(
                f"Could not read TensorFlow version: {error}"
            )

        st.divider()

        st.write("**Expected CNN class order:**")

        for index, class_name in enumerate(CLASSES):

            st.write(
                f"{index} = {class_name}"
            )

        st.divider()

        st.write("**CNN preprocessing layers:**")

        rescaling_found = False

        for layer in cnn.layers:

            if layer.__class__.__name__ == "Rescaling":

                rescaling_found = True

                st.write({
                    "layer_name": layer.name,
                    "layer_type": layer.__class__.__name__,
                    "configuration": layer.get_config()
                })

        if rescaling_found:

            st.success(
                "Rescaling layer detected. The app passes "
                "pixel values in the 0–255 range."
            )

        else:

            st.warning(
                "No top-level Rescaling layer detected. "
                "Do not change preprocessing until the model "
                "configuration has been checked."
            )

        st.divider()

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
            "**Input feature order:** Fruit code, days since purchase, "
            "temperature, humidity"
        )

        if hasattr(risk_model, "classes_"):

            st.write(
                "**Risk model classes:**",
                str(list(risk_model.classes_))
            )

        st.write("**Risk model SHA256 fingerprint:**")

        st.code(risk_fingerprint)

    st.divider()

    st.subheader("Deployment File Structure")

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
        """,
        language="text"
    )

    st.caption(
        "The deployed CNN class order must match the class order "
        "used during model training."
    )


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.caption(
    "FreshCheck AI | CNN + Machine Learning + Generative AI + SQLite"
)

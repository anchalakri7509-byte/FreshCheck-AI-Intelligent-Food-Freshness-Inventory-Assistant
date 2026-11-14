# 🍎 FreshCheck AI – Intelligent Food Freshness & Inventory Assistant

FreshCheck AI is an AI-powered food freshness and inventory management application that combines **Deep Learning, Machine Learning, and Generative AI** to help users assess the visible freshness of fruits, estimate spoilage risk, manage food inventory, and receive intelligent food-storage recommendations.

The application provides a simple, interactive interface built with Streamlit, making AI-powered food management accessible through a web browser.

## 🚀 Key Features

### 1. 🍊 AI Freshness Scanner
- Upload images of apples, bananas, and oranges.
- Classify fruits into six categories:
  - Fresh Apples
  - Rotten Apples
  - Fresh Bananas
  - Rotten Bananas
  - Fresh Oranges
  - Rotten Oranges
- Display the predicted category and confidence score.
- Uses a Convolutional Neural Network (CNN).

### 2. 📊 ML-Based Spoilage Risk Estimation
- Estimates spoilage risk using:
  - Fruit type
  - Days since purchase
  - Temperature
  - Humidity
- Uses a Random Forest Classifier.
- Provides an educational risk estimate based on synthetic training data.

### 3. 🤖 AI Food Assistant
- Ask questions about food storage and freshness.
- Get AI-generated recommendations and explanations.
- Powered by a Large Language Model (LLM) through OpenRouter.

### 4. 🥗 Food Inventory Management
- Add food items to your inventory.
- Record purchase and expiry dates.
- View stored food items.
- Delete items when they are no longer needed.

### 5. 📈 Analytics & Reports
- View inventory information.
- Identify items approaching their expiry dates.
- Generate and download inventory reports in CSV format.
- Receive AI-assisted inventory suggestions.

## 🧠 Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Web App | Streamlit |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn |
| CNN Architecture | Convolutional Neural Network |
| ML Algorithm | Random Forest Classifier |
| Generative AI | OpenRouter API |
| Language Model | Configured OpenRouter-supported LLM |
| Image Processing | Pillow (PIL), NumPy |
| Database | SQLite |
| Model Serialization | Joblib |
| Version Control | Git and GitHub |

## 🏗️ Project Architecture

```text
                  USER
                   |
                   v
             STREAMLIT APP
                   |
       +-----------+-----------+
       |           |           |
       v           v           v
  Freshness    Spoilage     AI Food
   Scanner      Risk        Assistant
       |        Model           |
       v           |            v
   CNN Model       v        OpenRouter
  (TensorFlow) Random Forest     API
       |       (Scikit-learn)     |
       +-----------+-------------+
                   |
                   v
          FOOD INVENTORY
                   |
                   v
               SQLite
                   |
                   v
          Analytics & Reports
```

## 📂 Project Structure

```text
FreshCheck_AI/
│
├── app.py
├── train_cnn.py
├── train_risk_model.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── fruit_dataset/
│   │   ├── train/
│   │   │   ├── freshapples/
│   │   │   ├── freshbanana/
│   │   │   ├── freshoranges/
│   │   │   ├── rottenapples/
│   │   │   ├── rottenbanana/
│   │   │   └── rottenoranges/
│   │   │
│   │   └── test/
│   │       ├── freshapples/
│   │       ├── freshbanana/
│   │       ├── freshoranges/
│   │       ├── rottenapples/
│   │       ├── rottenbanana/
│   │       └── rottenoranges/
│   │
│   └── freshcheck.db
│
├── models/
│   ├── freshness_cnn.keras
│   └── spoilage_risk.joblib
│
└── .streamlit/
    └── secrets.toml
```

*Note: The database is created automatically by the application if it does not exist. The dataset, trained model files, and API secrets should be handled according to the repository setup and hosting requirements. Do not upload private API keys or sensitive data to GitHub.*

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/FreshCheck_AI.git
cd FreshCheck_AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure OpenRouter API Key

Create a `.streamlit/secrets.toml` file:

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key"
```

Alternatively, configure the API key as an environment variable if supported by your application.

**Important:** Never commit `secrets.toml` or your API key to GitHub. Add the secrets file to `.gitignore` and configure the key securely on your hosting platform.

### Step 5: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🧪 Model Development

### Deep Learning – CNN

The CNN model is trained using fruit images organized into six classes.

**Training script:**

```bash
python train_cnn.py
```

The trained model is saved as:

```text
models/freshness_cnn.keras
```

The model performs image classification to identify visible freshness categories.

### Machine Learning – Random Forest

The spoilage-risk model uses fruit type, days since purchase, temperature, and humidity as input features.

**Training script:**

```bash
python train_risk_model.py
```

The trained model is saved as:

```text
models/spoilage_risk.joblib
```

**Important:** The current risk model uses synthetic proxy labels for demonstration. Its predictions are not validated against real-world food spoilage observations.

## 🎯 Project Objectives

- Develop an AI-powered fruit freshness classification system.
- Apply CNN-based deep learning to image classification.
- Implement a machine learning model for educational spoilage-risk estimation.
- Integrate a Generative AI assistant for food-storage guidance.
- Build a digital food inventory management system.
- Provide useful inventory analytics and downloadable reports.
- Demonstrate how multiple AI technologies can work together in a practical application.

## 🌍 Potential Applications

- Household food inventory management.
- Grocery and retail inventory assistance.
- Food-storage awareness and education.
- Educational demonstrations of AI in food technology.
- Smart kitchen and food management applications.

## ⚠️ Limitations

- Freshness classification is based on visible image features only.
- The model supports only apples, bananas, and oranges in six predefined categories.
- Image predictions may be affected by lighting, image quality, background, and fruit appearance.
- The spoilage-risk model currently uses synthetic training labels and requires real-world validation.
- AI-generated storage recommendations should be checked against reliable food-safety guidance.
- The application cannot detect invisible pathogens, toxins, or guarantee that food is safe to eat.

**FreshCheck AI is an educational decision-support tool, not a substitute for professional food-safety assessment.**

## 🔮 Future Enhancements

- Expand image classification to more fruit and vegetable categories.
- Train the risk model using real-world food storage and spoilage datasets.
- Improve model performance through transfer learning and model evaluation.
- Add user authentication and personalized inventory management.
- Integrate barcode scanning and automated product entry.
- Add expiry notifications and smart reminders.
- Deploy the application on a public cloud platform.

## 👩‍💻 Learning Outcomes

Through this project, I explored:

- CNN model development and image classification.
- Machine learning with Random Forest.
- Model training, evaluation, and serialization.
- Generative AI integration using OpenRouter.
- API integration and prompt engineering.
- Streamlit application development.
- SQLite database operations.
- Integration of Deep Learning, Machine Learning, and Generative AI into one application.

## 👩‍🔬 Project Summary

FreshCheck AI demonstrates how Deep Learning, Machine Learning, and Generative AI can be integrated into a single application to support food freshness awareness and inventory management.

The project focuses on building a practical, user-friendly system while exploring the real-world possibilities and limitations of AI in food technology.

---

**Built with Python, Deep Learning, Machine Learning, and Generative AI.** 🍎
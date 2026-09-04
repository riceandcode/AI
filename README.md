# ♻️ AI-Based Waste Classification System

### Environmental Studies — CA1 Project

## Problem Statement

Improper solid waste segregation at the source is one of the leading causes of inefficient recycling and increased landfill burden. Most households and institutions dispose of waste without separating it into recyclable categories, making downstream recycling costly, slow, and often impractical.

This project addresses that gap by using a machine learning image classification model to automatically identify the category of a waste item (Plastic, Paper, Metal, or Organic) from a photo, and provide the user with a clear, actionable disposal instruction — encouraging correct segregation at the point of disposal, where it matters most.

## Key Features

- **Dual input modes** — classify waste using either a live webcam capture or an uploaded image file (JPG/PNG)
- **Real-time ML classification** — powered by a custom-trained image classification model, returning results in seconds
- **Confidence scoring** — displays the model's confidence percentage alongside its prediction, so users can judge result reliability
- **Tailored disposal guidance** — each predicted category comes with a specific, practical disposal or recycling tip
- **Simple, accessible interface** — built with Streamlit for a clean, no-friction user experience requiring no technical knowledge to operate

## Tech Stack

| Component          | Technology                          |
|---------------------|--------------------------------------|
| Model Training       | Google Teachable Machine (Transfer Learning) |
| ML Framework         | TensorFlow / Keras                  |
| Web Interface         | Streamlit                           |
| Image Processing      | Pillow (PIL)                        |
| Numerical Computation | NumPy                               |
| Language              | Python 3.11                         |

## Local Setup & Run Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment
This keeps project dependencies isolated from your system Python — required, since this project depends on a specific TensorFlow version for model compatibility.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Required Files
Make sure the following files are present in the project root alongside `app.py`:
- `keras_model.h5` — the trained classification model
- `labels.txt` — the class label mapping

## Team Contributions

| Name | Roll No. | Reg. No. | Contribution |
|------|----------|----------|--------------|
| [Name] | [Roll No.] | [Reg. No.] | [e.g., Model training, dataset collection] |
| [Name] | [Roll No.] | [Reg. No.] | [e.g., Streamlit app development, UI design] |
| [Name] | [Roll No.] | [Reg. No.] | [e.g., Documentation, testing, report writing] |

## Course Outcome Alignment

- **CO1 — Environmental Issues:** This project directly engages with the environmental issue of solid waste mismanagement by demonstrating how AI-assisted classification can reduce contamination in recycling streams and support more sustainable waste handling practices.
- **CO3 — Pollution Control Policies/Practices:** The system operationalizes a practical, low-cost pollution control practice — accurate source segregation — which is a foundational requirement for effective municipal solid waste management policy and reduces the volume of recyclable material lost to landfill or incineration.

---

*This project was developed as part of the Environmental Studies coursework, demonstrating the application of AI/ML techniques to real-world environmental sustainability challenges.*

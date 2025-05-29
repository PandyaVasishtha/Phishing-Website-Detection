import gradio as gr
import pickle
import pandas as pd  # For the examples, though not strictly needed for prediction function
import numpy as np  # For the examples, though not strictly needed for prediction function

# --- Code from your notebook (or ensure these are available if in a new file) ---
# If you're running this in a new file, you might need to define
# the tokenizer or ensure its class is available if it's custom and pickled.
# However, since it's part of a scikit-learn pipeline, pickle should handle it.
from nltk.tokenize import RegexpTokenizer  # Used in the pipeline

# --- End of code from your notebook ---

# 1. Load the pickled model (pipeline)
MODEL_PATH = r"C:\Users\vaspa\OneDrive\Desktop\Phishing_new\phishing.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        loaded_model_pipeline = pickle.load(f)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except FileNotFoundError:
    print(
        f"Error: '{MODEL_PATH}' not found. Make sure the model is in the same directory."
    )
    loaded_model_pipeline = None
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model_pipeline = None


# 2. Create a prediction function
def predict_phishing_url(url_text):
    if loaded_model_pipeline is None:
        return "Error: Model not loaded."
    if not url_text or not url_text.strip():
        return {"Error": 1.0, "Info": 0.0}  # Return a dictionary for gr.Label

    try:
        # The pipeline expects an iterable (like a list) of documents
        prediction = loaded_model_pipeline.predict([url_text])
        # predict() returns an array, e.g., array(['good'], dtype=object)
        result = prediction[0]

        # Format for gr.Label with confidences (or just distinct labels)
        if result.lower() == "good":
            return {"Good": 1.0, "Bad": 0.0}
        else:
            return {"Bad": 1.0, "Good": 0.0}

    except Exception as e:
        return {f"Error: {str(e)}": 1.0}


# 3. Set up the Gradio interface
iface = gr.Interface(
    fn=predict_phishing_url,
    inputs=gr.Textbox(lines=1, placeholder="Enter URL here...", label="URL to Check"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Result"),
    title="Phishing URL Detector",
    description="Enter a URL to check if it's likely a phishing site ('bad') or a legitimate site ('good'). Based on a Logistic Regression model.",
    examples=[
        ["youtube.com/"],
        ["yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php"],
        ["google.com/search?q=gradio"],
        ["paypal-servis-center.com/login"],
    ],
    allow_flagging="never",  # You can set to "auto" or "manual" if you want flagging
)

# Launch the interface (if running as a script)
# If in a Jupyter notebook, this will often launch in a new tab or inline.
if __name__ == "__main__":
    if loaded_model_pipeline:  # Only launch if model loaded successfully
        print("Launching Gradio interface...")
        iface.launch()
    else:
        print("Gradio interface not launched due to model loading error.")

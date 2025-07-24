import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model("braintumor96.h5")

# Class labels in order
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define prediction function
def predict_tumor(img):
    # Resize and normalize image
    img = img.resize((150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence_scores = {class_names[i]: float(round(prediction[i], 3)) for i in range(4)}

    return predicted_class, confidence_scores

# Create Gradio interface
interface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil", label="Upload Brain MRI"),
    outputs=[
        gr.Label(label="Predicted Tumor Type"),
        gr.Label(label="Prediction Confidence (Probability)")
    ],
    title="🧠 Brain Tumor Classifier",
    description="Upload a brain MRI image to classify into Glioma, Meningioma, No Tumor, or Pituitary tumor.",
    allow_flagging="never"
)

# Launch app
interface.launch()

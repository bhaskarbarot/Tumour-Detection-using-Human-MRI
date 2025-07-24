# Tumour-Detection-using-Human-MRI


📌 Project Overview
This project aims to automatically detect and classify brain tumors from MRI images using deep learning techniques. The model leverages Convolutional Neural Networks (CNNs) to classify MRI scans into different categories, helping radiologists and medical professionals with early and accurate diagnosis.

✅ Features
Input MRI images and predict tumor presence.

Multi-class classification (e.g., glioma, meningioma, pituitary tumor, no tumor).

Built using TensorFlow/Keras and OpenCV.

Preprocessing pipeline for image resizing, normalization, and augmentation.

Achieved high validation accuracy.

Simple Streamlit web app for easy user interaction.

📂 Dataset
Source: Kaggle - Brain MRI Images for Brain Tumor Detection

Contains MRI images categorized as:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

⚙️ Tech Stack
Language: Python

Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Streamlit

Tools: Jupyter Notebook / VSCode

🚀 How to Run Locally
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
2️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
📊 Model Architecture
Pre-trained VGG16 / ResNet50 used for transfer learning.

Fine-tuned with custom dense layers.

Categorical Cross-Entropy loss.

Optimizer: Adam.

Metrics: Accuracy.

📈 Results
Training accuracy: ~98%

Validation accuracy: ~95%

Tested on unseen MRI images.

📸 Screenshots
Upload MRI	Prediction Result
Add a sample MRI input image	Add a screenshot of output with predicted label

📌 Project Structure
kotlin
Copy
Edit
brain-tumor-detection/
├── data/
│   ├── Training/
│   ├── Testing/
├── models/
│   └── vgg16_best_model.h5
├── app.py
├── requirements.txt
├── README.md
└── notebooks/
    └── EDA_and_Model.ipynb
🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

✨ Acknowledgements
Dataset by Navoneel Chakrabarty (Kaggle)

TensorFlow & Keras for model building

Streamlit for web deployment


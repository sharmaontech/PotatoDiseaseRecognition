# 🥔 Potato Disease Recognition using Deep Learning

This project is a deep learning-based image classification system that identifies diseases in potato leaves. Using Convolutional Neural Networks (CNNs), the model classifies images into three categories: **Healthy**, **Early Blight**, and **Late Blight**. This system can help farmers and agricultural stakeholders detect potato diseases early, enabling timely and effective treatment.

## 📌 Table of Contents

* [Overview](#-overview)
* [Dataset](#-dataset)
* [Model Architecture](#-model-architecture)
* [Installation](#-installation)
* [Usage](#-usage)
* [Results](#-results)
* [Streamlit Interface](#-streamlit-interface)
* [Project Structure](#-project-structure)
* [Future Work](#-future-work)
* [License](#-license)

---

## 🔍 Overview

Potato crops are vulnerable to various diseases, especially **Early Blight** and **Late Blight**, which significantly affect crop yield and quality. This project leverages computer vision to automate the detection of these diseases from leaf images.

* **Type**: Image Classification
* **Model**: Custom CNN (built with TensorFlow/Keras)
* **Goal**: Classify potato leaf images into:

  * Healthy
  * Early Blight
  * Late Blight

---

## 📂 Dataset

We use the [**Potato Leaf Disease Dataset**](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) from the PlantVillage repository, hosted on Kaggle. It contains over 2,000 labeled images in the following categories:

* **Healthy**
* **Early Blight**
* **Late Blight**

The images are RGB, captured in various lighting conditions, and are resized to a uniform size during preprocessing.

---

## 🧠 Model Architecture

The model is a **Convolutional Neural Network (CNN)** designed from scratch using TensorFlow/Keras.

**Architecture Summary:**

* Input: 256x256 RGB image
* Conv2D → ReLU → MaxPooling
* Conv2D → ReLU → MaxPooling
* Dropout (to prevent overfitting)
* Flatten
* Dense (fully connected layers)
* Output Layer: 3 neurons (Softmax)

Model training and saving logic can be found in `training.py`.

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/PadamArora/PotatoDiseaseRecognition.git
   cd PotatoDiseaseRecognition
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) and extract it into a folder named `dataset/`.

---

## 🚀 Usage

To **train the model**, run:

```bash
python training.py
```

To **test the model** or run predictions using test images:

```bash
python testing.py
```

---

## 🔢 Streamlit Interface

This project includes a Streamlit web interface (`app.py`) to interactively test the trained model.

### To launch the app:

```bash
streamlit run app.py
```

### Features:

* Upload a potato leaf image (.jpg, .png)
* Get real-time predictions
* Displays the uploaded image alongside predicted class (Healthy / Early Blight / Late Blight)

Perfect for showcasing the model’s capabilities in a user-friendly UI.

---

## 📊 Results

After training for \~10 epochs, the model achieved:

* **Accuracy**: \~96% on validation set
* **Loss**: Consistently decreasing, with minimal overfitting
* **Confusion Matrix**: High precision/recall across all classes

> *Model evaluation and metrics logging are performed in `testing.py`.*

---

## 📁 Project Structure

```
PotatoDiseaseRecognition/
│
├── app.py                   # Streamlit interface
├── dataset/                 # Images (downloaded from Kaggle)
├── samples/                 # Sample test images
├── training.py              # Model training script
├── testing.py               # Model testing / evaluation script
├── 1.keras                # Trained model weights
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚧 Future Work

* Improve model performance using data augmentation and transfer learning
* Extend support to detect more plant diseases
* Build a mobile app version
* Integrate real-time webcam prediction for field usability

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 🙌 Acknowledgements

* [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
* TensorFlow & Keras Documentation
* Agricultural AI research papers for guidance

---

> Made with ❤️ by [Padam Arora](https://github.com/PadamArora)

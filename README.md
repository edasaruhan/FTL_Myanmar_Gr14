# ♻️ Waste Classification System
**Group 14 — SDG Project**
- **Khin Yadanar Hlaing**
- **Myo Myat Htun**

---

## 🌍 Project Idea
Improper waste disposal causes pollution, resource loss, and health issues.  
This project aims to develop an **automated computer vision-based waste classification system** with **two levels of classification**:

1. **Recyclable vs Non-recyclable** (Binary Classification)
2. **Type of waste** (Glass, Metal, Organic, Paper, Plastic)

The system enhances waste management efficiency, promotes recycling, and ensures proper segregation of materials.

---

## 🎯 Relevance to SDGs
- **SDG 3 – Good Health & Well-being:** Reduces pollution and environmental hazards from improper disposal.
- **SDG 9 – Industry, Innovation & Infrastructure:** Promotes sustainable waste management innovation.
- **SDG 11 – Sustainable Cities & Communities:** Enables smart municipal waste handling.
- **SDG 12 – Responsible Consumption & Production:** Encourages recycling and resource reuse.
- **SDG 13 – Climate Action:** Reduces landfill emissions through better waste segregation.

---

## 📚 Literature References
1. **Trash Detection: Advanced Classification of Waste Materials Using ML Techniques (IEEE, 2024)** CNN-based model trained on five waste categories (paper, plastic, glass, metal, cardboard).  
   Achieved **93.39% accuracy**, demonstrating CNN’s potential for automated waste classification.

2. **An Automated Waste Classification System Using Deep Learning Techniques (Knowledge-Based Systems, 2025)** Proposed a three-stage lightweight CNN achieving **96% accuracy** on the TriCascade dataset,  
   showing real-time sorting capability and industrial scalability.

---

## 🗂️ Dataset
Publicly available datasets were used for model training and testing:
- [Waste Segregation Image Dataset – Kaggle](https://www.kaggle.com/datasets/aashidutt3/waste-segregation-image-dataset)
- [Waste Classification Dataset – Kaggle](https://www.kaggle.com/datasets/phenomsg/waste-classification)
- [Garbage Classification (12 classes) – Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

**Data Format:** JPEG/PNG images labeled into multiple classes.  
[cite_start]**Preprocessing:** Image resizing ($224 \times 224$), normalization ($1./255$), augmentation (rotation, shifts, flips), and label encoding[cite: 37, 267].

---

## 🧠 Approach
**Technique:** Deep Learning (Convolutional Neural Networks)

- **Model Architecture:** Transfer Learning with **VGG16**.
- **Classification Strategy:** Two specialized models trained on the VGG16 base:
    1. **Binary Model:** Determines if the item is Recyclable or Non-recyclable.
    2. **Multi-class Model:** Identifies the specific material type (Glass, Metal, Organic, Paper, Plastic).
- **Why Deep Learning:** Automatically extracts image features (edges, textures) under varying lighting conditions.
- **Deployment:** Web application hosted on **Streamlit Cloud** for real-time user interaction.

---

## 🚀 Future Work
- **Hazardous Waste Classification:** Develop a third model to identify hazardous materials (e.g., batteries, chemicals), which was scoped out of the current phase due to data and time constraints.
- **Object Detection:** Implement real-time object detection (e.g., YOLO) for video streams.
- **Edge Deployment:** Port models to edge devices (e.g., Raspberry Pi) for smart bins.
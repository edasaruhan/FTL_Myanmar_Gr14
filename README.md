# ♻️ Waste Classification System
**Group 14 — SDG Project**
- **Khin Yadanar Hlaing**
- **Myo Myat Htun**
- **Aye Nandar Bo**
---

## 🌍 Project Idea
Improper waste disposal causes pollution, resource loss, and health issues.  
This project aims to develop an **automated computer vision-based waste classification system** with **three levels of classification**:

1. **Recyclable vs Non-recyclable**  
2. **Type of waste** (paper, plastic, metal, glass, organic, etc.)  
3. **Hazardous waste classification**

The system enhances waste management efficiency, promotes recycling, and ensures safe disposal of hazardous materials.

---

## 🎯 Relevance to SDGs
- **SDG 3 – Good Health & Well-being:** Reduces exposure to hazardous waste  
- **SDG 9 – Industry, Innovation & Infrastructure:** Promotes sustainable waste management innovation  
- **SDG 11 – Sustainable Cities & Communities:** Enables smart municipal waste handling  
- **SDG 12 – Responsible Consumption & Production:** Encourages recycling and resource reuse  
- **SDG 13 – Climate Action:** Reduces landfill emissions through better waste segregation  

---

## 📚 Literature References
1. **Trash Detection: Advanced Classification of Waste Materials Using ML Techniques (IEEE, 2024)**  
   CNN-based model trained on five waste categories (paper, plastic, glass, metal, cardboard).  
   Achieved **93.39% accuracy**, demonstrating CNN’s potential for automated waste classification.

2. **An Automated Waste Classification System Using Deep Learning Techniques (Knowledge-Based Systems, 2025)**  
   Proposed a three-stage lightweight CNN achieving **96% accuracy** on the TriCascade dataset,  
   showing real-time sorting capability and industrial scalability.

---

## 🗂️ Dataset
Publicly available datasets will be used for model training and testing:
- [Waste Segregation Image Dataset – Kaggle](https://www.kaggle.com/datasets/aashidutt3/waste-segregation-image-dataset)
- [Waste Classification Dataset – Kaggle](https://www.kaggle.com/datasets/phenomsg/waste-classification)
- [Garbage Classification (12 classes) – Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- [Roboflow Waste Datasets – Roboflow Universe](https://universe.roboflow.com/search?q=class%3Awaste)

**Data Format:** JPEG/PNG images labeled into multiple classes.  
**Preprocessing:** Image resizing, normalization, augmentation, and label encoding for class balance.

---

## 🧠 Approach
**Technique:** Deep Learning (Convolutional Neural Networks)

- **Model Architecture:** Transfer Learning with **MobileNetV2** (efficient) and **ResNet50** (accurate)  
- **Multi-task Learning:** Three classification heads for hierarchical predictions  
- **Why Deep Learning:** Automatically extracts image features under varying lighting and textures  
- **Advantage:** Outperforms traditional ML in accuracy and feature generalization  
- **Deployment:** Suitable for **edge devices** in waste management facilities  

---



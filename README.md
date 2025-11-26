# 🌱 Saarthi – Soil Crop Recommendation System  
_A Smart Soil-Based Crop Recommendation Model for Maximizing Farmer Yield_

---

## 📌 Overview

**Saarthi** is an intelligent soil–crop recommendation system developed to help farmers increase their yield throughout the year.  
The system predicts the optimal crop for cultivation based on essential **soil nutrients** and **environmental parameters**, using both **Machine Learning (ML)** and **Deep Learning (DL)** techniques.

This project includes:
- CNN-based deep learning model  
- Classical ML models (Decision Tree, Naive Bayes, SVM, Random Forest, KNN)  
- Confusion matrix heatmaps for the Decision Tree model  
- Pickle-based model saving/loading  
- A well-structured ML pipeline  

---

## 🌾 Problem Statement

Farmers face difficulties selecting the right crop because soil conditions vary across regions and seasons. Saarthi analyzes soil nutrient levels and environmental indicators to recommend the most suitable crop, enabling farmers to maximize productivity throughout the year.


---

## 🌍 Social Impact

Saarthi helps farmers make **data-driven crop decisions**, improving yield, reducing farming risks, and increasing economic stability.  
By recommending the right crops for specific soil and climate conditions, it promotes:
- **Sustainable agriculture**  
- **Efficient use of fertilizers and resources**  
- **Improved income and livelihood for farmers**  
- **Reduction in crop failure rates**  
- **Year-round optimized cultivation**  

Saarthi supports rural communities by empowering farmers with intelligent technology for better agricultural outcomes.

---

## 📊 Dataset Description

The model uses agricultural datasets containing **7 key crop-growing parameters**:

| Parameter     | Description |
|---------------|-------------|
| **Nitrogen (N)** | Nutrient responsible for leaf development |
| **Phosphorus (P)** | Supports root and flower formation |
| **Potassium (K)** | Improves disease resistance |
| **Temperature** | Environmental heat measure |
| **Humidity** | Moisture level in air |
| **pH Level** | Soil acidity/alkalinity |
| **Geo-Location** | Region-based agronomic influence |

Datasets used:
- `crop_recommendation.csv`  
- `CropData.csv`  
- `FertilizerData.csv`  

---

## 🧠 Machine Learning & Deep Learning Models (Brief)

### 🔹 **1. Convolutional Neural Network (CNN)**
A deep-learning model capable of capturing complex, non-linear patterns between soil parameters.  
CNN enhances feature extraction and can generalize well across diverse crop types.

### 🔹 **2. Decision Tree**
A tree-based classifier that splits data based on the most informative features.  
- Easy to visualize  
- Useful for interpreting feature importance  
- **Accuracy: 0.9068**

### 🔹 **3. Naive Bayes**
A probability-based classifier assuming feature independence.  
- Works well for multi-class data  
- Fast and efficient  
- **Accuracy: 0.9886**

### 🔹 **4. Support Vector Machine (SVM)**
Classifies data by optimizing a separating hyperplane.  
- Strong for high-dimensional datasets  
- **Accuracy: 0.9772**

### 🔹 **5. Random Forest**
An ensemble of multiple decision trees.  
- Handles overfitting better  
- Stable predictions  
- **Accuracy: 0.9954**

### 🔹 **6. K-Nearest Neighbors (KNN)**
Classifies input based on similarity with its closest neighbors.  
- Effective for pattern-based data  
- **Accuracy: 0.9681**

---

## 📊 Confusion Matrix & Heatmaps

A confusion matrix heatmap was generated for the **Decision Tree model** to evaluate how well the model classifies different crop types.  
The visualization was created using **Matplotlib** and **Seaborn**.

### 🔍 Significance of Confusion Matrix & Heatmap

- **Class-wise Performance Insight**  
  Displays how accurately the model predicts each crop class and identifies which crops are predicted correctly more often.

- **Identifies Misclassifications**  
  Highlights incorrect predictions (off-diagonal values).  
  For example, if the model predicts “rice” instead of “maize,” it becomes visible in this matrix.

- **Visual Clarity Through Heatmap**  
  Color intensity makes it easy to interpret:  
  - Darker shades → higher accurate predictions  
  - Lighter shades → fewer predictions or misclassifications  

- **Highlights Areas for Improvement**  
  Helps decide if certain crops need more training data or if model tuning is required.

- **Better Analysis Than Accuracy Alone**  
  A model may show high accuracy yet fail on specific classes.  
  The confusion matrix reveals these weaknesses clearly, ensuring reliable real-world predictions.

In this project, the Decision Tree confusion matrix helps validate class-wise stability and supports further model optimization.

---

## 🧰 Libraries Used

### **1. Data Processing**
- **pandas** → Load, clean, and analyze data  
- **numpy** → Perform numerical computations  

### **2. ML Algorithms (scikit-learn)**
- DecisionTreeClassifier  
- GaussianNB  
- SVC (Support Vector Machine)  
- RandomForestClassifier  
- KNeighborsClassifier  
- train_test_split  
- accuracy_score  
- confusion_matrix  

### **3. Deep Learning**
- **TensorFlow / Keras** → Build CNN model  

### **4. Visualization**
- **matplotlib** → Plot graphs  
- **seaborn** → Heatmaps & advanced visualizations  

### **5. Model Serialization**
- **pickle** → Save and load trained ML models for deployment without retraining  

---

## 🔄 Project Pipeline (Descriptive Workflow)

### **1. Data Collection & Loading**
Import the agricultural datasets containing soil nutrients, environmental conditions, and crop labels.

### **2. Data Preprocessing**
- Handle missing values  
- Normalize or scale features where required  
- Encode categorical crop labels  
- Merge datasets if needed  

### **3. Exploratory Data Analysis (EDA)**
- Analyze data distribution  
- Visualize nutrient relationships  
- Generate correlation heatmaps  

### **4. Train-Test Split**
Divide the data into training and testing sets to evaluate generalization.

### **5. Model Training**
Train all ML and DL models:  
- CNN  
- Decision Tree  
- Naive Bayes  
- SVM  
- Random Forest  
- KNN  

Each model learns patterns between soil parameters and crop outputs.

### **6. Model Evaluation**
- Compute accuracy for each model  
- Generate confusion matrices  
- Visualize Decision Tree results with heatmaps  

### **7. Model Saving with Pickle**
Save the best-performing models for deployment:
```python
pickle.dump(model, open('crop_model.pkl', 'wb'))

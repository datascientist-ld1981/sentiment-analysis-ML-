# Sentiment Analysis on Skewed Small Dataset: ML Models

## 📄 Project Overview
This project performs **sentiment analysis** on a small, skewed dataset containing text reviews labeled as **positive** and **neutral**. The dataset lacks balance and is heavily skewed towards positive sentiments.  

We evaluate the performance of **three traditional machine learning models**:
1. Naive Bayes  
2. Random Forest  
3. Logistic Regression  

---

## 🔍 Challenges
1. **Small Dataset**: Limited data for training and testing.  
2. **Class Imbalance**: The dataset is heavily skewed towards the **positive** sentiment, with fewer examples of neutral.  
3. **Evaluation**: Metrics such as **precision, recall, and F1-score** are critical to understanding model performance for the underrepresented class.  

---

## 🔧 Tools & Libraries Used
- **Python**
- **Scikit-Learn** (Model implementation and evaluation)
- **Pandas** (Data manipulation)
- **NumPy** (Numerical operations)
- **Matplotlib** & **Seaborn** (Visualization)
- **Google Colab** (Execution environment)

---

## 🛠 Workflow

### 1. **Data Preprocessing**
- **Text Cleaning**:
  - Convert text to lowercase.
  - Remove punctuation, special characters, and stopwords.  
- **Feature Representation**:
  - Convert text into numerical vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency).

### 2. **Model Training**
The following machine learning models are trained on the processed dataset:
1. **Naive Bayes**: Suitable for text classification tasks.
2. **Random Forest**: Ensemble-based decision tree model.
3. **Logistic Regression**: Linear model for classification tasks.

### 3. **Evaluation**
- Metrics used to evaluate model performance:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**

Given the class imbalance, metrics like **precision** and **recall** are emphasized.

---

## 📊 Results Summary

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Naive Bayes          | 72%      | 68%       | 70%    | 69%      |
| Random Forest        | 80%      | 76%       | 70%    | 73%      |
| Logistic Regression  | 78%      | 75%       | 68%    | 71%      |

> Note: Results may slightly vary due to small dataset size and class imbalance.

---

## 💡 Key Insights
1. **Random Forest** outperformed other models on the small, skewed dataset.  
2. **Naive Bayes** performed adequately despite its simplicity, showing robustness for text classification tasks.  
3. **Logistic Regression** provided competitive results, balancing precision and recall well.  
4. The **skewed data** impacted model performance, especially for the minority (neutral) class.  

---

## 🚦 How to Run the Project

1. **Setup Environment**
   - Install required libraries:
     ```bash
     pip install scikit-learn pandas numpy matplotlib
     ```

2. **Steps to Execute**
   - Upload your dataset (e.g., `cleanedreview.csv`).
   - Run the Python script or open the Jupyter Notebook in Google Colab.

3. **Dataset Format**
   - The dataset should have the following columns:
     | Column Name | Description                       |
     |-------------|-----------------------------------|
     | `text`      | The text or review data.          |
     | `label`     | Sentiment labels: positive/neutral. |

4. **Run the Code**
   - Preprocess the text.
   - Train and evaluate models.
   - Generate a classification report and compare metrics.

---

## 📬 Contact
If you have any questions or need further assistance:

**Name**: Lakshmi Devi S  
**Email**: [datascientist.ld1981@gmail.com](mailto:datascientist.ld1981@gmail.com)  

---


# MLP Performance Analysis for Credit Risk Prediction

## Overview
This project analyzes the performance of a **Multi-Layer Perceptron (MLP)** model for **credit risk prediction**. The dataset used contains information on financial transactions, customer profiles, and loan repayment history, which helps determine the likelihood of loan default and also analyse the width and depth impact on the MLP model performance

## Dataset
- The dataset consists of **numerical and categorical features** representing customer financial behavior.
- Target variable: **Credit Risk (Good or Bad)**
- Data Preprocessing: **Feature Engineering, Normalization, Handling Missing Values**

## Objective
To develop and evaluate an MLP-based model to predict credit risk and compare its performance against traditional classification models.

## Project Workflow
1. **Data Preprocessing**
   - Handling missing values
   - Feature selection and scaling
   - Encoding categorical variables
2. **Model Training & Evaluation**
   - Building an MLP classifier
   - Comparing with baseline models (Logistic Regression, Random Forest, etc.)
   - Performance metrics: **Accuracy, Precision, Recall, F1-score, AUC-ROC**
3. **Hyperparameter Tuning**
   - Optimizing MLP architecture using dropout and regularisation 
4. **Results & Interpretation**
   - Visualization of evaluation metrics
   - Model performance comparison

## Model Architecture
- **Input Layer:** Processed feature set
- **Hidden Layers:** Fully connected layers with ReLU activation
- **Output Layer:** Sigmoid activation for binary classification
- **Optimization Algorithm:** Adam
- **Loss Function:** Binary Cross-Entropy

## Performance Evaluation
- **Confusion Matrix**
- **ROC Curve & AUC Score**
- **Precision-Recall Curve**
- **Comparision with Not hypertuned model**

## Installation & Requirements
### Prerequisites:
Ensure you have the following installed:
- Python 3.8+
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn

### Install Dependencies:
```bash
pip install tensorflow scikit-learn pandas matplotlib numpy seaborn
```

## Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mlp-credit-risk.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mlp-credit-risk
   ```
3. Run in the google collab [Runtime used-CPU]  and also please download dataset from github to execute

## Results & Findings
- The MLP model achieved **higher ROC/AUC** compared to traditional classifiers, making it suitable for imbalanced credit risk prediction tasks.
- Feature importance analysis provided insights into key factors influencing loan defaults.

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by research in **financial risk assessment using deep learning**.
- Dataset sourced from **[Credit Risk Dataset - Kaggle]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk/data))**.


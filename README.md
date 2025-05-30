# DECISION-TREE-IMPLEMENTATION

COMPANY : CODTECH IT SOLUTIONS

 NAME : Althi vinodh kumar

 INTERN ID : CT04DN428

DOMAIN: MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

##
🔍 Description: Decision Tree Classifier UI for Loan Approval Prediction
This application is a graphical user interface (GUI) built using Python’s tkinter library that allows users to interactively load data, train a Decision Tree Classifier, visualize the resulting tree, and make predictions on new data records. Specifically, this version of the application focuses on a loan approval prediction scenario, using synthetic customer data with fields such as age, income, employment status, credit score, and loan amount.

🌱 Decision Tree Overview
A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks. It models data by learning simple decision rules inferred from features. Each internal node in the tree represents a test on a feature, each branch represents an outcome of the test, and each leaf node represents a class label. This method is widely appreciated for its interpretability and ease of use.

In this application, a Decision Tree is used to classify whether a loan will be approved or not (Yes or No), based on the applicant's personal and financial information.

🖥️ Application Components
1. User Interface Design
The UI is designed using tkinter, offering a clean and accessible layout. It includes:

Title label and test size slider.

Buttons for training the model, predicting a new record, and exporting the tree visualization.

A text area to display the classification report.

Labels to show model accuracy and errors.

2. Dataset Integration
The application uses a custom CSV file named loan_approval_data.csv. This file contains 10 rows and the following columns:

Age: Applicant’s age.

Income: Monthly income in dollars.

Employment: Categorical field indicating if the person is employed (Yes/No).

Credit_Score: Numerical score representing creditworthiness.

Loan_Amount: Amount of loan applied for.

Approved: Target variable indicating if the loan was approved.

The dataset is loaded at runtime, and the UI automatically trains the model on it without needing manual upload.

🧠 Model Training Workflow
When the user clicks “Train Model”, the application:

Splits the dataset into training and test sets based on the slider’s value (typically 70% train / 30% test).

Preprocesses the features — notably converting the Employment field to dummy variables using pd.get_dummies.

Trains a DecisionTreeClassifier from scikit-learn on the preprocessed training data.

Evaluates the model on the test data and displays:

Accuracy Score

Precision, Recall, and F1-Score (via classification report)

All these results are shown clearly in the output text area.

🔍 Tree Visualization & Export
The “Export Tree” button allows users to save the trained decision tree as a .png image. This is helpful for analyzing the decision paths and understanding the model’s logic. The visualization is created using plot_tree from matplotlib.

📈 Prediction on New Records
One of the most interactive features is the “Predict New Record” function. It opens a popup form where users can input details for a new loan applicant. When the "Predict" button is clicked:

The input is collected and validated.

The categorical Employment field is encoded using the same method as during training.

The application ensures that the input data has the same columns (including dummies) as the training data.

The trained model predicts whether the loan will be approved or not, and the result is shown in a popup message.
##

# OUTPUT
![Image](https://github.com/user-attachments/assets/9f8e017b-55c7-44e3-9000-cf707ea21531)
![Image](https://github.com/user-attachments/assets/2af40b8e-af91-4782-9e20-6f8150dd607f)
![Image](https://github.com/user-attachments/assets/137e811c-413e-4bcd-82b2-6efb71c49d88)
![Image](https://github.com/user-attachments/assets/7c1f798f-cf1c-4cbb-8328-685b83db0bec)

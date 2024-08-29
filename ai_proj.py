# Jalila Muoadi - 1201611
# Manar Mansour - 1201816
# Loan Eligibility Prediction Project

import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import tkinter as tk
from tkinter import ttk


# Function to predict loan eligibility based on user input
def predict_loan_eligibility():
    # Get user input from the entry widgets
    gender = gender_var.get()
    married = married_var.get()
    dependents = dependents_var.get()
    education = education_var.get()
    self_employed = self_employed_var.get()
    applicant_income = applicant_income_var.get()
    coapplicant_income = coapplicant_income_var.get()
    loan_amount = loan_amount_var.get()
    loan_amount_term = loan_amount_term_var.get()
    credit_history = credit_history_var.get()
    property_area = property_area_var.get()

    # Convert categorical input to numerical format
    gender = 0 if gender == 'Male' else 1
    married = 0 if married == 'No' else 1
    dependents = 3 if dependents == '3+' else int(dependents)
    education = 1 if education == 'Graduate' else 0
    self_employed = 0 if self_employed == 'No' else 1
    property_area = {'Rural': 0, 'Urban': 1, 'Semiurban': 2}[property_area]

    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [float(applicant_income)],
        'CoapplicantIncome': [float(coapplicant_income)],
        'LoanAmount': [float(loan_amount)],
        'Loan_Amount_Term': [float(loan_amount_term)],
        'Credit_History': [float(credit_history)],
        'Property_Area': [property_area]
    })

    # Predict loan eligibility using the trained models
    prediction_DT = model_DT.predict(user_data)[0]
    prediction_NB = model_NB.predict(user_data)[0]
    prediction_RF = model_RF.predict(user_data)[0]

    # Display the result in the GUI
    result_label.config(text=f"Decision Tree: {'Approved' if prediction_DT == 1 else 'Rejected'}\n"
                             f"Naive Bayes: {'Approved' if prediction_NB == 1 else 'Rejected'}\n"
                             f"Random Forest: {'Approved' if prediction_RF == 1 else 'Rejected'}")


if __name__ == "__main__":
    loan_data_train = pd.read_csv('train.csv')

    # get clean data
    # Remove duplicate rows based on all columns
    loan_data_train = loan_data_train.drop_duplicates()
    # Remove rows with null values
    loan_data_train = loan_data_train.dropna()
    # Resetting the index after removing rows
    loan_data_train = loan_data_train.reset_index(drop=True)

    # Convert categorical columns to numerical format
    loan_data_train.replace({
        'Loan_Status': {'N': 0, 'Y': 1},
        'Gender': {'Male': 0, 'Female': 1},
        'Married': {'No': 0, 'Yes': 1},
        'Dependents': {'3+': 3},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Urban': 1, 'Semiurban': 2}
    }, inplace=True)

    # Now split the data set into two separate data sets: X = input set and Y = output(target) set
    X = loan_data_train.drop(columns=['Loan_ID', 'Loan_Status'])
    y = loan_data_train['Loan_Status']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # test_size=0.2 specifies that 20% of the data should be used for testing, and the remaining 80% for training.

    # create a new instance of tree model
    model_DT = DecisionTreeClassifier()
    # Create a Gaussian Naive Bayes classifier
    model_NB = GaussianNB()
    # Create a Random Forest classifier
    model_RF = RandomForestClassifier()

    # train the models to learn the patterns in the data
    model_DT.fit(X_train, y_train)
    model_NB.fit(X_train, y_train)
    model_RF.fit(X_train, y_train)

    # predict the target variable.
    predictionDT = model_DT.predict(X_test)
    predictionNB = model_NB.predict(X_test)
    predictionRF = model_RF.predict(X_test)

    # detailed classification report, including precision, recall, F1-score for Loan_status class & confusion matrix
    # compares the predicted values (prediction) with the actual values (y_test1).
    # Decision Tree
    print("Decision Tree:")
    print("-Confusion Matrix:")
    confusion_matrix_DT = confusion_matrix(y_test, predictionDT)
    print(
        tabulate(confusion_matrix_DT, headers=['Predicted No', 'Predicted Yes'], showindex=['Actual No', 'Actual Yes'],
                 tablefmt='fancy_grid'))
    metrics_DT = {
        'Recall': metrics.recall_score(y_test, predictionDT),
        'Precision': metrics.precision_score(y_test, predictionDT),
        'Accuracy': metrics.accuracy_score(y_test, predictionDT),
        'F1': metrics.f1_score(y_test, predictionDT)
    }

    # Naive Bayes
    print("\nNaive Bayes:")
    print("-Confusion Matrix:")
    confusion_matrix_NB = confusion_matrix(y_test, predictionNB)
    print(
        tabulate(confusion_matrix_NB, headers=['Predicted No', 'Predicted Yes'], showindex=['Actual No', 'Actual Yes'],
                 tablefmt='fancy_grid'))
    metrics_NB = {
        'Recall': metrics.recall_score(y_test, predictionNB),
        'Precision': metrics.precision_score(y_test, predictionNB),
        'Accuracy': metrics.accuracy_score(y_test, predictionNB),
        'F1': metrics.f1_score(y_test, predictionNB)
    }
    # Random Forest
    print("\nRandom Forest:")
    print("-Confusion Matrix:")
    confusion_matrix_RF = confusion_matrix(y_test, predictionRF)
    print(
        tabulate(confusion_matrix_RF, headers=['Predicted No', 'Predicted Yes'], showindex=['Actual No', 'Actual Yes'],
                 tablefmt='fancy_grid'))
    metrics_RF = {
        'Recall': metrics.recall_score(y_test, predictionRF),
        'Precision': metrics.precision_score(y_test, predictionRF),
        'Accuracy': metrics.accuracy_score(y_test, predictionRF),
        'F1': metrics.f1_score(y_test, predictionRF)
    }
    # Collect metrics for each model
    metrics_dict = {
        'Decision Tree': metrics_DT,
        'Naive Bayes': metrics_NB,
        'Random Forest': metrics_RF
    }

    # Print the results in a tabular format
    headers = ['Metric', 'Decision Tree', 'Naive Bayes', 'Random Forest']

    table_data = []
    for metric in metrics_DT.keys():
        row = [metric]
        for model, metrics_model in metrics_dict.items():
            row.append(metrics_model[metric])
        table_data.append(row)

    print("The Evaluation result for each model:")
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

    # Create a Tkinter window
    window = tk.Tk()
    window.title("Loan Eligibility Prediction")

    # Create and place widgets for user input
    gender_label = ttk.Label(window, text="Gender:")
    gender_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    gender_var = ttk.Combobox(window, values=['Male', 'Female'])
    gender_var.grid(row=0, column=1, padx=10, pady=10)

    married_label = ttk.Label(window, text="Married:")
    married_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
    married_var = ttk.Combobox(window, values=['No', 'Yes'])
    married_var.grid(row=1, column=1, padx=10, pady=10)

    dependents_label = ttk.Label(window, text="Dependents:")
    dependents_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')
    dependents_var = ttk.Combobox(window, values=['0', '1', '2', '3+'])
    dependents_var.grid(row=2, column=1, padx=10, pady=10)

    education_label = ttk.Label(window, text="Education:")
    education_label.grid(row=3, column=0, padx=10, pady=10, sticky='w')
    education_var = ttk.Combobox(window, values=['Graduate', 'Not Graduate'])
    education_var.grid(row=3, column=1, padx=10, pady=10)

    self_employed_label = ttk.Label(window, text="Self Employed:")
    self_employed_label.grid(row=4, column=0, padx=10, pady=10, sticky='w')
    self_employed_var = ttk.Combobox(window, values=['No', 'Yes'])
    self_employed_var.grid(row=4, column=1, padx=10, pady=10)

    # Entry widgets for additional numerical attributes
    applicant_income_label = ttk.Label(window, text="Applicant Income:")
    applicant_income_label.grid(row=5, column=0, padx=10, pady=10, sticky='w')
    applicant_income_var = ttk.Entry(window)
    applicant_income_var.grid(row=5, column=1, padx=10, pady=10)

    coapplicant_income_label = ttk.Label(window, text="Coapplicant Income:")
    coapplicant_income_label.grid(row=6, column=0, padx=10, pady=10, sticky='w')
    coapplicant_income_var = ttk.Entry(window)
    coapplicant_income_var.grid(row=6, column=1, padx=10, pady=10)

    loan_amount_label = ttk.Label(window, text="Loan Amount:")
    loan_amount_label.grid(row=7, column=0, padx=10, pady=10, sticky='w')
    loan_amount_var = ttk.Entry(window)
    loan_amount_var.grid(row=7, column=1, padx=10, pady=10)

    loan_amount_term_label = ttk.Label(window, text="Loan Amount Term:")
    loan_amount_term_label.grid(row=8, column=0, padx=10, pady=10, sticky='w')
    loan_amount_term_var = ttk.Entry(window)
    loan_amount_term_var.grid(row=8, column=1, padx=10, pady=10)

    credit_history_label = ttk.Label(window, text="Credit History:")
    credit_history_label.grid(row=9, column=0, padx=10, pady=10, sticky='w')
    credit_history_var = ttk.Combobox(window, values=['0', '1'])
    credit_history_var.grid(row=9, column=1, padx=10, pady=10)

    property_area_label = ttk.Label(window, text="Property Area:")
    property_area_label.grid(row=10, column=0, padx=10, pady=10, sticky='w')
    property_area_var = ttk.Combobox(window, values=['Urban', 'Semiurban', 'Rural'])
    property_area_var.grid(row=10, column=1, padx=10, pady=10)

    # Create a button to trigger the prediction
    predict_button = ttk.Button(window, text="Predict Loan Eligibility", command=predict_loan_eligibility)
    predict_button.grid(row=11, column=0, columnspan=2, pady=20)

    # Create a label to display the result
    result_label = ttk.Label(window, text="")
    result_label.grid(row=12, column=0, columnspan=2, pady=10)

    # Start the Tkinter event loop
    window.mainloop()

    # Start the Tkinter event loop
    window.mainloop()

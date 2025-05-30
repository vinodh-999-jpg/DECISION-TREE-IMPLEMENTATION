import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class LoanApprovalUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Loan Approval Classifier")
        self.root.geometry("800x650")
        self.root.configure(bg="#e8f0fe")

        self.data = pd.read_csv("loan_approval_data.csv")
        self.target_col = "Approved"
        self.test_size = tk.DoubleVar(value=0.3)

        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Loan Approval Classifier", font=("Helvetica", 18, "bold"), bg="#e8f0fe").pack(pady=10)

        tk.Label(self.root, text="Test Size (0.1 to 0.5):", bg="#e8f0fe").pack()
        tk.Scale(self.root, from_=0.1, to=0.5, resolution=0.05, orient="horizontal", variable=self.test_size, length=200).pack(pady=5)

        tk.Button(self.root, text="Train Model", command=self.train_model, bg="#4CAF50", fg="white", width=20).pack(pady=10)
        tk.Button(self.root, text="Predict New Record", command=self.predict_new_record, bg="#2196F3", fg="white", width=20).pack(pady=5)
        tk.Button(self.root, text="Export Tree", command=self.export_tree_image, bg="#f39c12", fg="white", width=20).pack(pady=5)

        self.output_text = tk.Text(self.root, height=15, width=90)
        self.output_text.pack(pady=10)

        self.accuracy_label = tk.Label(self.root, text="Accuracy: N/A", font=("Helvetica", 12), bg="#e8f0fe")
        self.accuracy_label.pack()

    def train_model(self):
        df = self.data.copy()
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X = pd.get_dummies(X)

        self.features = X.columns  # Save encoded column order
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size.get(), random_state=1)

        self.clf = DecisionTreeClassifier()
        self.clf.fit(X_train, y_train)

        preds = self.clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Classification Report:\n{report}")
        self.accuracy_label.config(text=f"Accuracy: {acc:.2%}")

        self.classes = self.clf.classes_

    def export_tree_image(self):
        if hasattr(self, 'clf'):
            filename = filedialog.asksaveasfilename(defaultextension=".png")
            if filename:
                plt.figure(figsize=(12, 6))
                plot_tree(self.clf, filled=True, feature_names=self.features, class_names=self.classes)
                plt.title("Loan Approval Decision Tree")
                plt.savefig(filename)
                plt.close()
                messagebox.showinfo("Exported", f"Tree saved as {filename}")
        else:
            messagebox.showerror("Error", "Train the model first.")

    def predict_new_record(self):
        if not hasattr(self, 'clf'):
            messagebox.showerror("Error", "Train the model first.")
            return

        form = tk.Toplevel(self.root)
        form.title("New Loan Application")
        form.geometry("300x400")
        form.configure(bg="#ffffff")

        fields = ["Age", "Income", "Employment", "Credit_Score", "Loan_Amount"]
        entries = {}

        for i, field in enumerate(fields):
            tk.Label(form, text=field, bg="#ffffff").pack()
            entry = tk.Entry(form)
            entry.pack()
            entries[field] = entry

        def make_prediction():
            try:
                record = []
                for field in fields:
                    val = entries[field].get()
                    if field == "Employment":
                        record.append(val)
                    else:
                        record.append(float(val))

                input_df = pd.DataFrame([record], columns=fields)

                # Encode same way as training
                input_df = pd.get_dummies(input_df)

                # Add any missing columns from training
                for col in self.features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[self.features]  # Reorder

                prediction = self.clf.predict(input_df)[0]
                messagebox.showinfo("Prediction", f"Loan Approval Prediction: {prediction}")
                form.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to predict: {e}")

        tk.Button(form, text="Predict", command=make_prediction, bg="#4CAF50", fg="white").pack(pady=20)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = LoanApprovalUI(root)
    root.mainloop()

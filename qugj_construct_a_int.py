import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class InteractiveMachineLearningModelSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Machine Learning Model Simulator")

        # Load dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target

        # Split dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Initialize machine learning model
        self.model = LogisticRegression()

        # Create GUI components
        self.create_components()

    def create_components(self):
        # Create notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, expand=True)

        # Create frames
        self.frame1 = ttk.Frame(self.notebook)
        self.frame2 = ttk.Frame(self.notebook)
        self.frame3 = ttk.Frame(self.notebook)

        # Add frames to notebook
        self.notebook.add(self.frame1, text="Dataset")
        self.notebook.add(self.frame2, text="Model")
        self.notebook.add(self.frame3, text="Results")

        # Create dataset frame components
        self.dataset_label = ttk.Label(self.frame1, text="Dataset:")
        self.dataset_label.pack(pady=10)
        self.dataset_text = tk.Text(self.frame1, height=10, width=40)
        self.dataset_text.pack()
        self.dataset_text.insert(tk.INSERT, str(self.iris.data[:5]))

        # Create model frame components
        self.model_label = ttk.Label(self.frame2, text="Model:")
        self.model_label.pack(pady=10)
        self.model_combo = ttk.Combobox(self.frame2, values=["Logistic Regression", "Decision Tree", "Random Forest"], state="readonly")
        self.model_combo.set("Logistic Regression")
        self.model_combo.pack(pady=10)
        self.train_button = ttk.Button(self.frame2, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        # Create results frame components
        self.results_label = ttk.Label(self.frame3, text="Results:")
        self.results_label.pack(pady=10)
        self.results_text = tk.Text(self.frame3, height=10, width=40)
        self.results_text.pack()
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame3)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def train_model(self):
        # Train model
        self.model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Evaluate model
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        conf_mat = confusion_matrix(self.y_test, y_pred)

        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.INSERT, f"Accuracy: {accuracy:.2f}\n\n{report}\n\n{conf_mat}")

        # Plot confusion matrix
        self.axes.clear()
        self.axes.imshow(conf_mat, interpolation='nearest')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    simulator = InteractiveMachineLearningModelSimulator(root)
    root.mainloop()
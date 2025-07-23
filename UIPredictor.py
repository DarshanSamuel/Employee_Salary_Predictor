import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image

# Load the saved XGBoost model
model = joblib.load("xgboost_salary_model.pkl")
user_input_file = "user_inputs.csv"

# Extract unique job titles and education levels from original dataset
try:
    original_data = pd.read_csv("salary_data.csv")
    job_options = sorted(original_data["Job Title"].dropna().unique().tolist())
    edu_options = sorted(original_data["Education Level"].dropna().unique().tolist())
except Exception as e:
    job_options = ["Software Engineer", "Manager", "Analyst", "Consultant", "Other"]
    edu_options = ["High School", "Bachelor's", "Master's", "PhD"]

# Set up the main GUI window
root = tk.Tk()
root.title("Employee Salary Predictor")
root.geometry("1000x1000")
root.configure(bg="#f5f5f5")

# Function to refresh inputs
def refresh_inputs():
    for var in vars.values():
        var.set("")

    result_label.config(text="")

    for widget in plot_canvas_frame.winfo_children():
        widget.destroy()

    for label in warning_labels.values():
        label.config(text="")

    canvas.yview_moveto(0)
    canvas.configure(scrollregion=canvas.bbox("all"))  # dynamically reset scroll size

    # Repack both buttons
    predict_btn.pack(side="left", padx=10)
    refresh_inputs_btn.pack(side="right", padx=10)



# Add refresh button (top-right corner)
refresh_img = Image.open("refresh_icon.png").resize((90, 54))
refresh_icon = ImageTk.PhotoImage(refresh_img)
refresh_button = tk.Button(root, image=refresh_icon, command=refresh_inputs, bg="#f5f5f5", bd=0, highlightthickness=0)
refresh_button.place(x=1000, y=10)

header = tk.Label(root, text="Employee Salary Prediction", font=("Helvetica", 20, "bold"), bg="#f5f5f5", fg="#333")
header.pack(pady=20)

canvas_frame = tk.Frame(root, bg="#f5f5f5")
canvas_frame.pack(fill='both', expand=True)

canvas = tk.Canvas(canvas_frame, bg="#f5f5f5", height=700)
scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#f5f5f5")

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

vars = {
    "Age": tk.StringVar(),
    "Gender": tk.StringVar(),
    "Education Level": tk.StringVar(),
    "Job Title": tk.StringVar(),
    "Years of Experience": tk.StringVar(),
    "Current Salary": tk.StringVar(),
    "Duration (Years)": tk.StringVar()
}

warning_labels = {}

gender_options = ["Male", "Female", "Other"]

input_widgets = []
row = 0
for label, var in vars.items():
    tk.Label(scrollable_frame, text=label + ":", font=("Arial", 13), bg="#f5f5f5", anchor="w", width=20).grid(row=row, column=0, sticky='w', padx=10, pady=8)
    if label in ["Gender", "Education Level", "Job Title"]:
        options = gender_options if label == "Gender" else edu_options if label == "Education Level" else job_options
        entry = ttk.Combobox(scrollable_frame, textvariable=var, values=options, state="readonly", font=("Arial", 12), width=30)
    else:
        entry = tk.Entry(scrollable_frame, textvariable=var, font=("Arial", 12), width=33)
    entry.grid(row=row, column=1, padx=10, pady=5)
    warning = tk.Label(scrollable_frame, text="", fg="red", bg="#f5f5f5", font=("Arial", 10))
    warning.grid(row=row, column=2, padx=5, sticky="w")
    warning_labels[label] = warning
    input_widgets.append(entry)
    row += 1

result_label = tk.Label(scrollable_frame, text="", font=("Arial", 13), bg="#f5f5f5")
result_label.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

plot_canvas_frame = tk.Frame(scrollable_frame, bg="#f5f5f5")
plot_canvas_frame.grid(row=row, column=0, columnspan=2)
row += 1

button_frame = tk.Frame(scrollable_frame, bg="#f5f5f5")
button_frame.grid(row=row, column=0, columnspan=2, pady=20)

predict_btn = tk.Button(button_frame, text="Predict Salary", font=("Arial", 14), bg="#4caf50", fg="white", width=20)
predict_btn.pack(side="left", padx=10)

refresh_inputs_btn = tk.Button(button_frame, text="Clear Inputs", font=("Arial", 14), bg="#f44336", fg="white", width=20)
refresh_inputs_btn.pack(side="right", padx=10)

predict_btn.config(command=lambda: predict_salary())
refresh_inputs_btn.config(command=refresh_inputs)

def save_user_input(df):
    if os.path.exists(user_input_file):
        df.to_csv(user_input_file, mode='a', header=False, index=False)
    else:
        df.to_csv(user_input_file, index=False)

def predict_salary():
    for label in warning_labels.values():
        label.config(text="")
    try:
        has_error = False
        for field, var in vars.items():
            value = var.get()
            if value.strip() == "":
                warning_labels[field].config(text="This field is required")
                has_error = True

        if has_error:
            return

        age = float(vars["Age"].get())
        gender = vars["Gender"].get()
        education = vars["Education Level"].get()
        job = vars["Job Title"].get()
        exp = float(vars["Years of Experience"].get())
        current_salary = float(vars["Current Salary"].get())
        duration = float(vars["Duration (Years)"].get())

        if age < 0:
            warning_labels["Age"].config(text="Invalid Age")
            has_error = True
        if exp < 0:
            warning_labels["Years of Experience"].config(text="Invalid Experience")
            has_error = True
        if current_salary < 0:
            warning_labels["Current Salary"].config(text="Invalid Salary")
            has_error = True
        if duration < 0:
            warning_labels["Duration (Years)"].config(text="Invalid Duration")
            has_error = True
        if exp > age:
            warning_labels["Years of Experience"].config(text="Experience > Age")
            has_error = True

        if has_error:
            return

        future_exp = duration
        age_exp_interaction = age * future_exp
        exp_level = "Junior" if future_exp <= 5 else "Mid" if future_exp <= 10 else "Senior" if future_exp <= 20 else "Expert"

        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job,
            "Years of Experience": future_exp,
            "Age_Exp_Interaction": age_exp_interaction,
            "Experience_Level": exp_level,
            "Current Salary": current_salary
        }])

        prediction = model.predict(input_df)[0]
        delta = prediction - current_salary

        input_df["Current Salary"] = current_salary
        input_df["Duration"] = duration
        input_df["Predicted Salary"] = prediction
        input_df["Estimated Growth"] = delta
        save_user_input(input_df)

        result_label.config(text=f"Predicted Salary after {int(duration)} years: ₹{prediction:,.2f}\nExpected Growth: ₹{delta:,.2f}")

        plot_salary_over_years(current_salary, prediction, duration)

    except Exception as e:
        messagebox.showerror("Error", f"Please check inputs!\n\n{e}")

def plot_salary_over_years(current_salary, predicted_salary, years):
    for widget in plot_canvas_frame.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

    years = int(years)
    annual_growth = (predicted_salary - current_salary) / years if years > 0 else 0
    year_labels = list(range(1, years + 1))
    salary_vals = [current_salary + i * annual_growth for i in range(1, years + 1)]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(year_labels, salary_vals, marker='o', linestyle='-', color='green')
    ax.set_title("Salary Growth Over Time")
    ax.set_xlabel("Years")
    ax.set_ylabel("Salary (INR)")
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=plot_canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

root.mainloop()

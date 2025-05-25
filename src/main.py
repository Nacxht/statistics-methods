from scipy.stats import norm

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

class DataStatistic:
  def __init__(self, csv_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.df = pd.read_csv(f"{current_dir}/{csv_path}")

  def normality_histogram(self, column_name: str):
    data = np.array(self.df[column_name])
    sns.histplot(data, kde=True, stat="density", color="skyblue", label="data")
    
    xmin, xmax = plt.xlim()
    
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(data), np.std(data))

    plt.plot(x, p, "r", label=f"Distribusi Normal Dari Kolom {column_name}")
    plt.title(f"Histogram kolom '{column_name}' dengan kurva normal")
    plt.legend()
    plt.grid(True)
    plt.show()

  def scatter_plot(self, x_column: str, y_column: str):
    data_x = np.array(self.df[x_column])
    data_y = np.array(self.df[y_column])

    plt.scatter(data_x, data_y, alpha=0.7)
    plt.title(f"Korelasi kolom '{data_x}' dengan kolom '{data_y}'")
    
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    plt.grid(True)
    plt.show()

  def scatter_matrix(self, columns_name: list[str]):
    data = pd.DataFrame({column: self.df[column] for column in columns_name})

    sns.pairplot(data)
    plt.suptitle(f"Pair Plot Kolom: {columns_name}")
    plt.show()
    

data = DataStatistic("datasets/student_habits_performance.csv")
data.normality_histogram("study_hours_per_day")
data.normality_histogram("netflix_hours")
data.scatter_plot("exam_score", "study_hours_per_day")
data.scatter_matrix(["exam_score", "study_hours_per_day", "sleep_hours"])
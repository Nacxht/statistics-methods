import pandas as pd
import matplotlib.pyplot as plt

from .DataStatistic import DataStatistic
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class K_Means(DataStatistic):
  def __init__(self, csv_path: str, features: list[str] | None) -> None:
    super().__init__(csv_path)
    
    self.df: 'DataFrame' = self.df[features] if features else self.df

  def delete_missing_values(self) -> None:
    self.df = self.df.dropna()
  
  def scale_data(self):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(self.df)
    
    return X_scaled
  
  def show_elbow_graph(self, k_range_max: int):
    X_scaled = self.scale_data()
    
    if "X_scaled" in locals():
      sse = []
      k_range = range(1, k_range_max)

      for k in k_range:
        # inisialisasi model K-Means
        # 'n_init'='auto' akan menjalankan KMeans beberapa kali dengan meletakkan centroid secara random
        # lalu memilih hasil terbaik (untuk menghindari local minima)
        k_means = KMeans(n_clusters=k, random_state=42, n_init='auto')
        
        # latih model
        k_means.fit(X_scaled)

        # dapatkan nilai inersia (SSE/WCSS)
        sse.append(k_means.inertia_)

    # plot grafik elbow
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o', linestyle='--')
    plt.title("Grafik Elbow")
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Sum of Squared Error (SSE)")
    plt.grid(True)
    plt.xticks(k_range)
    plt.show()
  
  def show_silhouette_score(self, k_range_max: int) -> None:
    X_scaled = self.scale_data()

    if "X_scaled" in locals():
      silhouette_scores = []
      k_range = range(2, k_range_max)

      for k in k_range:
        # inisialisasi model K-Means
        k_means = KMeans(n_clusters=k, random_state=42, n_init='auto')

        # latih model dan dapatkan label cluster
        cluster_labels = k_means.fit_predict(X_scaled)

        # hitung silhouette score
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
      
      # plot grafik silhouette score
      plt.figure(figsize=(10, 6))
      plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
      plt.title("Silhouette Score")
      plt.xlabel("Jumlah Cluster (K)")
      plt.ylabel("Rata-Rata Silhouette Score")
      plt.grid(True)
      plt.xticks(k_range)
      plt.show()
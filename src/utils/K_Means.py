import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .DataStatistic import DataStatistic
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class K_Means(DataStatistic):
  def __init__(self, csv_path: str, features: list[str] | None) -> None:
    super().__init__(csv_path)
    
    self.features: list[str] | None = features
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
    
  def clustering(self, k: int, feature_1: str, feature_2: str, max_iters:int = 10) -> 'KMeans | None':
    X_scaled = self.scale_data()
    clustered_df = self.df.copy()

    if "X_scaled" in locals():     
      k_means = KMeans(n_clusters=k, random_state=42, n_init=max_iters, verbose=True)
      clustered_df["cluster"] = k_means.fit_predict(X_scaled)

      # menghitung rata-rata dari setiap fitur untuk setiap cluster
      cluster_means = clustered_df.groupby("cluster").mean()
      print(f"\nRata-rata fitur per-cluster (dalam skala asli data):\n{cluster_means}")

      # k_means_init = KMeans(n_clusters=k, random_state=42, n_init=1)
      # k_means_init.fit(X_scaled)
      # initial_centroid = k_means_init.cluster_centers_
      
      # tolerance_value = 1e-4 
      # centroid_history = []

      # temp_k_means = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
      # temp_k_means.fit(X_scaled)
      # centroids = temp_k_means.cluster_centers_

      # for i in range(max_iters):
      #   centroid_history.append(centroids.copy())

      #   distances = np.sqrt(((X_scaled - centroids[:, np.newaxis])**2).sum(axis=2))
      #   labels = np.argmin(distances, axis=0)

      #   new_centroids =  np.array(
      #     [
      #       X_scaled[labels == j].mean(axis=0) if
      #       np.sum(labels == j) > 0 else
      #       centroids[j] for j in range(k)
      #     ]
      #   )
        
        # if np.all(np.abs(new_centroids - centroids) < tolerance_value): # Gunakan tol dari KMeans asli
        #   print(f"\nKonvergensi tercapai pada iterasi {i+1}.")
        #   centroids = new_centroids
        #   break

      #   centroids = new_centroids
      #   centroid_history.append(centroids.copy()) # Simpan posisi final setelah konvergensi

      # print(f"Jumlah iterasi: {len(centroid_history)}")
      # print(f"Letak centroid di tiap iterasi:\n{centroid_history}")

      # menghitung ukuran cluster (jumlah anggota)
      cluster_size = clustered_df["cluster"].value_counts().sort_index()
      print(f"\nUkuran cluster (jumlah anggota):\n{cluster_size}")

      # letak-letak cluster
      print(f"\nLetak akhir cluster:\n{k_means.cluster_centers_}")

      # visualisasi profil cluster
      # membantu membandingkan fitur-fitur diantara cluster
      for column in clustered_df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=clustered_df.index, y=clustered_df[column])
        plt.title(f"Rata-rata {column} per-cluster")
        plt.xlabel("Cluster")
        plt.ylabel(f"Rata-rata {column}")
        plt.grid(axis='y')
        plt.show()
    
      # visualisasi clustering
      plt.figure(figsize=(10, 8))
      sns.scatterplot(
        x=feature_1, y=feature_2, hue='cluster',
        data=clustered_df, palette='viridis', s=100, alpha=0.8,
        legend='full'
      )

      centroids_scaled = k_means.cluster_centers_

      plt.title(f"Hasil K-Means Clustering (K={k})")
      plt.xlabel(feature_1)
      plt.ylabel(feature_2)
      plt.legend()
      plt.grid(True)
      plt.show()

      # mengevaluasi kualitas cluster
      if k <= 1:
        return
      
      # metode calinski harabasz index
      # semakin tinggi skor, semakin baik kualitasnya
      ch_score = calinski_harabasz_score(X_scaled, clustered_df["cluster"])
      print(f"\nCalinski-Harabasz Index: {ch_score:.2f}")

      # metode davies bouldin index
      # semakin rendah skor, semakin baik kualitasnya
      db_score = davies_bouldin_score(X_scaled, clustered_df["cluster"])
      print(f"Davies-Bouldin Index: {db_score:.2f}")

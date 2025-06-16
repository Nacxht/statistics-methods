from utils import K_Means

student_habits = K_Means(
    "../datasets/student_habits_performance.csv",
    ["social_media_hours", "sleep_hours"]
)

# menghapus baris yang memiliki data kosong
student_habits.delete_missing_values()

# mencari K optimal menggunakan silhouette score
student_habits.show_silhouette_score(10)

# menentukan K optimal setelah menganalisis grafik silhouette score
optimal_k = 6

# penerapan K-Means
student_habits.clustering(optimal_k, "social_media_hours", "sleep_hours")
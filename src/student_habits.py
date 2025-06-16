from utils import K_Means

student_habits = K_Means(
    "../datasets/student_habits_performance.csv",
    ["social_media_hours", "sleep_hours"]
)

# menghapus baris yang memiliki data kosong
student_habits.delete_missing_values()

# mencari K optimal menggunakan silhouette score
student_habits.show_silhouette_score(10)
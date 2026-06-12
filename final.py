import pandas as pd
social=pd.read_csv("social-media.csv")
labelled=pd.read_csv("social-media-labelled.csv")
print("Datasets loaded successfully!")

print("Social Media Dataset:")
print(social.head())

print("\nLabelled Dataset:")
print(labelled.head())

print(social.columns)
print(labelled.columns)

from sklearn.cluster import KMeans

num_clusters=labelled["label"].nunique()
print(f"\nNumber of labels found: {num_clusters}")

X_cluster = social[["visit_score", "spending_rank"]]

kmeans = KMeans(n_clusters=num_clusters, random_state=42)

social["cluster"] = kmeans.fit_predict(X_cluster)   

print("\nClustering done.")
print(social.head())

import matplotlib.pyplot as plt
center=kmeans.cluster_centers_
plt.figure(figsize=(8, 6))
plt.scatter(social["visit_score"], social["spending_rank"], c=social["cluster"])
plt.scatter(center[:, 0], center[:, 1], c='red', marker='x', s=300)
plt.xlabel("Visit Score")
plt.ylabel("Spending Rank")
plt.title("K-Means Clustering of Social Media Users")
plt.savefig("kmeans_clusters.png")
plt.show()

from sklearn.ensemble import RandomForestClassifier
X_train=labelled[["visit_score", "spending_rank"]]
y_train=labelled["label"]

model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)
print("\nClassification model trained.")

X_unlabelled=social[["visit_score", "spending_rank"]]
social["predicted_label"]=model.predict(X_unlabelled)
print("Labels generated")

print("Sample predictions:")
print(social.head())   

output_file="social-media-final-labelled.csv"
social.to_csv(output_file, index=False)
print(f"\nFinal labelled dataset saved as:{output_file}")

print("\nProject completed successfully!")
print("f,Total records labelled: {len(social)}")
print("Tasks completed: Clustering, Classification, and Automatic Labelling")

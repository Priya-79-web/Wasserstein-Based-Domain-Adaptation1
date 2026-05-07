# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import ot
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ==========================================
# 2. LOAD DATASET
# ==========================================
data = pd.read_csv(
    "/kaggle/input/datasets/hurshd0/abalone-uci/abalone_original.csv"
)

data.columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings"
]

print("Dataset Shape:", data.shape)

# ==========================================
# 3. CLASSIFICATION LABEL
# ==========================================
data["AgeClass"] = (data["Rings"] > 10).astype(int)

# ==========================================
# 4. ENCODE SEX
# ==========================================
data_encoded = pd.get_dummies(data, columns=["Sex"])

X = data_encoded.drop(["Rings", "AgeClass"], axis=1)
y = data_encoded["AgeClass"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 5. BASELINE RANDOM FOREST
# ==========================================
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))

# ==========================================
# 6. NEURAL NETWORK
# ==========================================
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    verbose=0
)

pred_nn = (model.predict(X_test) > 0.5).astype(int)

print("\n=== Neural Network ===")
print("Accuracy :", accuracy_score(y_test, pred_nn))
print("Precision:", precision_score(y_test, pred_nn))
print("Recall   :", recall_score(y_test, pred_nn))
print("F1 Score :", f1_score(y_test, pred_nn))

# ==========================================
# 7. PAPER SOURCE TARGET SPLIT
# ==========================================
source = data[
    (data["Rings"] >= 6) &
    (data["Rings"] <= 14)
]

target = data[
    (data["Rings"] >= 7) &
    (data["Rings"] <= 15)
]

feature_cols = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight"
]

Xs = source[feature_cols].values
Xt = target[feature_cols].values

scaler_ot = StandardScaler()
Xs = scaler_ot.fit_transform(Xs)
Xt = scaler_ot.transform(Xt)

# ==========================================
# 8. WASSERSTEIN DISTANCE
# ==========================================
M = ot.dist(Xs, Xt)
M = M / (np.std(M)+1e-8)

a = np.ones(len(Xs))/len(Xs)
b = np.ones(len(Xt))/len(Xt)

gamma = ot.emd(a, b, M)

W = np.sum(gamma*M)

print("\nWasserstein Distance:", round(W,4))

# ==========================================
# 9. PAPER WAX
# ==========================================
paper_scores = np.zeros(len(feature_cols))

for i in range(len(feature_cols)):
    diff = np.abs(
        Xs[:,i][:,None] -
        Xt[:,i][None,:]
    )

    paper_scores[i] = np.sum(gamma*diff)

paper_scores = paper_scores/paper_scores.sum()

print("\n=== Paper WaX ===")
for f,s in zip(feature_cols,paper_scores):
    print(f"{f}: {s:.4f}")

# ==========================================
# 10. ADAPTIVE WAX
# ==========================================
adaptive_scores = np.zeros(len(feature_cols))

for i in range(len(feature_cols)):

    mean_shift = abs(
        np.mean(Xs[:,i]) -
        np.mean(Xt[:,i])
    )

    variance_shift = abs(
        np.var(Xs[:,i]) -
        np.var(Xt[:,i])
    )

    adaptive_scores[i] = (
        paper_scores[i]
        + 0.01
        + 0.03*mean_shift
        + 0.02*variance_shift
    )

print("\n=== Adaptive WaX ===")
for f,s in zip(feature_cols,adaptive_scores):
    print(f"{f}: {s:.4f}")

# ==========================================
# 11. U-WAX SUBSPACES
# ==========================================
subspaces = {
    "Subspace 1": ["Length","Diameter"],
    "Subspace 2": ["Height","Whole weight"],
    "Subspace 3": [
        "Shucked weight",
        "Viscera weight",
        "Shell weight"
    ]
}

subspace_scores = {}

print("\n=== U-WaX ===")

for name, features in subspaces.items():

    idx = [feature_cols.index(f) for f in features]

    Xs_sub = Xs[:,idx]
    Xt_sub = Xt[:,idx]

    M_sub = ot.dist(Xs_sub,Xt_sub)
    M_sub = M_sub/(np.std(M_sub)+1e-8)

    gamma_sub = ot.emd(
        np.ones(len(Xs_sub))/len(Xs_sub),
        np.ones(len(Xt_sub))/len(Xt_sub),
        M_sub
    )

    W_sub = np.sum(gamma_sub*M_sub)

    subspace_scores[name] = W_sub
    print(name,":",round(W_sub,4))

# ==========================================
# 12. ADAPTIVE U-WAX (NEW)
# ==========================================
adaptive_subspace_scores = {}

print("\n=== Adaptive U-WaX (Novel) ===")

for name, features in subspaces.items():

    idx = [feature_cols.index(f) for f in features]

    sub_score = 0

    for i in idx:

        mean_shift = abs(
            np.mean(Xs[:,i]) -
            np.mean(Xt[:,i])
        )

        variance_shift = abs(
            np.var(Xs[:,i]) -
            np.var(Xt[:,i])
        )

        enhanced = (
            paper_scores[i]
            + 0.01
            + 0.03*mean_shift
            + 0.02*variance_shift
        )

        sub_score += enhanced

    adaptive_subspace_scores[name] = sub_score

    print(name,":",round(sub_score,4))

# ==========================================
# 13. PCA VISUALIZATION
# ==========================================
pca = PCA(n_components=2)

Xs_2d = pca.fit_transform(Xs)
Xt_2d = pca.transform(Xt)

# ==========================================
# 14. CLUSTER BASELINE
# ==========================================
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(Xs_2d)

# ==========================================
# 15. FINAL VISUALIZATION
# ==========================================
fig, axes = plt.subplots(
    2,
    3,
    figsize=(18,10)
)

# Paper WaX
axes[0,0].barh(
    feature_cols,
    paper_scores,
    color='salmon'
)
axes[0,0].set_title("Paper WaX")

# Adaptive WaX
axes[0,1].barh(
    feature_cols,
    adaptive_scores,
    color='orange'
)
axes[0,1].set_title("Adaptive WaX")

# Adaptive U-WaX
axes[0,2].bar(
    adaptive_subspace_scores.keys(),
    adaptive_subspace_scores.values(),
    color='green'
)
axes[0,2].set_title("Adaptive U-WaX")

# Source Target
axes[1,0].scatter(
    Xs_2d[:,0],
    Xs_2d[:,1],
    alpha=0.3,
    label="Source"
)
axes[1,0].scatter(
    Xt_2d[:,0],
    Xt_2d[:,1],
    alpha=0.3,
    label="Target"
)
axes[1,0].legend()
axes[1,0].set_title("Source vs Target")

# Clusters
for i in range(3):
    idx = np.where(clusters==i)
    axes[1,1].scatter(
        Xs_2d[idx,0],
        Xs_2d[idx,1],
        label=f"Cluster {i+1}"
    )

axes[1,1].legend()
axes[1,1].set_title("Cluster Baseline")

# Empty subplot
axes[1,2].axis("off")

plt.tight_layout()
plt.show()
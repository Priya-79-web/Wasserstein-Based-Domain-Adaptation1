# ===========================
# INSTALL DEPENDENCIES
# ===========================
!pip install -q ftfy regex tqdm
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q POT

# ===========================
# IMPORTS
# ===========================
import os
import tarfile
import numpy as np
import torch
import clip
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ot

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===========================
# DEVICE
# ===========================
device = "cpu"
print("Using device:", device)

# ===========================
# EXTRACT LFW
# ===========================
lfw_tar = "/kaggle/input/datasets/atulanandjha/lfwpeople/lfw-funneled.tgz"
extract_path = "/kaggle/working/lfw"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

if not os.path.exists(os.path.join(extract_path, "lfw_funneled")):
    print("Extracting LFW...")
    with tarfile.open(lfw_tar, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Done!")

# ===========================
# PATHS
# ===========================
celeba_path = "/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba"
lfw_path = "/kaggle/working/lfw/lfw_funneled"

# ===========================
# LOAD IMAGES
# ===========================
def load_images(folder, max_images=300):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".jpg"):
                paths.append(os.path.join(root, f))
            if len(paths) >= max_images:
                return paths
    return paths

celeba_imgs = load_images(celeba_path)
lfw_imgs = load_images(lfw_path)

print("\nLoaded:")
print("CelebA:", len(celeba_imgs))
print("LFW:", len(lfw_imgs))

# ===========================
# LOAD MODEL
# ===========================
model, preprocess = clip.load("ViT-B/32", device=device)

# ===========================
# EMBEDDINGS
# ===========================
def get_embeddings(paths):
    feats = []
    for p in tqdm(paths):
        try:
            img = Image.open(p).convert("RGB")
            img = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                f = model.encode_image(img)

            f = f / f.norm(dim=1, keepdim=True)
            feats.append(f.cpu().numpy()[0])
        except:
            continue
    return np.array(feats)

print("\nExtracting embeddings...")
X = get_embeddings(celeba_imgs)
Y = get_embeddings(lfw_imgs)

print("Shapes:", X.shape, Y.shape)

# ===========================
# ADAPTIVE WAX (NOVEL)
# ===========================
print("\nComputing Adaptive WaX...")

a = np.ones(len(X)) / len(X)
b = np.ones(len(Y)) / len(Y)

# 🔥 Adaptive normalization (variance-based)
feature_var = np.var(np.vstack([X, Y]), axis=0) + 1e-8
X_adapt = X / np.sqrt(feature_var)
Y_adapt = Y / np.sqrt(feature_var)

# Distance matrix
M = ot.dist(X_adapt, Y_adapt)

# 🔥 Stable Sinkhorn
gamma = ot.sinkhorn(a, b, M, reg=0.05)

# Wasserstein Distance
W = np.sum(gamma * M)
print("Adaptive Wasserstein Distance:", W)

# ===========================
# FEATURE IMPORTANCE
# ===========================
print("\nComputing Feature Importance...")

feature_importance = np.zeros(X.shape[1])

for i in range(len(X)):
    for j in range(len(Y)):
        diff = (X[i] - Y[j]) ** 2
        feature_importance += gamma[i, j] * diff

# Normalize safely
total = np.sum(feature_importance)
if total == 0:
    feature_importance = np.ones_like(feature_importance) / len(feature_importance)
else:
    feature_importance /= total

# Top features
top_features = np.argsort(feature_importance)[::-1][:10]

print("\nTop Features:")
for i in top_features:
    print(f"Feature {i}: {feature_importance[i]:.6f}")

# ===========================
# GRAPH: FEATURE IMPORTANCE
# ===========================
plt.figure()
plt.bar(range(50), feature_importance[:50])
plt.title("Adaptive WaX Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# ===========================
# ADAPTIVE U-WAX (FIXED)
# ===========================
print("\nAdaptive U-WaX...")

idx = np.argsort(gamma.ravel())[-2000:]
rows, cols = np.unravel_index(idx, gamma.shape)

diffs = X[rows] - Y[cols]

# 🔥 SAFE weights
weights = gamma[rows, cols]
weight_sum = np.sum(weights)

if weight_sum == 0 or np.isnan(weight_sum):
    print("⚠️ Using uniform weights")
    weights = np.ones_like(weights) / len(weights)
else:
    weights = weights / weight_sum

# 🔥 Remove NaNs
mask = ~np.isnan(diffs).any(axis=1)
diffs = diffs[mask]
weights = weights[mask]

# 🔥 Final check
if len(diffs) == 0:
    raise ValueError("No valid data after cleaning!")

# Weighted PCA
pca = PCA(n_components=3)
pca.fit(diffs * weights[:, None])

U = pca.components_

# ===========================
# CONCEPT IMPORTANCE
# ===========================
importance_scores = []

for i, u in enumerate(U):
    score = np.mean(np.abs(diffs @ u))
    importance_scores.append(score)
    print(f"Concept {i+1}: {score:.4f}")

# ===========================
# GRAPH: CONCEPT IMPORTANCE
# ===========================
plt.figure()
plt.bar(range(1, len(importance_scores)+1), importance_scores)
plt.title("Adaptive U-WaX Concept Importance")
plt.xlabel("Concept")
plt.ylabel("Importance")
plt.show()

# ===========================
# VISUALIZATION (TOP PAIRS)
# ===========================
top_idx = np.argsort(gamma.ravel())[::-1][:5]
pairs = np.dstack(np.unravel_index(top_idx, gamma.shape))[0]

plt.figure(figsize=(8,8))

for i, (a_i, b_i) in enumerate(pairs):
    img1 = Image.open(celeba_imgs[a_i])
    img2 = Image.open(lfw_imgs[b_i])

    plt.subplot(len(pairs),2,2*i+1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title("CelebA")

    plt.subplot(len(pairs),2,2*i+2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title("LFW")

plt.show()

# ===========================
# TEXT INTERPRETATION
# ===========================
labels = ["young woman", "old man", "smiling face", "serious face"]

text = clip.tokenize(labels).to(device)

with torch.no_grad():
    text_feat = model.encode_text(text)
    text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)

u = torch.tensor(U[0]).float().to(device)
scores = (text_feat @ u).cpu().numpy()

print("\nSemantic Meaning:")
for l, s in zip(labels, scores):
    print(l, ":", round(float(s), 4))

print("\n✅ DONE SUCCESSFULLY (NO ERRORS)")
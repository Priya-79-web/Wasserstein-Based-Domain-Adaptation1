
# ==========================================
# 1. IMPORTS
# ==========================================
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import ot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

from torchvision.models import (
    resnet18, resnet50, vgg16,
    ResNet18_Weights, ResNet50_Weights, VGG16_Weights,
    googlenet, GoogLeNet_Weights,
    vit_b_16, ViT_B_16_Weights,
    alexnet, AlexNet_Weights
)

# ==========================================
# 2. SETTINGS
# ==========================================
DATASET_PATH = r"C:\Users\priya\OneDrive\Desktop\Wasserstein\domain\office_caltech_10"
SOURCE_DOMAIN = "amazon"
TARGET_DOMAIN = "webcam"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================================
# 4. MODEL LOADER
# ==========================================
def get_model(name):

    if name == "ResNet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()

    elif name == "ResNet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()

    elif name == "VGG16":
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    elif name == "CaffeNet":
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    elif name == "GoogLeNet":
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        model.fc = torch.nn.Identity()

    elif name == "ViT":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Identity()

    return model.to(DEVICE).eval()

# ==========================================
# 5. LOAD DATA
# ==========================================
def load_domain(domain_path, model):
    X, y = [], []
    classes = sorted(os.listdir(domain_path))

    for label, cls in enumerate(classes):
        cls_path = os.path.join(domain_path, cls)

        if not os.path.isdir(cls_path):
            continue

        for img_name in os.listdir(cls_path):
            try:
                img = Image.open(os.path.join(cls_path, img_name)).convert("RGB")
                img = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feat = model(img).cpu().numpy().flatten()

                X.append(feat)
                y.append(label)

            except:
                continue

    return np.array(X), np.array(y)

# ==========================================
# 6. WAX PIPELINE (FIXED NOVEL)
# ==========================================
def wax_pipeline(Xs, Xt):

    scaler = StandardScaler()
    X_all = np.vstack([Xs, Xt])
    X_all = scaler.fit_transform(X_all)

    Xs = X_all[:len(Xs)]
    Xt = X_all[len(Xs):]

    # PCA
    if Xs.shape[1] > 2000:
        pca = PCA(n_components=0.95)
        Xs = pca.fit_transform(Xs)
        Xt = pca.transform(Xt)

    # OT
    N, M = len(Xs), len(Xt)
    a = np.ones(N) / N
    b = np.ones(M) / M

    C = ot.dist(Xs, Xt)
    C = C / (C.max() + 1e-8)

    gamma = ot.emd(a, b, C)
    W_dist = np.sum(gamma * C)

    # WaX relevance
    d = Xs.shape[1]
    R = np.zeros(d)

    for k in range(N):
        for l in range(M):
            diff = np.abs(Xs[k] - Xt[l])
            if diff.sum() == 0:
                continue
            R += gamma[k, l] * (diff / (diff.sum() + 1e-8))

    sigma = np.std(np.vstack([Xs, Xt]), axis=0)
    R_DA = R / (sigma**1.5 + 1e-8)

    # PAPER FEATURES (top 70%)
    idx = np.argsort(-R_DA)
    keep_paper = idx[int(0.3 * d):]

    # NOVEL FEATURES (bottom 30%) FIXED
    keep_novel = np.argsort(R_DA)[:int(0.3 * d)]

    return (
        Xs, Xt,
        Xs[:, keep_paper], Xt[:, keep_paper],
        Xs[:, keep_novel], Xt[:, keep_novel],
        W_dist
    )

# ==========================================
# 7. MAIN (WITH REAL METRICS STORAGE)
# ==========================================
models_list = ["VGG16", "CaffeNet", "GoogLeNet", "ResNet18", "ResNet50", "ViT"]

model_names = []
baseline_scores = []
paper_scores = []
novel_scores = []

all_preds = []
true_labels = None

for m in models_list:

    print(f"\nRunning model: {m}")

    model = get_model(m)

    Xs, ys = load_domain(os.path.join(DATASET_PATH, SOURCE_DOMAIN), model)
    Xt, yt = load_domain(os.path.join(DATASET_PATH, TARGET_DOMAIN), model)

    if true_labels is None:
        true_labels = yt

    print("Shapes:", Xs.shape, Xt.shape)

    Xo, Xto, Xp, Xtp, Xn, Xtn, W = wax_pipeline(Xs, Xt)

    if m == "CaffeNet":
        clf = SVC(kernel='linear', C=10)
    else:
        clf = SVC(kernel='rbf', C=100, gamma='scale')

    # BASELINE
    clf.fit(Xo, ys)
    base_acc = clf.score(Xto, yt)

    # PAPER
    clf.fit(Xp, ys)
    paper_acc = clf.score(Xtp, yt)

    # NOVEL
    clf.fit(Xn, ys)
    novel_acc = clf.score(Xtn, yt)

    print("Wax-Accuracy :", base_acc)
    print("Adaptive Wax-Accuracy :", paper_acc)
    #print("Novel  :", novel_acc)
    print("Wasserstein Distance:", W)

    # store results
    model_names.append(m)
    baseline_scores.append(base_acc * 100)
    paper_scores.append(paper_acc * 100)
    novel_scores.append(novel_acc * 100)

    # ensemble prediction source
    pred = clf.predict(Xtn)
    all_preds.append(pred)

# ==========================================
# 8. ENSEMBLE (MAJORITY VOTE)
# ==========================================
all_preds = np.array(all_preds)

final_pred = []
for i in range(all_preds.shape[1]):
    votes = all_preds[:, i]
    final_pred.append(Counter(votes).most_common(1)[0][0])

ensemble_acc = accuracy_score(true_labels, final_pred)

print("\n🔥 Ensemble Accuracy:", ensemble_acc)

# ==========================================
# 9. REAL RESULT GRAPH (FIXED)
# ==========================================
x = np.arange(len(model_names))

plt.figure(figsize=(8,5))

plt.plot(x, baseline_scores, '--o', label='Baseline')
plt.plot(x, paper_scores, '-.s', label='Wax-Score')
plt.plot(x, novel_scores, '-o', label='Adaptive-Wax-Score')

plt.xticks(x, model_names, rotation=20)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("WaX Performance (REAL OUTPUT FROM CODE)")

plt.legend()
plt.grid()
plt.show()
# %% [markdown]
# # Exercice 1 : Régularisation

# %% [markdown]
# ## Question 1

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV, RidgeCV, lasso_path
from sklearn.metrics import mean_squared_error

# --- Configuration pour la reproductibilité ---
np.random.seed(42)

# ==========================================
# 1. GÉNÉRATION DES DONNÉES
# ==========================================
n = 1000
p = 5000
s = 15  # Sparsité

# Génération de X (i.i.d gaussiens standards)
X = np.random.randn(n, p)

# Génération de beta
beta_true = np.zeros(p)
beta_true[:s] = 1.0  # Les 15 premiers coefficients à 1

# Génération du bruit eta
eta = np.random.randn(n)

# Génération de Y
Y = X @ beta_true + eta

# Séparation de l'échantillon (2/3 train, 1/3 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.34, random_state=42
)

print(f"Dimensions Train: {X_train.shape}, Test: {X_test.shape}")
print("-" * 50)

# ==========================================
# a. ESTIMATION ELASTIC-NET POUR ALPHA DANS {0, ..., 1}
# ==========================================
# Note: Dans sklearn, le paramètre de mélange est 'l1_ratio'. 
# Le paramètre de régularisation est 'alpha' (lambda dans l'énoncé).
# Ici, nous définissons une grille de ratios.

l1_ratios = [round(x * 0.1, 1) for x in range(11)] # [0.0, 0.1, ..., 1.0]

print("Liste des ratios alpha (l1_ratio) à tester :", l1_ratios)
# Note : L'estimation concrète se fait généralement conjointement 
# avec la recherche du lambda optimal (voir partie c), car un ElasticNet 
# sans lambda défini n'a pas de sens.

# ==========================================
# b. CHEMIN DE RÉGULARISATION DU LASSO
# ==========================================
print("Calcul du chemin de régularisation LASSO...")

# eps=5e-3 définit la longueur du chemin (ratio alpha_min / alpha_max)
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps=5e-3)

plt.figure(figsize=(10, 6))
# On trace les coefficients en fonction du log(lambda)
log_alphas_lasso = np.log10(alphas_lasso)

for coef_l in coefs_lasso:
    plt.plot(log_alphas_lasso, coef_l)

plt.xlabel('Log(Lambda)')
plt.ylabel('Coefficients')
plt.title('Chemin de régularisation du LASSO')
plt.axis('tight')
plt.show()

# ==========================================
# c. OPTIMISATION DES HYPERPARAMÈTRES (Sur Train uniquement)
# ==========================================
print("\nOptimisation des modèles par Validation Croisée (CV)...")

# 1. LASSO (l1_ratio = 1)
# ElasticNetCV gère automatiquement la recherche du lambda optimal (alpha dans sklearn)
# cv=5 pour une validation croisée à 5 plis
model_lasso_cv = ElasticNetCV(l1_ratio=1.0, cv=5, random_state=42, n_jobs=-1)
model_lasso_cv.fit(X_train, y_train)
best_lambda_lasso = model_lasso_cv.alpha_
print(f"LASSO optimal lambda: {best_lambda_lasso:.4f}")

# 2. RIDGE (l1_ratio = 0)
# Pour Ridge pur, RidgeCV est plus efficace et stable numériquement que ElasticNet(l1_ratio=0)
model_ridge_cv = RidgeCV(alphas=np.logspace(-2, 2, 20), cv=5)
model_ridge_cv.fit(X_train, y_train)
best_lambda_ridge = model_ridge_cv.alpha_
print(f"RIDGE optimal mu: {best_lambda_ridge:.4f}")

# 3. ELASTIC NET (Recherche sur grille 2D)
# ElasticNetCV permet de passer une liste de l1_ratios.
# Il cherchera le meilleur lambda pour CHAQUE l1_ratio, puis gardera le meilleur global.
# Nous excluons 0 et 1 car déjà traités par Ridge et Lasso spécifiquement, 
# mais on pourrait les inclure. Ici on prend ]0, 1[.
ratios_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]

model_en_cv = ElasticNetCV(l1_ratio=ratios_to_test, cv=5, random_state=42, n_jobs=-1)
model_en_cv.fit(X_train, y_train)

best_lambda_en = model_en_cv.alpha_
best_alpha_en = model_en_cv.l1_ratio_ # C'est le paramètre alpha de l'énoncé
print(f"ELASTIC-NET optimal : lambda={best_lambda_en:.4f}, alpha (ratio)={best_alpha_en}")


# ==========================================
# d. COMPARAISON SUR L'ÉCHANTILLON TEST
# ==========================================
print("-" * 50)
print("Évaluation sur l'échantillon TEST (MSE) :")

# Prédictions
y_pred_lasso = model_lasso_cv.predict(X_test)
y_pred_ridge = model_ridge_cv.predict(X_test)
y_pred_en = model_en_cv.predict(X_test)

# Calcul des erreurs quadratiques moyennes
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_en = mean_squared_error(y_test, y_pred_en)

print(f"MSE LASSO       : {mse_lasso:.4f}")
print(f"MSE RIDGE       : {mse_ridge:.4f}")
print(f"MSE ELASTIC-NET : {mse_en:.4f}")

# Identification du gagnant
errors = {"LASSO": mse_lasso, "RIDGE": mse_ridge, "ELASTIC-NET": mse_en}
best_model_name = min(errors, key=errors.get)

print(f"\n>>> Le meilleur modèle est : {best_model_name}")

# Interprétation contextuelle
print("\nInterprétation :")
print(f"Le vrai modèle a une sparsité s={s} sur p={p} (très clairsemé).")
if best_model_name != "RIDGE":
    print("Comme attendu, Lasso ou Elastic-Net performe mieux car Ridge ne fait pas de sélection de variable.")
else:
    print("Résultat inattendu pour un modèle clairsemé (vérifier le bruit ou la séparation).")

# %% [markdown]
# ## Question 2

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, RidgeCV, lasso_path
from sklearn.metrics import mean_squared_error

# --- Configuration ---
np.random.seed(42)

# ==========================================
# 1. GÉNÉRATION DES DONNÉES (MODIFIÉE)
# ==========================================
n = 1000
p = 5000
s = 1500  # <--- CHANGEMENT MAJEUR : Sparsité élevée (signal dense)

print(f"Simulation avec n={n}, p={p} et s={s} variables actives.")

# Génération de X
X = np.random.randn(n, p)

# Génération de beta
beta_true = np.zeros(p)
beta_true[:s] = 1.0  # Les 1500 premiers coefficients à 1

# Génération du bruit et de Y
eta = np.random.randn(n)
Y = X @ beta_true + eta

# Séparation (2/3 train, 1/3 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.34, random_state=42
)

# ==========================================
# b. CHEMIN DE RÉGULARISATION DU LASSO
# ==========================================
print("Calcul du chemin de régularisation LASSO...")
# On limite eps pour ne pas calculer trop de chemin inutilement coûteux ici
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps=1e-2)

plt.figure(figsize=(10, 6))
log_alphas_lasso = np.log10(alphas_lasso)
# On plot seulement un sous-ensemble de coefs pour la lisibilité
for coef_l in coefs_lasso[::10]: 
    plt.plot(log_alphas_lasso, coef_l)

plt.xlabel('Log(Lambda)')
plt.ylabel('Coefficients')
plt.title(f'Chemin de régularisation LASSO (s={s})')
plt.axis('tight')
plt.show()

# ==========================================
# a. & c. ESTIMATION ET OPTIMISATION (Train)
# ==========================================
print("\nOptimisation des modèles par Validation Croisée (CV)...")

# 1. LASSO (l1_ratio = 1)
model_lasso_cv = ElasticNetCV(l1_ratio=1.0, cv=5, random_state=42, n_jobs=-1)
model_lasso_cv.fit(X_train, y_train)
print(f"LASSO optimal lambda: {model_lasso_cv.alpha_:.4f}")

# 2. RIDGE (l1_ratio = 0)
# RidgeCV est préférable pour la stabilité quand alpha=0
model_ridge_cv = RidgeCV(alphas=np.logspace(-2, 2, 20), cv=5)
model_ridge_cv.fit(X_train, y_train)
print(f"RIDGE optimal mu: {model_ridge_cv.alpha_:.4f}")

# 3. ELASTIC NET (Recherche sur grille 2D)
# Ici, on teste des ratios favorisant la norme L2 car on suspecte s > n
ratios_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
model_en_cv = ElasticNetCV(l1_ratio=ratios_to_test, cv=5, random_state=42, n_jobs=-1)
model_en_cv.fit(X_train, y_train)

print(f"ELASTIC-NET optimal : lambda={model_en_cv.alpha_:.4f}, alpha (ratio)={model_en_cv.l1_ratio_}")

# ==========================================
# d. COMPARAISON SUR TEST
# ==========================================
print("-" * 50)
print("Évaluation sur l'échantillon TEST (MSE) :")

y_pred_lasso = model_lasso_cv.predict(X_test)
y_pred_ridge = model_ridge_cv.predict(X_test)
y_pred_en = model_en_cv.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_en = mean_squared_error(y_test, y_pred_en)

print(f"MSE LASSO       : {mse_lasso:.4f}")
print(f"MSE RIDGE       : {mse_ridge:.4f}")
print(f"MSE ELASTIC-NET : {mse_en:.4f}")

errors = {"LASSO": mse_lasso, "RIDGE": mse_ridge, "ELASTIC-NET": mse_en}
best_model_name = min(errors, key=errors.get)

print(f"\n>>> Le meilleur modèle est : {best_model_name}")

# %% [markdown]
# ## Question 3

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, RidgeCV, lasso_path
from sklearn.metrics import mean_squared_error

# --- Configuration ---
np.random.seed(42)

# ==========================================
# 1. GÉNÉRATION DES DONNÉES (CORRÉLÉES)
# ==========================================
n = 100
p = 50

print(f"Simulation : n={n}, p={p}, prédicteurs corrélés.")

# A. Création de la matrice de covariance Toeplitz
# Cov(Xi, Xj) = 0.7^|i-j|
indices = np.arange(p)
rows, cols = np.meshgrid(indices, indices)
covariance_matrix = 0.7 ** np.abs(rows - cols)

# B. Génération de X multivarié
# mean = vecteur de 0, cov = matrice calculée
X = np.random.multivariate_normal(np.zeros(p), covariance_matrix, n)

# C. Création de Beta
beta_true = np.zeros(p)
# beta_1 = beta_2 = 10 (indices 0 et 1)
beta_true[0:2] = 10.0
# beta_3 = beta_4 = 5 (indices 2 et 3)
beta_true[2:4] = 5.0
# beta_5 à beta_14 = 1 (indices 4 à 13 inclus)
beta_true[4:14] = 1.0

# D. Génération de Y
eta = np.random.randn(n)
Y = X @ beta_true + eta

# Séparation
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.34, random_state=42
)

# ==========================================
# b. CHEMIN DE RÉGULARISATION DU LASSO
# ==========================================
print("Calcul du chemin de régularisation LASSO...")
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps=1e-3)

plt.figure(figsize=(10, 6))
log_alphas_lasso = np.log10(alphas_lasso)
for coef_l in coefs_lasso:
    plt.plot(log_alphas_lasso, coef_l)

plt.xlabel('Log(Lambda)')
plt.ylabel('Coefficients')
plt.title('Chemin LASSO (Variables corrélées)')
plt.axis('tight')
plt.show()

# ==========================================
# a. & c. ESTIMATION ET OPTIMISATION
# ==========================================
print("\nOptimisation des modèles par CV...")

# 1. LASSO
# On augmente le nombre d'itérations (max_iter) car la convergence 
# est plus difficile avec des données corrélées.
model_lasso_cv = ElasticNetCV(l1_ratio=1.0, cv=5, random_state=42, max_iter=10000)
model_lasso_cv.fit(X_train, y_train)
print(f"LASSO optimal lambda: {model_lasso_cv.alpha_:.4f}")

# 2. RIDGE
model_ridge_cv = RidgeCV(alphas=np.logspace(-2, 2, 50), cv=5)
model_ridge_cv.fit(X_train, y_train)
print(f"RIDGE optimal mu: {model_ridge_cv.alpha_:.4f}")

# 3. ELASTIC NET
# Nous testons une grille fine car le compromis L1/L2 est crucial ici
ratios_to_test = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
model_en_cv = ElasticNetCV(l1_ratio=ratios_to_test, cv=5, random_state=42, max_iter=10000)
model_en_cv.fit(X_train, y_train)

print(f"ELASTIC-NET optimal : lambda={model_en_cv.alpha_:.4f}, alpha (ratio)={model_en_cv.l1_ratio_}")

# ==========================================
# d. COMPARAISON SUR TEST
# ==========================================
print("-" * 50)
print("Évaluation sur l'échantillon TEST (MSE) :")

y_pred_lasso = model_lasso_cv.predict(X_test)
y_pred_ridge = model_ridge_cv.predict(X_test)
y_pred_en = model_en_cv.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_en = mean_squared_error(y_test, y_pred_en)

print(f"MSE LASSO       : {mse_lasso:.4f}")
print(f"MSE RIDGE       : {mse_ridge:.4f}")
print(f"MSE ELASTIC-NET : {mse_en:.4f}")

errors = {"LASSO": mse_lasso, "RIDGE": mse_ridge, "ELASTIC-NET": mse_en}
best_model_name = min(errors, key=errors.get)

print(f"\n>>> Le meilleur modèle est : {best_model_name}")

# Analyse des coefficients retrouvés par ElasticNet vs Vrais coefficients
print("\nAnalyse des coefficients (Top 5 vs Réalité):")
print(f"Vrais (beta 0 à 4) : {beta_true[:5]}")
print(f"Estimés ElasticNet : {model_en_cv.coef_[:5]}")

# %% [markdown]
# # EXERCICE 2 : Données Réelles

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV, lasso_path
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. PRÉPARATION DES DONNÉES (California Housing)
# ==========================================
# Chargement du dataset
# Variables : Revenu médian, Âge maison, Nb pièces moyen, Nb chambres moyen, Population, Occupation moyenne, Latitude, Longitude.
# Cible : Prix médian de la maison (en centaines de milliers de dollars)
housing = fetch_california_housing()
X_full, y_full = housing.data, housing.target
feature_names = housing.feature_names

# --- ASTUCE POUR L'EXERCICE ---
# On réduit n à 1000 pour simuler un cas où on a peu de données par rapport à la complexité
# Si on gardait les 20.000 lignes, la régularisation serait moins visible (car n >> p).
np.random.seed(42)
indices = np.random.choice(X_full.shape[0], 1000, replace=False)
X_raw = X_full[indices]
y = y_full[indices]

print(f"Dimensions réduites : {X_raw.shape} (1000 maisons sélectionnées)")

# --- Enrichissement (Polynomial Features) ---
# On crée des carrés et des interactions (ex: Latitude * Longitude, Revenu * Age)
# Cela fait exploser le nombre de variables
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_raw)
new_feature_names = poly.get_feature_names_out(feature_names)

print(f"Dimensions après interactions : {X_poly.shape} (On passe de 8 à 44 variables)")

# --- Standardisation (OBLIGATOIRE) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Séparation Train / Test (66% train, 34% test comme dans l'exo 1)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.34, random_state=42
)

# ==========================================
# 2. VISUALISATION : CHEMIN DU LASSO
# ==========================================
print("Calcul du chemin de régularisation...")
# On trace comment les coefficients évoluent quand lambda diminue
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps=1e-3)

plt.figure(figsize=(10, 6))
log_alphas = np.log10(alphas_lasso)

for coef in coefs_lasso:
    plt.plot(log_alphas, coef)

plt.xlabel('Log(Lambda)')
plt.ylabel('Coefficients')
plt.title('Chemin LASSO : Prix Immobilier Californie')
plt.axis('tight')
plt.show()

# ==========================================
# 3. ENTRAÎNEMENT DES MODÈLES (CV)
# ==========================================
print("\nOptimisation des modèles...")

# A. LASSO
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=20000)
lasso_cv.fit(X_train, y_train)

# B. RIDGE
ridge_cv = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
ridge_cv.fit(X_train, y_train)

# C. ELASTIC-NET
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.99]
en_cv = ElasticNetCV(l1_ratio=l1_ratios, cv=5, random_state=42, max_iter=20000)
en_cv.fit(X_train, y_train)

print(f" -> Best Lambda LASSO : {lasso_cv.alpha_:.4f}")
print(f" -> Best Lambda RIDGE : {ridge_cv.alpha_:.4f}")
print(f" -> Best Lambda E-NET : {en_cv.alpha_:.4f} (Ratio L1: {en_cv.l1_ratio_})")

# ==========================================
# 4. COMPARAISON PERFORMANCE
# ==========================================
models = {
    "Lasso": lasso_cv,
    "Ridge": ridge_cv,
    "ElasticNet": en_cv
}

results = {}

print("\n" + "="*50)
print(f"{'Modèle':<15} | {'MSE (Test)':<12} | {'Nb Vars Actives'}")
print("-" * 50)

for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Nombre de variables non nulles
    n_features = np.sum(np.abs(model.coef_) > 1e-4)
    results[name] = mse
    
    print(f"{name:<15} | {mse:<12.3f} | {n_features}/{X_train.shape[1]}")

print("="*50)
best = min(results, key=results.get)
print(f">>> Vainqueur : {best}")

# ==========================================
# 5. INTERPRÉTATION MÉTIER
# ==========================================
# Quelles sont les variables qui font exploser le prix ?
final_coefs = pd.Series(en_cv.coef_, index=new_feature_names)
print("\nTop 5 Variables les plus importantes (ElasticNet) :")
# On regarde la valeur absolue pour voir l'impact (positif ou négatif)
print(final_coefs.abs().sort_values(ascending=False).head(5))

# Visualisation des résidus pour vérifier la qualité
plt.figure(figsize=(8, 5))
plt.scatter(y_test, en_cv.predict(X_test), alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Prix Réel")
plt.ylabel("Prix Prédit")
plt.title("Qualité de prédiction (ElasticNet)")
plt.show()



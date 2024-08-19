import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss
from patsy import dmatrix
import matplotlib.pyplot as plt

data = pd.read_csv('phoneme_data.csv')
X = data[[f'x.{i}' for i in range(1, 257)]]
y = data['g']
speaker = data['speak']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.decomposition import PCA
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
num_knots_choices = [2, 3, 4, 5, 6]
def create_spline_basis(X, num_knots):
    X_spline_list = []
    for i in range(X.shape[1]):
        min_val, max_val = np.min(X[:, i]), np.max(X[:, i])
        if np.isnan(min_val) or np.isnan(max_val):
            print(f"Skipping feature {i+1} due to NaN values.")
            continueaN
        if min_val == max_value: 
            max_val = min_val + 1e-5  
        knots = np.linspace(min_val, max_val, num=num_knots)
        try:
            spline = dmatrix(f"bs(x, knots={list(knots[1:-1])}, degree=3, include_intercept=False)", {"x": X[:, i]})
        except Exception as e:
            print(f"Error creating spline basis for feature {i+1} with knots={knots}: {e}")
            continue
        spline_df = pd.DataFrame(spline, columns=[f"x{i+1}_spline{j}" for j in range(spline.shape[1])])
        X_spline_list.append(spline_df)
    if X_spline_list:
        X_spline = pd.concat(X_spline_list, axis=1)
        return X_spline
    else:
        return pd.DataFrame()  

losses = []
for num_knots in num_knots_choices:
    try:
        print(f"Processing num_knots={num_knots}...")
        
        print("Creating spline basis for training data...")
        X_train_spline = create_spline_basis(X_train, num_knots)
        if X_train_spline.empty:
            print(f"No valid spline basis features created for num_knots={num_knots}. Skipping.")
            continue
        print(f"X_train_spline shape: {X_train_spline.shape}")
        
        print("Creating spline basis for test data...")
        X_test_spline = create_spline_basis(X_test, num_knots)
        if X_test_spline.empty:
            print(f"No valid spline basis features created for num_knots={num_knots}. Skipping.")
            continue
        print(f"X_test_spline shape: {X_test_spline.shape}")

        print("Fitting QDA model with cross-validation...")
        qda = QuadraticDiscriminantAnalysis()
        cv_scores = cross_val_score(qda, X_train_spline, y_train, cv=5, scoring='neg_log_loss')
        mean_cv_score = -cv_scores.mean()
        print(f"Mean CV log loss for num_knots={num_knots}: {mean_cv_score}")
        
        losses.append((num_knots, mean_cv_score))
    except Exception as e:
        print(f"Error processing num_knots={num_knots}: {e}")


if not losses:
    raise ValueError("No valid log loss values were computed. Please check the data and the knot choices.")


best_knots_idx = np.argmin([loss for _, loss in losses])
best_num_knots = losses[best_knots_idx][0]


print("Losses for different knot choices:", losses)
print("Best number of knots choice (index, num_knots):", best_knots_idx, best_num_knots)

if losses:  
    plt.plot([knot for knot, _ in losses], [loss for _, loss in losses], marker='o')
    plt.xlabel('Number of Knots')
    plt.ylabel('Log Loss')
    plt.title('Log Loss for Different Knot Choices')
    plt.show()
else:
    print("No valid log loss values were computed; skipping the plot.")

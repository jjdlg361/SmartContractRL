import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

CSV_PATH = '../data/smart_contract_dataset_with_cfgs.csv'
df = pd.read_csv(CSV_PATH)

def count_cfg(path):
    try:
        text = open(path, 'r').read()
    except Exception as e:
        return None, None
    edges_count = len(re.findall(r"\b[\w]+\s*->\s*[\w]+", text))
    node_names = re.findall(r"\b([\w]+)\s*\[", text)
    nodes_count = len(set(node_names))
    return nodes_count, edges_count

df[['num_nodes', 'num_edges']] = df['cfg_path'].apply(lambda p: pd.Series(count_cfg(p)))
df = df.dropna(subset=['num_nodes','num_edges'])
X = df[['num_nodes', 'num_edges']].values
y = df['slither']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
models = {
    'SVM': (SVC(), {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'class_weight': [None, 'balanced']
    }),
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'class_weight': [None, 'balanced']
    }),
    'MLP': (MLPClassifier(max_iter=500, random_state=42), {
        'hidden_layer_sizes': [(32,), (64,32)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    })
}
results = []
for name, (clf, params) in models.items():
    print(f"\nStarting grid search for {name}...")
    gs = GridSearchCV(clf, params, cv=5, scoring='f1_weighted', n_jobs=-1)
    gs.fit(X_train_scaled, y_train)
    best = gs.best_estimator_
    print(f"Best params for {name}: {gs.best_params_}")

    y_pred = best.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1  = f1_score(y_test, y_pred, average='weighted')
    results.append((name, acc, rec, f1, gs.best_params_))
    print(f"Classification report for {name}:")
    print(classification_report(y_test, y_pred))
print("\nSummary of traditional ML methods on CFG counts:")
for name, acc, rec, f1, params in results:
    print(f"{name}: Accuracy={acc:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

out_df = pd.DataFrame(results, columns=['Model','Accuracy','Recall','F1','Best_Params'])
out_df.to_csv('ml_method_comparison.csv', index=False)
print("Comparison CSV written to ml_method_comparison.csv")

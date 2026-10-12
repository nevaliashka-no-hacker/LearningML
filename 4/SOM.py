import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42


def evaluate_binary(y_true, y_pred, title="Модель", average="macro"):
        print("-" * len(title))
        print(title)
        print("-" * len(title))

        print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
        print(f"Precision: {precision_score(y_true, y_pred, average=average):.3f}")
        print(f"Recall   : {recall_score(y_true, y_pred, average=average):.3f}")
        print(f"F1-score : {f1_score(y_true, y_pred, average=average):.3f}")

        print("\nПодробный отчет:")
        print(classification_report(y_true, y_pred, digits=3))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            cmap="Blues"
        )
        plt.title(title)
        plt.show()

def main():
    filename = "satellite_telemetry_modes.csv"
    df = pd.read_csv(filename)
    print("Размер датасета:", df.shape)
    print()
    print(df.head())
    print()
    print(df.info())
    print()

    print("\nПропущенные значения:")
    print(df.isna().sum())
    print()

    print("\nКоличество объектов каждого класса:")
    print(df["mode"].value_counts())
    print()

    plt.figure(figsize=(8, 6))

    for class_value, label in [("IDLE", "Класс IDLE"), ("OBSERVATION", "Класс OBSERVATION"), ("TRANSMISSION", "Класс TRANSMISSION")]:
        part = df[df["mode"] == class_value]
        plt.scatter(
            part["cpu_load"],
            part["memory_load"],
            label=label,
            alpha=0.5,
            s=25
        )

    plt.xlabel("CPU load")
    plt.ylabel("Memory load")
    plt.title("Telemetry: распределение классов")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.figure(figsize=(8, 6))
    
    for class_value, label in [("IDLE", "Класс IDLE"), ("OBSERVATION", "Класс OBSERVATION"), ("TRANSMISSION", "Класс TRANSMISSION")]:
        part = df[df["mode"] == class_value]
        plt.scatter(
            part["communication_load"],
            part["power_load"],
            label=label,
            alpha=0.5,
            s=25
        )
    
    plt.xlabel("Communication load")
    plt.ylabel("Power load")
    plt.title("Telemetry: распределение классов")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.show()
    
    X = df[["cpu_load", "memory_load", "communication_load", "power_load"]]
    y = df["mode"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print()
    print("Обучающая выборка:", X_train.shape)
    print("Тестовая выборка:", X_test.shape)
    print()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    evaluate_binary(y_test, y_pred_knn, "k-NN, k = 5")

    # Logistic Regression
    log_reg = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log = log_reg.predict(X_test_scaled)
    evaluate_binary(y_test, y_pred_log, "Логистическая регрессия")
    probabilities = log_reg.predict_proba(X_test_scaled[:10])
    result = pd.DataFrame(
        probabilities,
        columns=["P(class=IDLE)", "P(class=OBSERVATION)", "P(class=TRANSMISSION)"]
    )
    result["prediction"] = log_reg.predict(X_test_scaled[:10])
    result["real_class"] = y_test.iloc[:10].values
    result

    #  Decision Tree
    tree = DecisionTreeClassifier(
        max_depth=4,
        random_state=RANDOM_STATE
    )
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    evaluate_binary(y_test, y_pred_tree, "Дерево решений, max_depth = 4")
    feature_importance_tree = pd.Series(
        tree.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    feature_importance_tree

    #  Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    evaluate_binary(y_test, y_pred_rf, "Random Forest")
    feature_importance_rf = pd.Series(
        random_forest.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    feature_importance_rf

    # Naive Bayes
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X_train, y_train)
    y_pred_gnb = gaussian_nb.predict(X_test)
    evaluate_binary(y_test, y_pred_gnb, "Gaussian Naive Bayes")

    # Общее сравнение
    som_predictions = {
        "k-NN": y_pred_knn,
        "Logistic Regression": y_pred_log,
        "Decision Tree": y_pred_tree,
        "Random Forest": y_pred_rf,
        "Gaussian Naive Bayes": y_pred_gnb
    }

    rows = []

    for model_name, prediction in som_predictions.items():
        rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, average="macro"),
            "Recall": recall_score(y_test, prediction, average="macro"),
            "F1": f1_score(y_test, prediction, average="macro")
        })

    som_results = pd.DataFrame(rows).sort_values(
        "F1",
        ascending=False
    )

    som_results.round(3)

    arr = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    print("|        Model        |      Accuracy       |      Precision      |       Recall        |         F1          |")
    print("|---------------------|---------------------|---------------------|---------------------|---------------------|")
    for i in rows:
        print("|", end='')
        for j in arr:
            print(f"{i[j]:21}", end='|')
        print('\n', end='')
        
    


if __name__ == "__main__":
    main()
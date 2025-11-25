# flake8: noqa: E501

# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class DataProcessor:
    """Handles data loading and cleaning operations."""

    @staticmethod
    def load_data(
        train_path: str, test_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test datasets."""
        train_df = pd.read_csv(train_path, compression="zip")
        test_df = pd.read_csv(test_path, compression="zip")
        return train_df, test_df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset according to specifications."""
        df = df.copy()

        if "default payment next month" in df.columns:
            df = df.rename(columns={"default payment next month": "default"})

        if "ID" in df.columns:
            df = df.drop("ID", axis=1)

        df = df.dropna()

        for col in ["SEX", "EDUCATION", "MARRIAGE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df[df[col] != 0]

        if "EDUCATION" in df.columns:
            df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

        return df

    @staticmethod
    def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataset into features and target."""
        X = df.drop("default", axis=1)
        y = df["default"]
        return X, y


class ModelBuilder:
    """Handles model creation and training."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.categorical_features = ["EDUCATION"]

    def create_pipeline(self) -> Pipeline:
        """Create preprocessing and modeling pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                )
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=self.random_state)),
            ]
        )

        return pipeline

    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> GridSearchCV:
        """Perform hyperparameter optimization using grid search with cross-validation."""
        pipeline = self.create_pipeline()

        param_grid = {
            "classifier__n_estimators": [200],
            "classifier__max_depth": [None],
            "classifier__min_samples_leaf": [2],
            "classifier__min_samples_split": [5],
            "classifier__max_features": ["sqrt"],
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=10,
            verbose=1,
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        return grid_search


class MetricsCalculator:
    """Handles metrics calculation and formatting."""

    @staticmethod
    def calculate_performance_metrics(
        estimator, X: pd.DataFrame, y: pd.Series, dataset_name: str
    ) -> dict:
        """Calculate precision, balanced accuracy, recall, and F1-score."""
        y_pred = estimator.predict(X)

        return {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": round(precision_score(y, y_pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
            "recall": round(recall_score(y, y_pred), 4),
            "f1_score": round(f1_score(y, y_pred), 4),
        }

    @staticmethod
    def calculate_confusion_matrix(
        estimator, X: pd.DataFrame, y: pd.Series, dataset_name: str
    ) -> dict:
        """Calculate and format confusion matrix."""
        y_pred = estimator.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        return {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
            "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
        }


class ModelPersistence:
    """Handles model and metrics persistence."""

    @staticmethod
    def save_model(model, filepath: str) -> None:
        """Save model as compressed pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with gzip.open(filepath, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def save_metrics(metrics_list: list, filepath: str) -> None:
        """Save metrics in JSON Lines format."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for metric in metrics_list:
                f.write(json.dumps(metric) + "\n")


def main():
    """Main execution pipeline."""

    processor = DataProcessor()
    train_df, test_df = processor.load_data(
        "files/input/train_data.csv.zip", "files/input/test_data.csv.zip"
    )
    print("✓ Data loaded")

    train_df = processor.clean_data(train_df)
    test_df = processor.clean_data(test_df)
    print("✓ Data cleaned")

    X_train, y_train = processor.split_features_target(train_df)
    X_test, y_test = processor.split_features_target(test_df)
    print("✓ Features and target split")

    model_builder = ModelBuilder(random_state=42)
    best_model = model_builder.optimize_hyperparameters(X_train, y_train)
    print("✓ Model optimized")

    ModelPersistence.save_model(best_model, "files/models/model.pkl.gz")
    print("✓ Model saved")

    metrics_calc = MetricsCalculator()

    metrics_train = metrics_calc.calculate_performance_metrics(
        best_model, X_train, y_train, "train"
    )
    metrics_test = metrics_calc.calculate_performance_metrics(
        best_model, X_test, y_test, "test"
    )
    print("✓ Performance metrics calculated")

    cm_train = metrics_calc.calculate_confusion_matrix(
        best_model, X_train, y_train, "train"
    )
    cm_test = metrics_calc.calculate_confusion_matrix(
        best_model, X_test, y_test, "test"
    )
    print("✓ Confusion matrices calculated")

    all_metrics = [metrics_train, metrics_test, cm_train, cm_test]
    ModelPersistence.save_metrics(all_metrics, "files/output/metrics.json")
    print("✓ Metrics saved")

    print("\n✓ Pipeline completed successfully")


if __name__ == "__main__":
    main()
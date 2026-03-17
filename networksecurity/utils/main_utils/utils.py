import yaml
import os
import sys
import numpy as np
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


# ✅ YAML FUNCTIONS
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ✅ NUMPY FUNCTIONS
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ✅ PICKLE FUNCTIONS
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Saving object...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Object saved successfully")

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ✅ FAST + OPTIMIZED MODEL EVALUATION
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}   # ✅ NEW

        for name, model in models.items():
            logging.info(f"Training model: {name}")
            print(f"🚀 Training {name}...")

            try:
                if name in param and len(param[name]) > 0:
                    gs = GridSearchCV(model, param[name], cv=3, n_jobs=-1)
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    best_model = model.fit(X_train, y_train)

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                print(f"{name} → Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")

                report[name] = test_score
                trained_models[name] = best_model   # ✅ NEW

            except Exception as model_error:
                print(f"❌ Error in model {name}: {model_error}")
                continue

        return report, trained_models   # ✅ UPDATED

    except Exception as e:
        raise NetworkSecurityException(e, sys)
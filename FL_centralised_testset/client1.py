import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from modelAnn import get_ann
from sklearn.utils import shuffle

# Load dataset
botnet = pd.read_csv("I:\\My Drive\\Colab Notebooks\\datasets\\data\\Prepared\\botnet_trainval_set.csv")
print(botnet.head())

botnet = shuffle(botnet).reset_index(drop=True)
# Split dataset
X = botnet.drop(columns=['Label'])
y = botnet['Label']
print(f"X shape: {X.shape}")
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.1, random_state=42, shuffle=True) 

# Normalize data
scalar = StandardScaler()
scalar.fit(X_train)
X_train_scaled = scalar.transform(X_train)
X_val_scaled = scalar.transform(X_val)
X_test_scaled = scalar.transform(X_test)

print(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")

VERBOSE = 0
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val,X_test, y_test):
        self.model = get_ann()
        # self.model.build(self, input_shape=(None, 43))
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=15, batch_size=128, verbose=VERBOSE)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        # Generate predictions
        y_pred = self.model.predict(self.X_test,verbose=VERBOSE)
        y_pred = (y_pred > 0.5)

        # Compute classification report
        class_report = classification_report(self.y_test, y_pred, digits=5)

        return loss, len(self.X_test), {"loss": loss, "accuracy": accuracy, "classification_report": class_report}

fl.client.start_client(server_address="localhost:8080",
                        client=FlowerClient(get_ann(), X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test).to_client())
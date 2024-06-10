import flwr as fl
from my_server import weighted_average, get_evaluate_fn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from sklearn.utils import shuffle

# Load dataset
# clients_2 = pd.read_csv("I:\\My Drive\\Colab Notebooks\\datasets\\data\\Prepared\\clients_2.csv")
clients_3 = pd.read_csv("I:\\My Drive\\Colab Notebooks\\datasets\\data\\Prepared\\clients_3.csv")
# clients_4 = pd.read_csv("I:\\My Drive\\Colab Notebooks\\datasets\\data\\Prepared\\clients_4.csv")

clients_3 = shuffle(clients_3).reset_index(drop=True)
X = clients_3.drop(columns="Label")
y = clients_3['Label']

# Normalizing the dataset
scalar = StandardScaler()
scalar.fit(X)
X_scaled = scalar.transform(X)

NUM_CLIENTS = 3

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than the number of clients for training
    min_evaluate_clients=NUM_CLIENTS,  # Never sample less than the number of clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all the number of clients are available
    # evaluate_metrics_aggregation_fn=weighted_average, # aggregates federated metrics
    evaluate_fn=get_evaluate_fn(X_scaled, y),## global evaluation function
)

history = fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    )

with open('history2_test.pickle', 'wb') as f:
    pickle.dump(history, f)

import flwr as fl
from sklearn.metrics import classification_report
from modelAnn import get_ann


VERBOSE = 0
NUM_CLIENTS = 3
def weighted_average(metrics):
    total_samples = sum(num_samples for num_samples, _ in metrics)
    agg_loss = sum(num_samples * m.get("loss",0) for num_samples, m in metrics) / total_samples
    agg_accuracy = sum(num_samples * m.get("accuracy",1) for num_samples, m in metrics) / total_samples

    # Collect classification reports
    classification_reports = [m.get("classification_report", "") for _, m in metrics]

    # Print each classification report with client number
    # for i, report in enumerate(classification_reports):
    #     print(f"Classification report for client {i+1}:\n{report}\n")

    return {"agg_loss": agg_loss, "agg_accuracy": agg_accuracy}


from typing import Dict, List, Tuple

def get_evaluate_fn(X_test, y_test):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_ann()  # Construct the model
        # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(X_test, y_test, verbose=VERBOSE)
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5)
        class_report = classification_report(y_test, y_pred, digits=5)

        return loss, {"loss":loss, "accuracy": accuracy, "Centralised report": class_report}

    return evaluate
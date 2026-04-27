import flwr as fl
import logging

# Configure logging for the central server
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Omni-Shield FL Server - %(levelname)s - %(message)s')

def get_evaluate_fn(model):
    """
    Optional: The central server can evaluate the aggregated global model 
    on a small, completely public/synthetic dataset to check accuracy 
    before sending it back to the edge nodes.
    """
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        # In a real deployment, you would load the aggregated weights into a 
        # local model and test it against a synthetic dataset.
        logging.info(f"Evaluating Global Model for Round {server_round}...")
        
        # Placeholder for global evaluation metrics
        loss = 0.0
        accuracy = 0.95 # Simulated baseline
        return loss, {"accuracy": accuracy}
    return evaluate

def main():
    logging.info("🚀 Starting Omni-Shield Federated Aggregation Server...")

    # Define the Federated Averaging (FedAvg) Strategy
    # This dictates how the server merges the mathematical knowledge from all edge devices
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,            # Sample 100% of available clients for training
        fraction_evaluate=0.5,       # Sample 50% of available clients for evaluation
        min_fit_clients=2,           # Minimum number of edge nodes required to start training
        min_evaluate_clients=2,      # Minimum number of edge nodes required to evaluate
        min_available_clients=2,     # Wait until at least 2 edge nodes are connected
        # evaluate_fn=get_evaluate_fn(None), # Uncomment if you want central server evaluation
    )

    # Start the Flower server on port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5), # Run for 5 global training rounds
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
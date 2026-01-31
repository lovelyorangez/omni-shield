import flwr as fl

# Define strategy (Federated Averaging)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,    # Sample 100% of available clients
    min_fit_clients=1,   # Minimum clients to start training (For demo, 1 is fine)
    min_available_clients=1,
)

print("🚀 OMNI-SHIELD Federated Aggregator Starting...")
print("   - Waiting for secure weight updates...")
print("   - NO RAW DATA will be accepted.")

# Start the server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
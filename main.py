import os
from src.train import train_models
from src.evaluate import evaluate_models

if __name__ == "__main__":
    os.makedirs("models/prophet_models", exist_ok=True)
    # Train Prophet models for each store-product
    train_models(
        data_path="data/retail_warehouse_inventory_dataset.csv"
    )
    # Evaluate and plot
    evaluate_models(
        data_path="data/retail_warehouse_inventory_dataset.csv",
        models_dir="models/prophet_models"
    )

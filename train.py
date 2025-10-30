# --- Standard Libraries ---
import argparse

# --- Third-Party Libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# --- Local Source ---
from src.data import get_cifar10_loaders
from src.model import get_wrn_28_10


def train(args):
    """ Main training function """
    wandb.init(
        entity="pesu-ai-ml",
        project="sam-replication-cifar10",
        config=args
    )
    print(f"Starting training with learning rate: {wandb.config.learning_rate}")

    print("Setting up the experiment...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_cifar10_loaders(batch_size=wandb.config.batch_size)
    
    model = get_wrn_28_10().to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.9, weight_decay=5e-4)
    
    print("Starting training...")
    for epoch in range(wandb.config.epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})
        print(f"Epoch {epoch+1}/{wandb.config.epochs} | Loss: {loss.item():.4f}")

    print("Finished training.")
    wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SAM Replication Training Harness')
    parser.add_argument('--epochs', type=int, default=10, dest='epochs',
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, dest='learning_rate',
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, dest='batch_size',
                        help='Batch size for training')
    args = parser.parse_args()
    train(args)


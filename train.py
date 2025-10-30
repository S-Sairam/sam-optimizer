# --- Standard Libraries ---
import argparse
import time 
# --- Third-Party Libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
# --- Local Source ---
from src.data import get_cifar10_loaders
from src.model import get_wrn_28_10
from src.sam import SAM

def evaluate(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad() :
        val_loss = 0
        total_samples = 0
        correct_predictions = 0
        for batch_idx, (inputs,targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs,targets).item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
        val_accuracy = correct_predictions / total_samples
    return val_loss/len(val_loader), val_accuracy*100.00

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

    optimizer = SAM(model.parameters(), lr=wandb.config.learning_rate, base_optimizer = optim.SGD, momentum=0.9, weight_decay=5e-4, rho = wandb.config.rho)

    scheduler = CosineAnnealingLR(optimizer, T_max = 200)


    print("Starting training...")
    for epoch in range(wandb.config.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
            wandb.log({"train_loss": loss.item()})
            progress_bar.set_postfix(loss=loss.item())
        
        val_loss, val_accuracy = evaluate(model,val_loader, criterion, device)
        wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch+1}/{wandb.config.epochs} | Loss: {loss.item():.4f} | val_loss: {val_loss} | val_accuracy: {val_accuracy}")
        scheduler.step()
    
    print("Finished training.")
    wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SAM Replication Training Harness')
    parser.add_argument('--epochs', type=int, default=200, dest='epochs',
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, dest='learning_rate',
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, dest='batch_size',
                        help='Batch size for training')

    parser.add_argument('--rho', type=float, default=0.05, dest='rho', help='Rho for SAM')
    args = parser.parse_args()
    train(args)


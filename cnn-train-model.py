import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import time
from datetime import datetime

# Set device
device = torch.device("cpu")
print(f"Using device: {device}")

# Load and prepare data
df = pd.read_csv('newcnndataset.csv')
print("Original dataset shape:", df.shape)
print(df.head())

# Check for NaN values in the class column
nan_count = df.iloc[:, 0].isna().sum()
print(f"Found {nan_count} NaN values in class column")

# Remove rows with NaN class values
df = df.dropna(subset=[df.columns[0]])
print("Dataset shape after removing NaN classes:", df.shape)

X = df.iloc[:,1:].values  # Convert to numpy array
y = df.iloc[:,0].values

# Get number of classes and unique class values
unique_classes = np.unique(y)
num_classes = len(unique_classes)
print(f"Number of classes: {num_classes}")
print(f"Classes: {unique_classes}")

# Create string labels for reports and plotting
string_labels = ["Situps Down", "Situps Up", "Pushups Down", "Pushups Up", "Squat Up", "Squat Down", "Jumping Jack Up", "Jumping Jack Down"]
print("Class labels:", string_labels)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Reshape data for CNN (samples, timesteps, features)
# Treat each frame as having 33 timesteps (landmarks) with 2 features (x,y)
X_reshaped = X.reshape(X.shape[0], 33, 2)

# Three-way split
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X_reshaped, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.25, random_state=42, stratify=y_train_temp
)

# NOW normalize ALL datasets using training statistics
scaler = StandardScaler()

# Reshape for fitting: (samples, features)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Fit scaler only on training data, transform all datasets
X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled = scaler.transform(X_val_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Reshape back to (samples, timesteps, features)
X_train_scaled = X_train_scaled.reshape(X_train.shape[0], 33, 2)
X_val_scaled = X_val_scaled.reshape(X_val.shape[0], 33, 2)
X_test_scaled = X_test_scaled.reshape(X_test.shape[0], 33, 2)

print(f"Training data shape: {X_train_scaled.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val_scaled.shape}, {y_val.shape}")
print(f"Testing data shape: {X_test_scaled.shape}, {y_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).permute(0, 2, 1)
X_val_tensor = torch.FloatTensor(X_val_scaled).permute(0, 2, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled).permute(0, 2, 1)
y_train_tensor = torch.LongTensor(y_train)
y_val_tensor = torch.LongTensor(y_val)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model in PyTorch
class ExerciseCNN(nn.Module):
    def __init__(self, num_classes=8, input_channels=2):
        super(ExerciseCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 33 -> 16
            nn.Dropout(0.1),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 16 -> 8
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Create and initialize the model
model = ExerciseCNN(num_classes).to(device)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model architecture:\n{model}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Define loss function and optimizer
learning_rate = 0.001  # Start conservative
weight_decay = 1e-4  # L2 regularization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# Training functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Training loop with early stopping
num_epochs = 100
patience = 20
best_val_acc = 0.0
patience_counter = 0
best_model_state = None

# History tracking
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'lr': []
}

print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Monitor learning rate changes
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_acc)
    new_lr = optimizer.param_groups[0]['lr']
    
    if old_lr != new_lr:
        print(f"Epoch {epoch+1}: Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
    
    # Store metrics
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['lr'].append(new_lr)
    
    # Calculate gaps
    loss_gap = val_loss - train_loss
    acc_gap = train_acc - val_acc

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        print(f'New best validation accuracy: {best_val_acc:.4f} at epoch {epoch+1}')
        print(f'Accuracy gap (Train - Val): {acc_gap:.4f}, Loss gap (Val - Train): {loss_gap:.4f}')
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_exercise_cnn_model.pth')
    else:
        patience_counter += 1
        print(f'Validation accuracy did not improve: {val_acc:.4f} (best: {best_val_acc:.4f})')
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Best Val Acc: {best_val_acc:.4f}')
        print(f'LR: {new_lr:.6f}')
        print('-' * 50)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Load best model for final evaluation
if best_model_state:
    model.load_state_dict(best_model_state)

# Final evaluation
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"Final Test accuracy: {test_acc:.4f}")

# Generate predictions for classification report and confusion matrix
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Print classification report
print("\nClassification Report:")
print(classification_report(all_targets, all_predictions, target_names=string_labels))

# Generate confusion matrix
cm = confusion_matrix(all_targets, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
ax = plt.subplot()
sn.set(font_scale=1.4)
sn.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 16})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title("PyTorch CNN Model Confusion Matrix")
ax.xaxis.set_ticklabels(string_labels)
ax.yaxis.set_ticklabels(string_labels)
ax.yaxis.label.set(rotation='horizontal', ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# OVERFITTING AND UNDERFITTING ANALYSIS
print("\n" + "="*60)
print("OVERFITTING AND UNDERFITTING ANALYSIS")
print("="*60)

# 1. Training vs Validation Performance Analysis
final_train_loss = history['train_loss'][-1]
final_val_loss = history['val_loss'][-1]
final_train_acc = history['train_acc'][-1]
final_val_acc = history['val_acc'][-1]

print(f"\nFinal Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# 2. Gap Analysis
loss_gap = final_val_loss - final_train_loss
acc_gap = final_train_acc - final_val_acc

print(f"\nLoss Gap (Val - Train): {loss_gap:.4f}")
print(f"Accuracy Gap (Train - Val): {acc_gap:.4f}")

# 3. Overfitting Detection
overfitting_threshold = 0.05  # 5% gap threshold
if acc_gap > overfitting_threshold:
    print(f"\n⚠️  OVERFITTING DETECTED!")
    print(f"   Training accuracy is {acc_gap:.1%} higher than validation accuracy")
elif acc_gap < -overfitting_threshold:
    print(f"\n✅ GOOD GENERALIZATION")
    print(f"   Validation accuracy is slightly higher than training accuracy")
else:
    print(f"\n✅ BALANCED PERFORMANCE")
    print(f"   Training and validation accuracies are well balanced")

# 4. Underfitting Detection
low_performance_threshold = 0.7  # 70% accuracy threshold
if final_train_acc < low_performance_threshold:
    print(f"\n⚠️  UNDERFITTING DETECTED!")
    print(f"   Training accuracy ({final_train_acc:.1%}) is below {low_performance_threshold:.0%}")

# 5. Learning Curve Analysis
def analyze_learning_curves(history):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    # Check if validation loss starts increasing while training loss decreases
    mid_point = len(train_loss) // 2
    
    # Calculate trends in second half of training
    train_loss_trend = np.mean(train_loss[mid_point:]) - np.mean(train_loss[:mid_point])
    val_loss_trend = np.mean(val_loss[mid_point:]) - np.mean(val_loss[:mid_point])
    
    print(f"\nLearning Curve Analysis:")
    print(f"Training loss trend (second half): {'Decreasing' if train_loss_trend < 0 else 'Increasing'}")
    print(f"Validation loss trend (second half): {'Decreasing' if val_loss_trend < 0 else 'Increasing'}")
    
    if train_loss_trend < -0.01 and val_loss_trend > 0.01:
        print("⚠️  Classic overfitting pattern detected in learning curves!")
    
    # Check for convergence
    last_10_val_loss = val_loss[-10:] if len(val_loss) >= 10 else val_loss
    val_loss_std = np.std(last_10_val_loss)
    
    if val_loss_std < 0.005:
        print("✅ Model has converged (validation loss is stable)")
    else:
        print("⚠️  Model may not have converged yet")

analyze_learning_curves(history)

# 6. Model Complexity Analysis
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Complexity:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Data-to-parameter ratio: {len(X_train):.0f}:{total_params}")

if len(X_train) < total_params:
    print("⚠️  Dataset is smaller than number of parameters - high overfitting risk!")
elif len(X_train) < total_params * 10:
    print("⚠️  Small dataset relative to model complexity - monitor for overfitting")
else:
    print("✅ Good data-to-parameter ratio")

# Enhanced plotting
plt.figure(figsize=(16, 12))

# Plot 1: Training History
plt.subplot()
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Training-and-Validation-Loss.png')
plt.show()

plt.subplot()
plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Training-and-Validation-Accuracy.png')
plt.show()

# Plot 3: Loss Gap Over Time
plt.subplot()
loss_gaps = np.array(history['val_loss']) - np.array(history['train_loss'])
plt.plot(loss_gaps, color='red', linewidth=2)
plt.title('Validation-Training Loss Gap')
plt.xlabel('Epoch')
plt.ylabel('Loss Gap')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.savefig('Validation-Training-Loss-Gap.png')
plt.show()

# Plot 4: Accuracy Gap Over Time
plt.subplot()
acc_gaps = np.array(history['train_acc']) - np.array(history['val_acc'])
plt.plot(acc_gaps, color='blue', linewidth=2)
plt.title('Training-Validation Accuracy Gap')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Gap')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=overfitting_threshold, color='red', linestyle='--', alpha=0.7, label='Overfitting threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Training-Validation-Accuracy-Gap.png')
plt.show()

# Plot 5: Learning Rate 
plt.subplot()
plt.plot(history['lr'], color='green', linewidth=2)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('Learning-Rate-Schedule.png')
plt.show()

# Plot 6: Confusion Matrix
plt.subplot()
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=string_labels, yticklabels=string_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('training_analysis.png')
plt.show()

# Summary Report
print("\n" + "="*60)
print("TRAINING SUMMARY REPORT")
print("="*60)

if acc_gap > overfitting_threshold:
    status = "OVERFITTING"
    color = "⚠️"
elif final_train_acc < low_performance_threshold:
    status = "UNDERFITTING"
    color = "⚠️"
else:
    status = "GOOD FIT"
    color = "✅"

print(f"{color} Model Status: {status}")
print(f"Final Test Accuracy: {test_acc:.1%}")
print(f"Training-Validation Gap: {acc_gap:.1%}")
print(f"Model Complexity: {total_params:,} parameters")
print(f"Training completed in {len(history['train_loss'])} epochs")

# Save the entire model
torch.save(model, 'exercise_cnn_model.pt')
print("Complete model saved as 'exercise_cnn_model.pt'")

# Save just the state dict (recommended way)
torch.save(model.state_dict(), 'exercise_cnn_model_state.pth')
print("Model state dict saved as 'exercise_cnn_model_state.pth'")

# Save the scaler and label encoder for preprocessing new data
with open('cnn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'cnn_scaler.pkl'")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved as 'label_encoder.pkl'")

# Save model info
model_info = {
    'num_classes': num_classes,
    'class_names': string_labels,
    'input_shape': (2, 33)  # (channels, length)
}

with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("Model info saved as 'model_info.pkl'")

# Save class mappings for reference
class_mapping = {i: label for i, label in enumerate(string_labels)}
with open('class_mapping.txt', 'w') as f:
    for idx, label in class_mapping.items():
        f.write(f"{idx}: {label}\n")
print("Class mapping saved as 'class_mapping.txt'")

# Save training analysis report
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f'training_analysis_{timestamp}.txt', 'w') as f:
    f.write("PyTorch CNN Model Training Analysis Report\n")
    f.write("="*40 + "\n\n")
    f.write(f"Model Status: {status}\n")
    f.write(f"Final Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Training Accuracy: {final_train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {final_val_acc:.4f}\n")
    f.write(f"Accuracy Gap: {acc_gap:.4f}\n")
    f.write(f"Loss Gap: {loss_gap:.4f}\n")
    f.write(f"Total Parameters: {total_params}\n")
    f.write(f"Training Epochs: {len(history['train_loss'])}\n")
    f.write(f"Dataset Size: {len(X_train)} training samples\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")

print(f"Training analysis saved as 'training_analysis_{timestamp}.txt'")
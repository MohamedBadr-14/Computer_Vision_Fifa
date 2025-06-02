from rl import *
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler



def process_json_objects(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    
    actions = []
    results = []

    for index, item in enumerate(data):
        result = get_state_from_positions(item)
        actions.append(item['action'])
        results.append(result)
        print(f"Processed item {index+1}/{len(data)}")


    return results, actions
        





# x_train,X_test,Y_train,Y_test=train_test_split(states, actions, test_size=0.2, random_state=42)




# unique_actions = set(actions)
# action_map = {action: idx for idx, action in enumerate(unique_actions)}
# actions = [action_map[action] for action in actions]  # Remap actions to 0..N-1
# print(f"Remapped actions to {len(unique_actions)} unique classes: {unique_actions}")
    
class StateActionDataset(Dataset):
    def __init__(self, states, actions):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(states)
        self.y = torch.tensor(actions, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        last = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
def predict_action(state, model_path='models/state_action_model.pth'):
    checkpoint = torch.load(model_path ,weights_only=False)
    
    input_size = checkpoint['config']['input_size']
    hidden_sizes = checkpoint['config']['hidden_sizes']
    num_classes = checkpoint['config']['num_classes']
    
    model = SimpleMLP(input_size, hidden_sizes, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    
    if state.ndim == 1:
        state = state.reshape(1, -1)
    
    scaled_state = scaler.transform(state)
    state_tensor = torch.tensor(scaled_state, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(state_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        top_p, top_class = torch.max(probabilities, dim=1)
        
        sorted_prob, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
        second_best = sorted_indices[0, 1].item()
        second_prob = sorted_prob[0, 1].item()
        
    action_id = top_class.item()
    action_prob = top_p.item()
    
    print(f"Predicted action: {action_id} with probability {action_prob:.4f}")
    print(f"Second best action: {second_best} with probability {second_prob:.4f}")
    
    return action_id, probabilities.numpy()
    

if __name__ == "__main__":

   
    states, actions = process_json_objects('dba.json')    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    lr = 1e-3
    num_epochs = 500
    hidden_sizes = [500, 400,300,200,100, 17] 

    dataset = StateActionDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = dataset.X.shape[1]
    num_classes = 17
    model = SimpleMLP(input_size, hidden_sizes, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / total
        acc = 100 * correct / total
        print(f"Epoch {epoch:02d}/{num_epochs} — Loss: {avg_loss:.4f} — Acc: {acc:.2f}%")

    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': dataset.scaler,
        'config': {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'num_classes': num_classes
        }
    }, 'state_action_model.pth')





    
    
    
    

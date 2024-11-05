
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import intake
import numpy as np
import os
import pandas as pd
from common import remove_outliers ### our customized function
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def build_model(input_size, output_size, hidden_layers):
    """
    Build a neural network model based on the configuration.
    
    Parameters:
    - input_size: Size of the input layer.
    - output_size: Size of the output layer.
    - hidden_layers: List of hidden layer sizes.
    
    Returns:
    - model: The neural network model.
    """
    print('--- Input size:', input_size)
    print('--- Output size:', output_size)
    model_layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
    for i in range(len(hidden_layers) - 1):
        model_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        model_layers.append(nn.ReLU())
    model_layers.append(nn.Linear(hidden_layers[-1], output_size))
    model = nn.Sequential(*model_layers)
    return model

def initialize_model(input_size, output_size, hidden_layers, learning_rate):
    """
    Initialize the model, criterion, and optimizer.
    
    Parameters:
    - input_size: Size of the input layer.
    - output_size: Size of the output layer.
    - hidden_layers: List of hidden layer sizes.
    - learning_rate: Learning rate for the optimizer.
    
    Returns:
    - model: The neural network model.
    - criterion: The loss function.
    - optimizer: The optimizer.
    """
    model = build_model(input_size, output_size, hidden_layers)
    criterion = nn.MSELoss()  # Assuming regression task, adjust if classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

    

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        epoch_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')
    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_labels.extend(labels.numpy())
            all_predictions.extend(outputs.numpy())
    
    # Calculate RMSE and R²
    rmse = mean_squared_error(all_labels, all_predictions, squared=False)
    r2 = r2_score(all_labels, all_predictions)
    return rmse, r2

# def evaluate_model(model, x_test_tensor, y_test_tensor):
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(x_test_tensor).numpy()
#         rmse = np.sqrt(mean_squared_error(y_test_tensor, y_pred))
#         r2 = r2_score(y_test_tensor.numpy(), y_pred)

#     print('--- NN rmse:', rmse)
#     print('--- NN r2:', r2)
#     return rmse, r2


RANDOM_STATE = 20

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_STATE)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import intake

def load_and_preprocess_data(catalog_path, random_state=42, test_size=0.20):
    """
    Load and preprocess the data from the given catalog.
    
    Parameters:
    - catalog_path: Path to the intake catalog file.
    - random_state: The seed used by the random number generator.
    - test_size: The proportion of the dataset to include in the test split.
    
    Returns:
    - x_train, x_test, y_train, y_test: The training and testing datasets.
    """
    cat = intake.open_catalog(catalog_path, persist_mode='default')

    ### Saunders2021 data processing
    src = cat["Saunders2021"]
    if not src.is_persisted:
        src.persist()
    dfs = src.read()
    dfs["PARin"] = dfs["solar"] * 0.45 * 4.57  # L26
    dfs = dfs.rename(columns={"VPDleaf": "VPD"})
    dfs = dfs[["PARin", "SWC", "VPD", "Cond", "Species"]]

    ### Anderegg2018 data processing
    src = cat["Anderegg2018"]
    if not src.is_persisted:
        src.persist()
    dfa = src.read()
    N = 1.56
    M = 1 - 1 / N
    ALPHA = 0.036
    dfa["SWC"] = dfa["SWC"].fillna(
        1 / ((1 + (-1 * (dfa["LWPpredawn"]) / ALPHA) ** N) ** M)
    )  # L17
    dfa = dfa[["PARin", "SWC", "VPD", "Cond", "Species"]]

    # Combine datasets and create binary columns from unique values in Species
    df = pd.concat([dfa, dfs])
    df = remove_outliers(df, ["PARin", "VPD", "SWC", "Cond"], verbose=True)
    df = pd.get_dummies(df, columns=["Species"])

    # Split the data into features and target
    x = df.drop(columns="Cond")
    y = df["Cond"]

    # Scale the features
    x_scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test

def create_dataloaders(catalog_path, batch_size, random_state=42, test_size=0.20):
    # Step 1: Load and preprocess the data
    x_train, x_test, y_train, y_test = load_and_preprocess_data(catalog_path, random_state, test_size)
    
    # Step 2: Convert the datasets into PyTorch tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # Step 3: Create TensorDataset objects
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    # Step 4: Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = x_train.shape[1]
    output_size = 1 # assume single output regression
    
    return train_loader, test_loader, input_size, output_size, x_test_tensor, y_test_tensor


if __name__ == "__main__":
    config = load_config('config.yaml')

    hidden_layers = [layer['hidden_size'] for layer in config['model']['layers']]
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    catalog_path = config['data']['catalog_path']

    # Create DataLoaders and get input/output sizes
    train_loader, test_loader, input_size, output_size, x_test_tensor, y_test_tensor = create_dataloaders(catalog_path, batch_size)

    # Initialize model, criterion, and optimizer
    model, criterion, optimizer = initialize_model(input_size, output_size, hidden_layers, learning_rate)

    # Train and evaluate the model
    model = train_model(model, train_loader, criterion, optimizer, num_epochs)
    rmse, r2 = evaluate_model(model, test_loader)

    print(f'RMSE: {rmse:.4f}')
    print(f'R²: {r2:.4f}')


        # print(x_test_tensor.shape)
    # print(y_test_tensor.shape)
    # rmse, r2 = evaluate_model(model, x_test_tensor, y_test_tensor)
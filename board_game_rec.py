########################################################################################
# Board Game Recommender
# Created for: DSC680 Bellview University
# Created by David Hatchett
# Created on: 2026-02-14
#
# discription:
# This script is used to train a board game recommender model.
# It uses a neural network to recommend board games to users based on their ratings of other board games.
# If this code is called normally it will run a training pass of the model.
# if the code is loaded as a module it will not run a training pass. but other 
# functions and classes will be usable.
#
# this is based on the model created here:https://pureai.substack.com/p/recommender-systems-with-pytorch
# and here:https://www.youtube.com/watch?v=cqnrFrF3nJ8
# it use an NCF model created orgionally here:https://arxiv.org/abs/1708.05031
#
# AI was used to help fix errors in some of the functions, however most of the code was written by me.
########################################################################################

import torch
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import ast
import json
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

class BoardGameRecommender(nn.Module):
    '''
    this is the class that defines the neural network model.
    it takes the following parameters:
    - num_users: the number of users in the dataset
    - num_games: the number of games in the dataset
    - num_categories: the number of categories in the dataset
    - num_mechanics: the number of mechanics in the dataset
    - dropout_rate: the dropout rate for the neural network
    - embedding_user_dim: the dimension of the user embedding
    - embedding_game_dim: the dimension of the game embedding
    - embedding_category_dim: the dimension of the category embedding
    - embedding_mechanic_dim: the dimension of the mechanic embedding
    - hidden_dim: the dimension of the hidden layer

    it returns the following:
    - a tensor of the predicted ratings
    '''
    def __init__(
        self,
        num_users,
        num_games,
        num_categories,
        num_mechanics,
        dropout_rate=0.2,
        embedding_user_dim=128,
        embedding_game_dim=32,
        embedding_category_dim=8,
        embedding_mechanic_dim=16,
        hidden_dim=64,
    ):
        super(BoardGameRecommender, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_user_dim).to(self.device)
        self.game_embedding = nn.Embedding(num_games, embedding_game_dim).to(self.device)
        self.category_embedding = nn.EmbeddingBag(num_categories, embedding_category_dim, mode="mean").to(self.device)
        self.mechanic_embedding = nn.EmbeddingBag(num_mechanics, embedding_mechanic_dim, mode="mean").to(self.device)
        self.num_numeric_features = 5
        self.embedding_dim = embedding_user_dim + embedding_game_dim + embedding_category_dim + embedding_mechanic_dim + self.num_numeric_features

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.fc2 = nn.Linear(self.embedding_dim, hidden_dim).to(self.device)
        self.fc3 = nn.Linear(hidden_dim, 1).to(self.device)
        self.relu = nn.ReLU()

    def forward(
        self,
        user_id,
        game_id,
        avg_usr_rating,
        avg_usr_weight,
        bayes_average,
        age,
        game_owners,
        category_indices,
        category_offsets,
        mechanic_indices,
        mechanic_offsets,
    ):
        """
        Forward pass for the BoardGameRecommender model.
        it takes the following parameters:
        - user_id: the id of the user
        - game_id: the id of the game
        - avg_usr_rating: the average user rating of the game
        - avg_usr_weight: the average user weight of the game
        - bayes_average: the bayes average of the game
        - age: the age of the game
        - game_owners: the number of owners of the game
        - category_indices: the indices of the categories of the game
        - category_offsets: the offsets of the categories of the game
        - mechanic_indices: the indices of the mechanics of the game
        - mechanic_offsets: the offsets of the mechanics of the game
        """
        x = torch.cat([
                    self.user_embedding(user_id),
                    self.game_embedding(game_id),
                    self.category_embedding(category_indices, category_offsets),
                    self.mechanic_embedding(mechanic_indices, mechanic_offsets),
                    avg_usr_rating.unsqueeze(1),
                    avg_usr_weight.unsqueeze(1),
                    bayes_average.unsqueeze(1),
                    age.unsqueeze(1),
                    game_owners.unsqueeze(1)], dim=1).to(self.device)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class UserGameDataSet(Dataset):
    '''
    this is the dataset class for the game data and user rating
    we will use this to batch the data out to users.
    It needs to be used along side the collate_fn function to batch the data
    correctly.
    '''
    def __init__(
            self,
            data:pd.DataFrame,
        ):
        self.users_id = data["user_id"]
        self.game_id = data["game_id_encoded"]
        self.user_rating = data["user_rating"]
        self.avg_usr_rating = data["avg_usr_rating_scaled"]
        self.avg_usr_weight = data["avg_usr_weight_scaled"]
        self.bayes_average = data["bayes_average_scaled"]
        self.age = data["age_scaled"]
        self.game_owners = data["game_owners_scaled"]
        self.category_indices = data["category_indices"]
        self.mechanic_indices = data["mechanic_indices"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __len__(self):
        return len(self.users_id)
    
    def __getitem__(self, idx):
        return {
        "users_id": torch.tensor(self.users_id.iloc[idx], dtype=torch.long),
        "game_id": torch.tensor(self.game_id.iloc[idx], dtype=torch.long),
        "user_rating": torch.tensor(self.user_rating.iloc[idx], dtype=torch.float32),
        "avg_usr_rating": torch.tensor(self.avg_usr_rating.iloc[idx], dtype=torch.float32),
        "avg_usr_weight": torch.tensor(self.avg_usr_weight.iloc[idx], dtype=torch.float32),
        "bayes_average": torch.tensor(self.bayes_average.iloc[idx], dtype=torch.float32),
        "age": torch.tensor(self.age.iloc[idx], dtype=torch.long),
        "game_owners": torch.tensor(self.game_owners.iloc[idx], dtype=torch.long),
        "category_indices": torch.tensor(self.category_indices.iloc[idx], dtype=torch.long),
        "mechanic_indices": torch.tensor(self.mechanic_indices.iloc[idx], dtype=torch.long),
        }

def collate_fn(batch):
    '''
    this collate function preers the data for the neural network.
    it creates the prameters for the ebedding bags and the offsets
    '''
    category_indices, category_offsets = get_ebedding_bag(batch,'category_indices')
    mechanic_indices, mechanic_offsets = get_ebedding_bag(batch,'mechanic_indices')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return{
        "users_id": torch.stack([b["users_id"] for b in batch]).to(device),
        "game_id": torch.stack([b["game_id"] for b in batch]).to(device),
        "user_rating": torch.stack([b["user_rating"] for b in batch]).to(device),
        "avg_usr_rating": torch.stack([b["avg_usr_rating"] for b in batch]).to(device),
        "bayes_average": torch.stack([b["bayes_average"] for b in batch]).to(device),
        "game_owners": torch.stack([b["game_owners"] for b in batch]).to(device),
        "avg_usr_weight": torch.stack([b["avg_usr_weight"] for b in batch]).to(device),
        "age": torch.stack([b["age"] for b in batch]).to(device),
        
        "category_indices":category_indices.to(device),
        "category_offsets":category_offsets.to(device),
        "mechanic_indices":mechanic_indices.to(device),
        "mechanic_offsets":mechanic_offsets.to(device)
    }


        
def get_ebedding_bag(batch,field:str) -> torch.Tensor| torch.Tensor:
    """
    creates all the varables for the embedding bags for the model.
    it will take a batch of data and a field and return the indices and offsets.
    """
    indices = []
    offsets = []
    ofset = 0

    for row in batch:
        offsets.append(ofset)
        tokens = row[field].tolist()
        indices.extend(tokens)
        ofset += len(tokens)
    
    return ( 
        torch.tensor(indices, dtype=torch.long)
        , torch.tensor(offsets, dtype=torch.long) 
        )

def json_out(file_name:str, data:dict):
    '''
    this function will save a dictionary to a json file.
    it will take a file name and a dictionary and save the dictionary to the file.
    '''
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f,ensure_ascii=True, indent=4)



def get_vocab(series:pd.Series):
    '''
    Create vocab for a Series of Lists
    use the set to remove duplicates and then create a dictionary of integers
    '''
    out_set = set()

    for record in series:
        for token in record:
            out_set.add(token.strip())

    return {token: i for i, token in enumerate(out_set)}

def encode_text(series:pd.Series):
    '''
    Encode a text series into a dictionary of integers
    '''
    return {iteam: i for i, iteam in enumerate(set(series))}


def create_encoder(file_name:str, data:pd.Series=None, field:str=None) -> dict:
    '''
    this function will create an encoder for a given field.
    it will take a file name and a series if the file already exists
    it will load that file if not it will write it.

    ** add a flag later to overide the file if it exists.

    '''
    if os.path.exists(file_name):
        return json.load(open(file_name, "r", encoding="utf-8"))
    
    if data is not None and field is not None:
        if field == "category" or field == "mechanic":
            vocab = get_vocab(data)
            json.dump(vocab, open(file_name, "w", encoding="utf-8"),ensure_ascii=True)
        else:
            vocab = encode_text(data)
            json.dump(vocab, open(file_name, "w", encoding="utf-8"),ensure_ascii=True)
        return vocab


def process_game_data(file_name:str) -> pd.DataFrame:
    '''
    this function will process the game data from the csv file.
    it will also clean the data and provide back a dataframe used 
    for the model.
    '''
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = df[[
            'id','name','yearpublished','boardgamecategory',
            'boardgamemechanic','average','bayesaverage',
            'owned','averageweight'
        ]].copy()
        
        df.rename(columns={
            'id':'game_id',
            'name':'game_name',
            'yearpublished':'year_published',
            'boardgamecategory':'category',
            'boardgamemechanic':'mechanic',
            'average':'avg_usr_rating',
            'owned':'game_owners',
            'averageweight':'avg_usr_weight',
            'bayesaverage':'bayes_average',
        }, inplace=True) 
        df.dropna(inplace=True)

        ## clean the data
        df['game_id'] = df['game_id'].astype(int)
        df['age'] = df['year_published'].apply(lambda x : 2025 - x if x >= 0 else x * -1)
        df['avg_usr_weight'] = df['avg_usr_weight'].replace(0,np.nan)
        df['avg_usr_weight'] = df['avg_usr_weight'].fillna(df['avg_usr_weight'].mean())
        df['category'] = df['category'].apply(ast.literal_eval)
        df['mechanic'] = df['mechanic'].apply(ast.literal_eval)

        return df

def prep_game_data(game_data:pd.DataFrame, game_encoder:dict, category_encoder:dict, mechanic_encoder:dict) -> pd.DataFrame:
    '''
    does the final process of the game data for the model.
    ** possibly combine with previous fucntion later.
    '''
    game_data = game_data.copy()
    game_data["game_id_encoded"] = game_data["game_id"].map(game_encoder)

    game_data["category_indices"] = game_data["category"].apply(lambda x : [category_encoder[item] for item in x])
    game_data["mechanic_indices"] = game_data["mechanic"].apply(lambda x : [mechanic_encoder[item] for item in x])

    # StandardScaler needs the full column (2D array); fit once, then transform 
    numeric_cols = ['age', 'avg_usr_weight', 'avg_usr_rating', 'bayes_average', 'game_owners']
    for col in numeric_cols:
        scaler = StandardScaler()
        game_data[f'{col}_scaled'] = scaler.fit_transform(game_data[[col]]).ravel()

    return game_data

def get_game_data(config:dict) -> pd.DataFrame:
    '''
    this is like the main fucntion for game data 
    it will call the prevous two function to process the game data.
    it also will have the encoders created if needed.
    it will also check to see if the data model already 
    exists if so loaded it and end the process.
    '''
    game_data_file = config["data"]["games"]
    game_data_model_path = config["data_model"]["game_data_model"]
    game_encoder_path = config["encoders"]["game_name"]
    category_encoder_path = config["encoders"]["category"]
    mechanic_encoder_path = config["encoders"]["mechanic"]

    ## Process game data
    print("Process game data")
    if os.path.exists(game_data_model_path):
        game_data = pd.read_csv(game_data_model_path)      
    else:
        game_data = process_game_data(game_data_file)

    ## Create encoders
    print("Create encoders")
    game_encoder = create_encoder(game_encoder_path, game_data['game_id'],"game_id")
    category_encoder = create_encoder(category_encoder_path, game_data['category'], "category")
    mechanic_encoder = create_encoder(mechanic_encoder_path, game_data['mechanic'], "mechanic")

    ## prep game data for training
    print("Prep game data for training")
    if not os.path.exists(game_data_model_path):
        game_data = prep_game_data(game_data, game_encoder, category_encoder, mechanic_encoder)
        game_data.to_csv(game_data_model_path, index=False)

    return game_data


def process_user_data(user_data_file:str) -> pd.DataFrame:
    '''
    loads the user data from csv and cleans it.
    '''
    if os.path.exists(user_data_file):
        user_data = pd.read_csv(user_data_file, usecols=['ID','user','rating'])
        user_data.rename(columns={'ID':'game_id','rating':'user_rating'}, inplace=True)
        user_data['game_id'] = user_data['game_id'].astype(int)
    return user_data

def prep_user_data(user_data:pd.DataFrame, user_encoder:dict) -> pd.DataFrame:
    '''
    does the final process of the user data for the model.
    it will map the user ids to the encoder and return the dataframe.
    *** most likely should combine with previous function later.
    '''
    user_data = user_data.copy()
    user_data["user_id"] = user_data["user"].map(user_encoder)
    return user_data


def get_user_data(config:dict) -> pd.DataFrame|dict:
    '''
    this is like the main fucntion for user data 
    it will call the prevous two function to process the user data.
    it also will have the encoders created if needed.
    it will also check to see if the data model already 
    exists if so loaded it and end the process.
    '''
    user_data_file = config["data"]["users"]
    user_data_model_path = config["data_model"]["user_data_model"]
    user_encoder_path = config["encoders"]["user_id"]


    print("Get User Data")
    if not os.path.exists(user_data_model_path):
        user_data = process_user_data(user_data_file)
    else:
        user_data = pd.read_csv(user_data_model_path)
    
    user_encoder = create_encoder(user_encoder_path, user_data['user'].unique(),"user")

    ## prep user data for training
    print("Prep user data for training")
    if not os.path.exists(user_data_model_path):
        user_data = prep_user_data(user_data, user_encoder)
        user_data.to_csv(user_data_model_path, index=False)
        user_data.dropna(inplace=True)

    return user_data

def setup_config(config_file:str) -> dict:
    '''
    loads the config file and returns the config dictionary.
    '''
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def create_train_data(game_data:pd.DataFrame, user_data:pd.DataFrame, config:dict) -> pd.DataFrame|pd.DataFrame|pd.DataFrame:
    '''
    this function will create the train data for the model.
    it will merge the game and user data and return the dataframe.
    this also creates the train, validation, and test sets.
    '''
    ## merge game and user data
    print("Merge game and user data")
    game_data_model = pd.merge(user_data, game_data, left_on='game_id', right_on='game_id', how='inner')
    game_data_model = game_data_model[['user_id',
                'game_id_encoded',
                'user_rating',
                'avg_usr_rating_scaled',
                'avg_usr_weight_scaled',
                'bayes_average_scaled',
                'age_scaled',
                'game_owners_scaled',
                'category_indices',
                'mechanic_indices']]
    game_data_model.dropna(inplace=True)
    print(game_data_model.shape)

    ## fix the list fields to act correctly. ## future move to the prep_game_data function.
    game_data_model['category_indices'] = game_data_model['category_indices'].apply(ast.literal_eval)
    game_data_model['mechanic_indices'] = game_data_model['mechanic_indices'].apply(ast.literal_eval)

    ## split into train, validation, and test sets 
    print("Split into train, validation, and test sets")
    train_data, test_data = model_selection.train_test_split(game_data_model, test_size=0.2, random_state=42)
    train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.2, random_state=42)

    ## save train, validation, and test sets
    print("Save train, validation, and test sets")
    train_data.to_csv(config["data_model"]["train_data_path"], index=False)
    validation_data.to_csv(config["data_model"]["validation_data_path"], index=False)
    test_data.to_csv(config["data_model"]["test_data_path"], index=False)

    return train_data, validation_data, test_data

def get_data_loaders(train_data:pd.DataFrame,
                     validation_data:pd.DataFrame,
                     test_data:pd.DataFrame,
                     batch_size:int=100,
                     num_workers:int=0) -> DataLoader|DataLoader|DataLoader:
    '''
    This function will create the data laoders for training, validation, and test sets.
    '''
    ## create dataset
    print("Create dataset")
    train_dataset = UserGameDataSet(train_data)
    validation_dataset = UserGameDataSet(validation_data)
    test_dataset = UserGameDataSet(test_data)

    ## create data loaders
    print("Create data loaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, validation_loader, test_loader

def get_encoders(config:dict) -> dict|dict|dict|dict:
    '''
    this function will load the encoders from the disk and return them.
    '''
    with open(config["encoders"]["user_id"], "r", encoding="utf-8") as f:
        user_encoder = json.load(f)
    with open(config["encoders"]["game_name"], "r", encoding="utf-8") as f:
        game_encoder = json.load(f)
    with open(config["encoders"]["category"], "r", encoding="utf-8") as f:
        category_encoder = json.load(f)
    with open(config["encoders"]["mechanic"], "r", encoding="utf-8") as f:
        mechanic_encoder = json.load(f)
    return user_encoder, game_encoder, category_encoder, mechanic_encoder

def log_progress(epoch,EPOCHS, step_count, train_loss, train_precision, train_recall, data_size):
    '''
    this function will log the progress of the training.
    it will take the epoch, the total epochs, the step count, the train loss, the train precision, the train recall, and the data size.
    it will write the progress to the console.
    '''
    avg_loss = sum(train_loss) / len(train_loss)
    avg_precision = sum(train_precision) / len(train_precision)
    avg_recall = sum(train_recall) / len(train_recall)
    sys.stderr.write(
        f"\r{epoch+1:02d}/{EPOCHS:02d} | Step: {step_count}/{data_size} | Avg Loss: {avg_loss:<6.9f} | Avg Precision: {avg_precision:<6.9f} | Avg Recall: {avg_recall:<6.9f}"
    )
    sys.stderr.flush()





def train_model(model:nn.Module, 
                train_loader:DataLoader, 
                validation_loader:DataLoader,
                config:dict, 
                epochs:int=10, 
                learning_rate:float=0.001,
                weight_decay:float=0.0001,
                threshold:float=7.0,
                ) -> nn.model|dict:
    '''
    this function will train the model.
    it will take the model, the train loader, the validation loader, the config, the epochs, the learning rate, the weight decay, the threshold, and return the model and the history.
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    log_progress_step = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_precision": [],
        "train_recall": [],
        "validation_precision": [],
        "validation_recall": []
    }

    print(f'Training model for {epochs} epochs')
    for epoch in range(epochs):
        model.train()
        train_loss = []
        validation_loss = []
        train_precision = []
        train_recall = []
        validation_precision = []
        validation_recall = []
        step_count = 0
        data_size = len(train_loader)
        

        for i, batch in enumerate(train_loader):
            x = model(
                user_id = batch["users_id"],
                game_id = batch["game_id"],
                avg_usr_rating = batch["avg_usr_rating"],
                avg_usr_weight = batch["avg_usr_weight"],
                bayes_average = batch["bayes_average"],
                age = batch["age"],
                game_owners = batch["game_owners"],
                category_indices = batch["category_indices"],
                category_offsets = batch["category_offsets"],
                mechanic_indices = batch["mechanic_indices"],
                mechanic_offsets = batch["mechanic_offsets"],
            ).to(device)
            x = x.squeeze()

            loss = criterion(x, (batch["user_rating"].to(torch.float32)))
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_true = batch["user_rating"].detach().cpu().numpy()
            y_pred = x.detach().cpu().numpy()
            
            # Apply threshold: treat >= threshold as positive (1), else negative (0)
            y_true_binary = (y_true >= threshold).astype(np.int64)   
            y_pred_binary = (y_pred >= threshold).astype(np.int64)


            train_precision.append(precision_score(y_true_binary, y_pred_binary, zero_division=0))
            train_recall.append(recall_score(y_true_binary, y_pred_binary, zero_division=0))

            if step_count % log_progress_step == 0:
                log_progress(epoch,epochs, step_count, train_loss, train_precision, train_recall, data_size)
            step_count += 1

        data_size = len(validation_loader)
        step_count = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(validation_loader):
                x = model(
                    user_id = batch["users_id"],
                    game_id = batch["game_id"],
                    avg_usr_rating = batch["avg_usr_rating"],
                    avg_usr_weight = batch["avg_usr_weight"],
                    bayes_average = batch["bayes_average"],
                    age = batch["age"],
                    game_owners = batch["game_owners"],
                    category_indices = batch["category_indices"],
                    category_offsets = batch["category_offsets"],
                    mechanic_indices = batch["mechanic_indices"],
                    mechanic_offsets = batch["mechanic_offsets"],
                ).to(device)   
                x = x.squeeze()
                loss = criterion(x, (batch["user_rating"].to(torch.float32)))
                validation_loss.append(loss.item())

                y_true = batch["user_rating"].detach().cpu().numpy()
                y_pred = x.detach().cpu().numpy()
                
                y_true_binary = (y_true >= threshold).astype(np.int64)   
                y_pred_binary = (y_pred >= threshold).astype(np.int64)

                validation_precision.append(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                validation_recall.append(recall_score(y_true_binary, y_pred_binary, zero_division=0))

                if step_count % log_progress_step == 0:
                    log_progress(epoch,epochs, step_count, validation_loss, validation_precision, validation_recall, data_size)
                step_count += 1

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {sum(train_loss)/len(train_loss)}, Validation Loss: {sum(validation_loss)/len(validation_loss)}")
        history["train_loss"].append(sum(train_loss)/len(train_loss))
        history["validation_loss"].append(sum(validation_loss)/len(validation_loss))
        history["train_precision"].append(sum(train_precision)/len(train_precision))
        history["train_recall"].append(sum(train_recall)/len(train_recall))
        history["validation_precision"].append(sum(validation_precision)/len(validation_precision))
        history["validation_recall"].append(sum(validation_recall)/len(validation_recall))
        torch.save(model.state_dict(), config["models"]["recommender"].format(epoch+1))

    return model, history

def main():

    ## Setup config
    print("Setup config")
    config = setup_config("config/config.json")

    ## check if train, validation, and test sets exist
    print("Check if train, validation, and test sets exist")
    if os.path.exists(config["data_model"]["train_data_path"]) and os.path.exists(config["data_model"]["validation_data_path"]) and os.path.exists(config["data_model"]["test_data_path"]):
        print("Train, validation, and test sets already exist")
        train_data = pd.read_csv(config["data_model"]["train_data_path"])
        validation_data = pd.read_csv(config["data_model"]["validation_data_path"])
        test_data = pd.read_csv(config["data_model"]["test_data_path"])

        train_data['category_indices'] = train_data['category_indices'].apply(ast.literal_eval)
        train_data['mechanic_indices'] = train_data['mechanic_indices'].apply(ast.literal_eval)
        validation_data['category_indices'] = validation_data['category_indices'].apply(ast.literal_eval)
        validation_data['mechanic_indices'] = validation_data['mechanic_indices'].apply(ast.literal_eval)
        test_data['category_indices'] = test_data['category_indices'].apply(ast.literal_eval)
        test_data['mechanic_indices'] = test_data['mechanic_indices'].apply(ast.literal_eval)

    else:
        ## if no train, validation, and test sets exist, create them
        
        ## Get Game Data
        print("Get Game Data")
        game_data = get_game_data(config)

        ## get User Data
        print("Get User Data")
        user_data = get_user_data(config)

        ## Create Train Data
        print("Create Train Data")
        train_data, validation_data, test_data = create_train_data(game_data, user_data, config)
        ## remove not needed dataframes
        del (game_data, user_data)


    ## Get Data Loaders
    print("Get Data Loaders")
    train_loader, validation_loader, test_loader = get_data_loaders(train_data, validation_data, test_data, batch_size=1000) 

    ## remove not needed dataframes
    del (train_data, validation_data, test_data)

    ## Get Encoders
    print("Get Encoders")
    user_encoder, game_encoder, category_encoder, mechanic_encoder = get_encoders(config)

    ## instantiate model
    print("Instantiate model")


    model = BoardGameRecommender(num_users=len(user_encoder), 
                                num_games=len(game_encoder), 
                                num_categories=len(category_encoder), 
                                num_mechanics=len(mechanic_encoder),
                                dropout_rate=0.2,
                                embedding_user_dim=128,
                                embedding_game_dim=32,
                                embedding_category_dim=8,
                                embedding_mechanic_dim=16,
                                hidden_dim=64
                                )

    ## train model
    print("Train model")
    model, history = train_model(
        model=model, 
        train_loader=train_loader, 
        validation_loader=validation_loader, 
        config=config, 
        epochs=10, 
        learning_rate=0.001, 
        weight_decay=0.0001, 
        threshold=7.0)

    ## save history
    print("Save history")
    with open(config["models"]["history"], "w", encoding="utf-8") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
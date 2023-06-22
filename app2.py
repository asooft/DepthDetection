import streamlit as st
import os
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm
from itertools import product
from IPython.display import display, clear_output
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler


st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='Recommender Movies Dashboard')

        
st.title('Recommender Movies Dashboard')


st.cache_data()
ratings=pd.read_csv("Dataset/ratings.csv")
movies=pd.read_csv("Dataset/movies.csv")
df = pd.merge(ratings, movies, on='movieId', how='left')
first_100_users = df['userId'].unique()[:100]
df = df[df['userId'].isin(first_100_users)]
genres_unique = pd.DataFrame(df.genres.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genre']) # Format into DataFrame to store later
df = df.join(df.genres.str.get_dummies().astype(int))
df.drop('genres', inplace=True, axis=1)
genres_unique = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical', 'Documentary', 'IMAX', 'Western', 'Film-Noir', '(no genres listed)']
cats_ohe = pd.get_dummies(df[genres_unique])
null_counts = ratings.isnull().sum()
null_percentages = (null_counts / len(ratings)) * 100
null_summary = pd.DataFrame({'Null Count': null_counts, 'Null Percentage': null_percentages})
null_counts = movies.isnull().sum()
null_percentages = (null_counts / len(ratings)) * 100
null_summary = pd.DataFrame({'Null Count': null_counts, 'Null Percentage': null_percentages})
unique_items = ratings['movieId'].nunique()
unique_users = ratings['userId'].nunique()
unique_items = movies['movieId'].nunique()
user_mappings = df['userId'].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'new_id'}).set_index('userId')

# Assuming you have the user_mappings DataFrame available

# Get the unique values from the userId column
user_ids = user_mappings.index.unique()

# Create a dictionary to map userId to new_id
mapping_dict = user_mappings['new_id'].to_dict()

# Create the dropdown menu
selected_user_id = st.selectbox('Select a userId:', user_ids)

# Get the corresponding new_id using the mapping_dict
selected_new_id = mapping_dict[selected_user_id]

# Create a DataFrame with the selected mapping
mapping_df = pd.DataFrame({'userId': [selected_user_id], 'new_id': [selected_new_id]})

# Display the mapping DataFrame as a table without the index column
st.write(mapping_df)

# Display a name for the input field and get the numeric input
valueS = st.number_input("Enter a Top Number of movies you want", value=5, step=1, format="%d")

# Convert the input value to an integer
integer_value = int(valueS)

# Use the entered integer
st.write("Top Number of movies you want:", integer_value)

#integer_value = 5
movie_mappings = df['movieId'].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'new_id'}).set_index('movieId')

df_copy = df.copy() # To avoid changing the original DataFrame
df_copy['ones'] = 1

columns_to_replace = ["userId", "movieId"]  # Specify the columns you want to replace

df_mapped = df_copy.copy()  # Create a copy of the original dataframe

# Replace values in the specified columns
df_mapped[columns_to_replace] = df_mapped[columns_to_replace].replace({
    "userId": user_mappings.new_id.to_dict(),
    "movieId": movie_mappings.new_id.to_dict()
})

# Replace None with the appropriate values
agg_history = pd.pivot_table(df_mapped,
               values='ones', index='userId', columns='movieId', fill_value=0)


# Replace None with correct values
# We need to normalize the aggregated history by dividing each value by the sum of all the values in the same row
agg_history_norm = agg_history / agg_history.values.sum(axis=1, keepdims=True)


df['Train'] = (ratings.groupby("userId").cumcount(ascending=False) != 0).replace({True:1, False:0})


final = pd.concat([df_mapped[['userId','movieId']], cats_ohe,df[['Train','rating']]], axis=1)

train = final[final.Train == 1]
test = final[final.Train == 0]


class DPMovieDataset(Dataset):
  def __init__(self, user_ids, data, agg_hist, active_matrix, recommendation=False):
    self.user_ids = user_ids
    self.data = data
    self.agg_hist = agg_hist
    self.active_matrix = active_matrix
    self.recommendation = recommendation

  def __len__(self):
    return self.user_ids.shape[0]

  def __getitem__(self, idx):
    batch_data = self.data[self.data['userId'].isin(idx)] # Select the rows corresponding to the list of user indices `idx` from self.data dataframe
    cat_cols = batch_data[['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical', 'Documentary', 'IMAX', 'Western', 'Film-Noir', '(no genres listed)']] # From batch_data extract only the one-hot encoded categorical columns
    agg_history = batch_data[['userId']].merge(self.agg_hist, left_on='userId', right_index=True) # Get the aggregated history for each selected transaction using merge
    active_groups = self.active_matrix[self.active_matrix.index.isin(batch_data.index)] # Select the rows corresponding to the indices of the transactions selected in batch_data

    features = torch.from_numpy(np.hstack((active_groups.values, agg_history.values, cat_cols.values))) # Concatenate the processed columns together horizontally

    if not self.recommendation:
      targets = batch_data['rating']
      return features, targets
    else:
      return features
      
active_columns = pd.get_dummies(final[['userId','movieId']].astype(str))
dataset_train = DPMovieDataset(user_mappings.values, train, agg_history_norm, active_columns)
dataset_test = DPMovieDataset(user_mappings.values, test, agg_history_norm, active_columns)

dataloader_train = DataLoader(dataset_train,
                              sampler=BatchSampler(SequentialSampler(dataset_train), batch_size=10, drop_last=False),
                              batch_size=None)

dataloader_test = DataLoader(dataset_test,
                              sampler=BatchSampler(SequentialSampler(dataset_test), batch_size=10, drop_last=False),
                              batch_size=None)
                              
class FactorizationMachine(torch.nn.Module):
  def __init__(self, n, k, bias=False):
    super(FactorizationMachine, self).__init__()
    self.n = n
    self.k = k
    self.linear = torch.nn.Linear(self.n, 1, bias)
    self.V = torch.nn.Parameter(torch.randn(n,k)) # Creating the latent matrix V of size (n X k) and initializing it with random values

  def forward(self, x_batch):
    x_batch = x_batch.float()
    part_1 = torch.matmul(x_batch, self.V).pow(2).sum(1, keepdim=True)  # perform the first part of the interaction term: row-wise-sum((XV)^2)
    part_2 = torch.matmul(x_batch.pow(2), self.V.pow(2)).sum(1, keepdim=True) # perform the second part of the interaction term: row-wise-sum((X)^2 * (V)^2))
    inter_term = (part_1 - part_2)/2 # Put the interaction term parts together (refer to the equations above)
    var_strength = self.linear(x_batch) # Perform the linear part of the model equation (refer to the demo notebook on how to use layers in pytorch models)
    return var_strength + inter_term
    
features,ratings=dataset_train[[1]]


model = FactorizationMachine(n=11413, k=integer_value)

def model_step(mode, x, y=None, optimizer=None, train=True):
  if train: # If we're in training phase, then zero the gradients and make sure the model is set to train
    model.train()
    optimizer.zero_grad()
  else: # If we're in evaluation phase, then make sure the model is set to eval
    model.eval()

  with torch.set_grad_enabled(train): # Either to perform the next lines with gradient tracing or not
    pred = model(x) # Get the model output from x
    pred = pred.reshape(pred.shape[0], ) # Flatten the prediction values

    y = torch.from_numpy(y.values.reshape(y.shape[0], )).float()

    criterion = torch.nn.MSELoss() # Define the criterion as MSELoss from torch
    loss = criterion(pred, y)

    if train:
      loss.backward()
      optimizer.step()

  return loss
  
def train_loop(model, train_loader, eval_loader, lr, w_decay, epochs, eval_step):
  step = 0
  """ Defining our optimizer """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
  epochs_l, steps, t_losses, v_losses = [], [], [], []

  epochs_tqdm = tqdm(range(epochs), desc='Training in Progress', leave=True)
  for epoch in epochs_tqdm:
    for x, y in train_loader:
      loss_batch = model_step(model, x, y, optimizer, train=True)
      step +=1
      if step % eval_step == 0:
        train_loss = loss_batch
        val_loss = 0
        for x, y in eval_loader:
          val_loss += model_step(model, x, y, train=False)
        epochs_l.append(epoch+1)
        steps.append(step)
        t_losses.append(train_loss.detach().numpy())
        v_losses.append(val_loss.detach().numpy())
        clear_output(wait=True)
        display(pd.DataFrame({'Epoch': epochs_l, 'Step': steps, 'Training Loss': t_losses, 'Validation Loss': v_losses}))
        



import pickle
file_path = "fm_model.pkl"  # Change the path as per your preference
with open(file_path, 'rb') as f:
    model = pickle.load(f)
selected_new_id = 2    
 # Save selected_new_id in a variable
new_id_variable = selected_new_id
    
new=user_mappings.loc[new_id_variable].new_id
features,ratings = dataset_test[[new]]


model.eval()
with torch.no_grad():
  st.write(f'Predicted rating for User of interest: {model(features).item()}') # Get the model output on the user of interest after running the previous cell to now their new_id
  st.write(f'Actual Rating: {ratings.values[0]}') # Extract the actual rating for the user of interest from dataset_test Dataset object

# Replace None with the new_id of the user
items_our_user_rated = (train[train.userId==new].movieId).unique().tolist()
items_our_user_rated.extend((test[test.userId==new].movieId).unique().tolist())

items_our_user_can_rate = movie_mappings[~movie_mappings.new_id.isin(items_our_user_rated)].new_id.tolist()

st.write(f'Number of unique items user of interest rated is {len(items_our_user_rated)}')
st.write(f'Number of unique items that can be recommended to user of interest is {len(items_our_user_can_rate)}')
st.write(f'Preview of the item list:\n\t{items_our_user_can_rate[:integer_value]}')

N = integer_value  # Number of recommendations

recommendations = []

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for item_id in items_our_user_can_rate:
        features = dataset_test[[item_id]][0]  # Create a dataset for the item
        #print(features)
        # Check if the dataset for the item is empty
        if features.nelement() == 0:
            continue

        predicted_rating = model(features).item()  # Get the predicted rating for the user

        recommendations.append((item_id, predicted_rating))

# Sort the recommendations based on predicted ratings in descending order
recommendations.sort(key=lambda x: x[1], reverse=True)

# Select the top N recommendations
top_recommendations = recommendations[:N]

# Print the top recommendations
st.write("Top Recommendations:")
for item_id, predicted_rating in top_recommendations:
    movie_title = movie_mappings[movie_mappings.new_id == item_id].index[0]
    st.write(f"Movie: {movie_title}, Predicted Rating: {predicted_rating}")
    
    # Print the top recommendations with movie names
print("Top Recommendations:")
for item_id, predicted_rating in top_recommendations:
    movie_title = movies[movies['movieId'] == item_id]['title'].values[0]
    st.write(f"Movie: {movie_title}, Predicted Rating: {predicted_rating}")


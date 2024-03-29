import numpy as np
import pandas as pd
from functions import *
import os

# READ DATA FROM FILE
# ------------------------------------------------------------------------
read_data = True
def get_training_txt_files(folder_path):
  """
  Gets a list of all txt files starting with "Training" in a folder.
  Args:
      folder_path: The path to the folder containing the files.
  Returns:
      A list of filenames (strings) that start with "Training" and have the .txt extension.
  """
  training_txt_files = []
  for filename in os.listdir(folder_path):
    if filename.startswith("Training") and filename.endswith(".txt"):
      training_txt_files.append(filename)
  return training_txt_files
def get_results_txt_files(folder_path):
  results_txt_files = []
  for filename in os.listdir(folder_path):
    if filename.startswith("Results") and filename.endswith(".txt"):
      results_txt_files.append(filename)
  return results_txt_files
def calculate_accumulated_average(arr):
  # Initialize arrays for accumulated average reward and iteration numbers
  accumulated_average= np.zeros_like(arr)
  # Calculate accumulated average reward for each iteration
  for i in range(len(arr)):
    accumulated_average[i] = np.mean(arr[:i + 1])
  return accumulated_average
def calculate_accumulated(arr):
  # Initialize arrays for accumulated average reward and iteration numbers
  accumulated= np.zeros_like(arr)
  # Calculate accumulated average reward for each iteration
  for i in range(len(arr)):
    accumulated[i] = np.sum(arr[:i + 1])
  return accumulated
# if read_data:
folder_path = "./"
# folder_path = r'C:\Users\salin\Documents\Doctorado\Glass System\18\Triangle4_C\7-5S3A-normstate-step10d\\'
# folder_path = folder_path.replace('\\','/')

# # To POSTPROCESS Results Files
# results_txt_files = get_results_txt_files(folder_path)
# for file in results_txt_files:
#     save_data2csv(f'{folder_path}{file}')

training_txt_files = get_training_txt_files(folder_path)
accumulated_reward = []
accumulated_average_reward = []
iteration = []
for f,file_name in enumerate(training_txt_files):
    data = read_data_from_file(f'{folder_path}{file_name}')
    df = pd.DataFrame(data)
    time = np.array(df['ran time'].tolist())
    size = np.array(df['workpiece size'].tolist())
    episode = np.array(df['episode'].tolist())
    indexes = np.array(df['index'].tolist())
    action = np.array(df['action'].tolist())
    reward = np.array(df['reward'].tolist())
    states = np.array(df['state'].tolist())
    angles = np.array(states[:,1])
    collision = np.array(states[:,0])
    TCP = np.array(df['TCP'].tolist())

    iteration.append(np.arange(len(reward)))
    accumulated_average_reward.append(calculate_accumulated_average(reward))
    accumulated_reward.append(calculate_accumulated(reward))
# iteration = np.mean(np.array(iteration),axis=0)
# accumulated_average_reward = np.mean(accumulated_average_reward,axis=0)
# accumulated_reward = np.mean(accumulated_reward,axis=0)
    # for i in range(len(iteration)):
size = len(iteration)
key_names = [f'iteration {i}' for i in range(size)]
df1 = pd.DataFrame(iteration).transpose()
df1.columns = key_names
size = len(accumulated_reward)
key_names = [f'accumulated_reward {i}' for i in range(size)]
df2 = pd.DataFrame(accumulated_reward).transpose()
df2.columns = key_names
frames = [df1,df2]
df = pd.concat(frames,axis=1)
for d,_ in enumerate(df):
    df = df.replace(np.nan, df.values[-1,d]).ffill()
df.to_csv(f'{folder_path}postprocessed_{file_name}.csv')
# df = pd.merge(df1, df2)
# postdata = {
# "iteration": iteration.tolist(),
# "Av. accumulated reward": accumulated_average_reward.tolist(),
# "Accumulated reward": accumulated_reward.tolist()
#         }
# save_data2txt(f'postprocessed_{file_name}',postdata)
# save_data2csv(f'postprocessed_{file_name}')
import numpy as np
import pandas as pd
from functions import *
import os

# READ DATA FROM FILE
# ------------------------------------------------------------------------
read_data = True
# Version 1: Tabulate iteration/Acc. Reward
# Version 2: Tabulate Episode/Acc. Reward
# Version 3: Tabulate Episode/Steps
# Version 4: Tabulate Episode/Steps separated in columns to make it easy to make boxplots
post_process_version = 4
# if read_data:
# folder_path = "./"
folder_path = r'C:\Users\salin\Documents\Doctorado\Glass System\20\Training different geometries\test\2\spiral\\'
# folder_path = r'C:\Users\salin\Documents\Doctorado\Glass System\19\Reloading Model test\Retrained Model\\'
# folder_path = r'C:\Users\salin\Documents\Doctorado\Glass System\19\Different methods solutios\control\\'
# folder_path = folder_path.replace('\\','/')
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

# # To POSTPROCESS Results Files
# results_txt_files = get_results_txt_files(folder_path)
# for file in results_txt_files:
#     save_data2csv(f'{folder_path}{file}')

training_txt_files = get_training_txt_files(folder_path)
if post_process_version == 1:
    accumulated_reward = []
    accumulated_average_reward = []
    iteration = []
    episodes = []
if post_process_version == 2:
    accumulated_reward = []
    accumulated_reward2 = []
    episodes = []
    episodes2 = []
if post_process_version == 3:
    steps = []
    episodes = []
if post_process_version == 4:
    steps = np.empty((0,1))
    episodes = np.empty((0,1))
for f,file_name in enumerate(training_txt_files):
    data = read_data_from_file(f'{folder_path}{file_name}')
    df = pd.DataFrame(data)
    time = np.array(df['ran time'].tolist())
    size = np.array(df['workpiece size'].tolist())
    episode = np.array(df['episode'].tolist())
    step = np.array(df['step'].tolist())
    indexes = np.array(df['Segment'].tolist()) #Antes se llamaba index
    action = np.array(df['action'].tolist())
    reward = np.array(df['reward'].tolist())
    states = np.array(df['state'].tolist())
    angles = np.array(states[:,1])
    collision = np.array(states[:,0])
    TCP = np.array(df['TCP'].tolist())

    if post_process_version == 1:
        iteration.append(np.arange(len(reward)))
        accumulated_average_reward.append(calculate_accumulated_average(reward))
        accumulated_reward.append(calculate_accumulated(reward))
    if post_process_version == 2:
        leps,lep = [len(episodes),len(episode)]
        episodes[leps:leps + lep] = episode
        episodes2.append(episode)
        # episodes = np.array(episodes)
        c_accr = calculate_accumulated(reward)
        laccr,lcaccr = [len(accumulated_reward),len(c_accr)]
        accumulated_reward[laccr:laccr + lcaccr] = c_accr
        accumulated_reward2.append(calculate_accumulated(reward))
    if post_process_version == 3:
        leps, lep = [len(episodes), len(episode)]
        episodes[leps:leps + lep] = episode
        # episodes.append(episode)
        # episodes = np.array(episodes)
        # c_accr = calculate_accumulated(reward)
        # laccr, lcaccr = [len(accumulated_reward), len(c_accr)]
        # accumulated_reward[laccr:laccr + lcaccr] = c_accr
        lsps, lsp = [len(steps), len(step)]
        steps[lsps:lsps + lsp] = step
    if post_process_version == 4:
        # leps, lep = [len(episodes), len(episode)]
        # episodes[leps:leps + lep] = episode
        episodes = np.vstack([episodes,episode.reshape(len(episode),1)])
        # episodes.append(episode)
        # episodes = np.array(episodes)
        # c_accr = calculate_accumulated(reward)
        # laccr, lcaccr = [len(accumulated_reward), len(c_accr)]
        # accumulated_reward[laccr:laccr + lcaccr] = c_accr
        # lsps, lsp = [len(steps), len(step)]
        # steps[lsps:lsps + lsp] = step
        steps = np.vstack([steps, step.reshape(len(step), 1)])
        # steps.append(step)

        # accumulated_reward = np.array(accumulated_reward)
        # episodes.append(episode)
        # accumulated_reward.append(calculate_accumulated(reward))

# iteration = np.mean(np.array(iteration),axis=0)
# accumulated_average_reward = np.mean(accumulated_average_reward,axis=0)
# accumulated_reward = np.mean(accumulated_reward,axis=0)
    # for i in range(len(iteration)):
if post_process_version == 1:
    size = len(iteration)
    key_names = [f'iteration {i}' for i in range(size)]
    df1 = pd.DataFrame(iteration).transpose()
    df1.columns = key_names
    size = len(accumulated_reward)
    key_names = [f'accumulated_reward {i}' for i in range(size)]
    df2 = pd.DataFrame(accumulated_reward).transpose()
    df2.columns = key_names
    # size = len(segment)
    # key_names = [f'Segment average']
    # df3 = pd.DataFrame(segment).transpose()
    # df3 = df3.ffill()
    # df3 = df3.mean(axis=1).to_frame(name='Segment average')
    # df3.columns = key_names
    frames = [df1,df2]
    df = pd.concat(frames, axis=1)
if post_process_version == 2:
    size = len(episodes2)
    key_names = [f'Episodes2 {i}' for i in range(size)]
    df3 = pd.DataFrame(episodes2).transpose()
    df3.columns = key_names
    size = len(accumulated_reward2)
    key_names = [f'accumulated_reward2 {i}' for i in range(size)]
    df4 = pd.DataFrame(accumulated_reward2).transpose()
    df4.columns = key_names

    accumulated_reward = np.array(accumulated_reward)
    episodes = np.array(episodes)
    size = len(episodes)
    key_names = [f'Episode {i}' for i in range(0,max(episode+1))]
    df = pd.DataFrame(episodes)  # .transpose()
    df1 = pd.DataFrame(columns=key_names)
    indexes = pd.DataFrame(columns=key_names)
    max_len = 0
    for e, key_name in enumerate(key_names):
        filtered_data = df[df[0] == e] # Filter by episode number
        max_len = len(filtered_data) if len(filtered_data) > max_len else max_len
        df1 = df1.reindex(range(0, max_len), fill_value=np.nan)
        df1[key_name].values[0:len(filtered_data)] = filtered_data.values[:,:].reshape(-1, )#.drop(0, axis=1)#.iloc[:,0]#.drop(0, axis=1)#.drop(0, axis=1)  # Add filtered data (excluding episode number column)
        indexes = indexes.reindex(range(0, max_len), fill_value=np.nan)
        indexes[key_name].values[0:len(filtered_data)] = filtered_data.index
    # df1 = df[df[0] == 0].to_frame(name=key_names[0])
    # for i in range(1,max(episode)):
    #     print(i)
    #     df1[key_names[i]] = df[df[0] == i]
        # df1 = pd.concat(df, axis=1)
    # df[df[0]==1]
    # df1 = pd.concat(df,axis=1)
    # df1.columns = key_names
    size = len(accumulated_reward)
    key_names = [f'Accumulated Reward {i}' for i in range(0,max(episode+1))]
    df = pd.DataFrame(accumulated_reward)  # .transpose()
    df2 = pd.DataFrame(columns=key_names)
    max_len = 0
    for e, key_name in enumerate(key_names):
        index = list(indexes.values[:,e])  # Filter by episode number
        index = np.array(index)[np.argwhere(~np.isnan(index))]
        max_len = len(index) if len(index) > max_len else max_len
        df2 = df2.reindex(range(0, max_len), fill_value=np.nan)
        if e > 0:
            # Reset accumulated Value for each episode
            arr = np.array(accumulated_reward)[index.astype(int)].reshape(-1, )
            df2[key_name].values[0:len(index)] = arr#-np.full_like(arr,)
        else:
            df2[key_name].values[0:len(index)] = np.array(accumulated_reward)[index.astype(int)].reshape(-1, )
    # df2 = pd.DataFrame(accumulated_reward)#.transpose()
    # df2.columns = key_names
    frames = [df1, df2]
    df = pd.concat(frames, axis=1)
if post_process_version == 3:
    size = len(episodes)
    key_names = [f'Episodes']
    df1 = pd.DataFrame(np.array(episodes).reshape(1,len(episodes))).transpose()
    df1.columns = key_names
    size = len(steps)
    key_names = [f'Steps']
    df2 = pd.DataFrame(np.array(steps).reshape(1,len(steps))).transpose()
    df2.columns = key_names
    frames = [df1, df2]
    df = pd.concat(frames, axis=1)
if post_process_version == 4:
    max_episodes = int(max(episodes)[0])
    # episode = np.empty((0,max_episodes))
    step = np.empty((0, ))
    df = pd.DataFrame(step)
    key_names = [0]
    for i in range(0,max_episodes):
        index,arr = np.where(episodes == i)
        # episode = np.hstack([episode,arr])
        df1 = pd.DataFrame(steps[index])
        if (df1 != 1).any()[0]:
            df = pd.concat([df,df1], axis=1)
            key_names.append(i+1)
        # step = np.hstack([step,steps[index].transpose()])
    # key_names = [f'{i}' for i in range(0, max_episodes+1)]
    # df = pd.DataFrame(columns=key_names)
    # df1 = pd.DataFrame(step).transpose()
    df.columns = key_names
    # size = len(episodes)
    # key_names = [f'Episodes']
    # df1 = pd.DataFrame(np.array(episodes).reshape(1,len(episodes))).transpose()
    # df1.columns = key_names
    # size = len(steps)
    # key_names = [f'Steps']
    # df2 = pd.DataFrame(np.array(steps).reshape(1,len(steps))).transpose()
    # df2.columns = key_names
    # frames = [df1]
# df = pd.concat(frames,axis=1)
# for d,_ in enumerate(df):
    # df = df.replace(np.nan, df.values[-1,d]).ffill()
# df = df.ffill()
df.to_csv(f'{folder_path}postprocessed_v{post_process_version}_{file_name}.csv')
# df = pd.merge(df1, df2)
# postdata = {
# "iteration": iteration.tolist(),
# "Av. accumulated reward": accumulated_average_reward.tolist(),
# "Accumulated reward": accumulated_reward.tolist()
#         }
# save_data2txt(f'postprocessed_{file_name}',postdata)
# save_data2csv(f'postprocessed_{file_name}')
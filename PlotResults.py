from functions import *
from environment import *
from init import *
import plotly.graph_objects as go
import pandas as pd
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# GENERATE VIDEO OPTIONS
# ------------------------------------------------------------------------
save_video = False
Plot = True

# READ DATA FROM FILE
# ------------------------------------------------------------------------
read_data = True
if read_data:
  data = read_data_from_file("results.txt")
  df = pd.DataFrame(data)
  indexes = np.array(df['index'].tolist())
  states = np.array(df['state'].tolist())
  angles = np.array(states[:,1])
  TCP = np.array(df['TCP'].tolist())

# SETUP PLOT
# ------------------------------------------------------------------------
if Plot or save_video:
  # Plot configuration
  fig = go.Figure()
  layout = dict(
      autosize=False,
      width=500,
      height=500,
      margin=dict(
          l=50,
          r=50,
          b=100,
          t=100,
          pad=4
      ),
          scene=dict(
              xaxis=dict(visible=False),
              yaxis=dict(visible=False),
              zaxis=dict(visible=False),
              aspectmode='data',
              aspectratio=dict(x=1, y=1, z=1)))
  pltLaser = go.Scatter3d(
                  x=np.array(lcd.points)[:,0], y=np.array(lcd.points)[:,1], z=np.array(lcd.points)[:,2],
                  mode='markers',
                  marker=dict(size=1, color=('red'))
              )
  pltFilament = go.Scatter3d(
                  x=np.array(fcd.points)[:,0], y=np.array(fcd.points)[:,1], z=np.array(fcd.points)[:,2],
                  mode='markers',
                  marker=dict(size=1, color=('blue'))
              )
  pltPiece = go.Scatter3d(
              x=np.array(env.mod_dcgeometry.points)[:,0], y=np.array(env.mod_dcgeometry.points)[:,1], z=np.array(env.mod_dcgeometry.points)[:,2],
              mode='markers',
              name='Piece',
              marker=dict(size=1, color=('black'))
          )
  vectors = []
  for v,vector in enumerate(TCP[:,:3,3]):
    R = np.array(TCP[v,:3,:3])
    T = np.array(TCP[v,:3,3])
    arr = np.array([0,0,1,1])
    scale = 1
    H = np.vstack([np.hstack([R, T.reshape(-1, 1)]), np.array([0, 0, 0, scale])])
    vectors.append(np.dot(H,arr))
  vectors = np.array(vectors)
  v = [np.array(vectors[0,0])-np.array(TCP[0,0,3]),np.array(vectors[0,1])-np.array(TCP[0,1,3]),np.array(vectors[0,2])-np.array(TCP[0,2,3])]
  v = 1*(v/np.linalg.norm(v))
  pltTCP = go.Cone(
    x=np.array(TCP[0,0,3]),
    y=np.array(TCP[0,1,3]),
    z=np.array(TCP[0,2,3]),
    u=np.array(v[0]),
    v=np.array(v[1]),
    w=np.array(v[2]),
    colorscale='Viridis',  # You can choose a different colorscale
    sizemode='scaled',
    showscale=False,
    # sizeref=2,  # Adjust the size of the cone head
    anchor="tip",
    name='TCP'
  )

  fig.add_traces(data=[pltLaser,pltFilament,pltPiece])
  camera = dict(eye=dict(x=2, y=2, z=2))
  fig.update_layout(layout,scene_camera=camera,scene=dict(aspectmode='data',))
  fig.add_traces(data=[pltTCP])
  fig['data'][0]['name'] = 'Laser'
  fig['data'][1]['name'] = 'Filament'
if save_video:
  frames = []
  frame =  plotly_fig2array(fig)
  frames.append(frame)
if read_data:
  max_steps_per_episode = indexes.shape[0]
else:
  max_steps_per_episode = ext_trajectory.shape[0] #10000
lastIndex = 0
process_step = 2

for timestep in range(0, max_steps_per_episode):
  state = env.reset()
  # This needs to reset the piece on each iteration
  env.process_index = lastIndex
  rt = 1
  print(timestep)
  # Analyze next step in process
  if read_data:
    lastAngle = angles[timestep]
    env.step(lastAngle)
  if Plot or save_video:
    # print(TCP[timestep,:,3], vectors[timestep,:])
    fig.update_traces(x=np.array(env.mod_dcgeometry.points)[:,0], y=np.array(env.mod_dcgeometry.points)[:,1], z=np.array(env.mod_dcgeometry.points)[:,2], selector=dict(name='Piece'))
    v = [np.array(vectors[timestep,0])-np.array(TCP[timestep,0,3]),np.array(vectors[timestep,1])-np.array(TCP[timestep,1,3]),np.array(vectors[timestep,2])-np.array(TCP[timestep,2,3])]
    v = 1*(v/np.linalg.norm(v))
    fig.update_traces(x=np.array(TCP[timestep,0,3]), y=np.array(TCP[timestep,1,3]), z=np.array(TCP[timestep,2,3]), u=np.array(v[0]), v=np.array(v[1]), w=np.array(v[2]), selector=dict(name='TCP'))
    if Plot:
        fig.show()
    # raise ValueError('Test 1')
  if read_data:
    lastIndex = int(indexes[timestep])
    rt = env.continue_process(lastIndex)
    # print(lastAngle, lastIndex)
  else:
    lastIndex = lastIndex + process_step
    rt = env.continue_process(lastIndex)
  if rt == 0:
    break
  if save_video:
    frame =  plotly_fig2array(fig)
    frames.append(frame)
if save_video:
  # quantity_frames = len(frames)
  fps = 2
  print("Saving video ...")
  print(f'Number of frames {len(frames)}. Time of video: {frames/fps} s')
  clip = ImageSequenceClip(frames, fps=fps)  # Adjust FPS as needed
  clip.write_videofile("my_animation.mp4", codec="libx264", audio=False)
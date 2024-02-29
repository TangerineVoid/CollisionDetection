import numpy as np
from functions import *
from environment import *
from init import *
import plotly.graph_objects as go
import tensorflow as tf
import keras
from keras import layers


# Enable/Disable Plot
Plot = False
# Enable/Disable Save Results to txt
Save = True
# Configuration parameters for the whole setup
gamma = 0.9#9  # Discount factor for past rewards
max_steps_per_episode = 1000#10000
geometry_segments =  ext_trajectory.shape[0]
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
num_inputs = 2 # state and one angle
num_actions = 3
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []

# state, reward, collision = env.step(0)
if Plot:
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
              aspectmode='data'))
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
  pltLastPoint = go.Scatter3d(
                  x=np.array(env.mod_dcgeometry.points)[-9:-1,0], y=np.array(env.mod_dcgeometry.points)[-9:-1,1], z=np.array(env.mod_dcgeometry.points)[-9:-1,2],
                  mode='markers',
                  name='Last Point',
                  marker=dict(size=2, color=('red'))
              )
  pltTrajectory = go.Scatter3d(
                  x=np.array(env.mod_trajectory)[:,0], y=np.array(env.mod_trajectory)[:,1], z=np.array(env.mod_trajectory)[:,2],
                  mode='markers',
                  name='Trajectory',
                  marker=dict(size=2, color=('green'))
              )
  # pltTCP= go.Scatter3d(
  #                 x=np.array(env.TCP)[:,0], y=np.array(env.TCP)[:,1], z=np.array(env.TCP)[:,2],
  #                 mode='markers',
  #                 name='Trajectoryt',
  #                 marker=dict(size=2, color=('green'))
  #             )
  fig.add_traces(data=[pltLaser,pltFilament,pltTrajectory])
  fig.add_annotation(text=f"episode_count = 0", showarrow = False)
  fig.add_annotation(text=f"timestep = 0", yshift=-10, showarrow = False)
  fig.update_layout(layout)
  camera = dict(
      eye=dict(x=2, y=2, z=2)
  )
  fig['data'][0]['name'] = 'Laser'
  fig['data'][1]['name'] = 'Filament'
  fig.update_layout(scene_camera=camera)

  fig.add_traces(data=[pltPiece, pltLastPoint])
  fig.update_layout(autosize=True)
  # fig.update_annotations(selector=0, text=f"episode_count = {episode_count}")

running_reward = 0
episode_count = 0
lastIndex = 0
episode = 0
process_step = 20
angle_step = 5
while True:  # Run until solved
    print(env.process_index)
    state = env.reset()
    # This needs to reset the piece on each iteration
    env.process_index = lastIndex

    episode_reward = 0
    angletotal = 0
    # process_index = 0
    # state, reward, collision = env.step(0)
    # env.continue_process(0)
    rt = 1
    print('Current Geometry')
    if Plot:
      fig.update_traces(x=np.array(env.mod_dcgeometry.points)[:,0], y=np.array(env.mod_dcgeometry.points)[:,1], z=np.array(env.mod_dcgeometry.points)[:,2], selector=dict(name='Piece'))
      fig.update_traces(x=np.array(env.mod_dcgeometry.points)[-9:-1,0], y=np.array(env.mod_dcgeometry.points)[-9:-1,1], z=np.array(env.mod_dcgeometry.points)[-9:-1,2], selector=dict(name='Last Point'))
      fig.update_traces(x=np.array(env.mod_trajectory)[:,0], y=np.array(env.mod_trajectory)[:,1], z=np.array(env.mod_trajectory)[:,2], selector=dict(name='Trajectory'))
      # fig.data = fig.data[:-2]
      # pltPiece = go.Scatter3d(
      #         x=np.array(env.mod_dcgeometry.points)[:,0], y=np.array(env.mod_dcgeometry.points)[:,1], z=np.array(env.mod_dcgeometry.points)[:,2],
      #         mode='markers',
      #         name='Piece',
      #         marker=dict(size=1, color=('black'))
      #     )
      # pltLastPoint = go.Scatter3d(
      #               x=np.array(env.mod_dcgeometry.points)[-9:-1,0], y=np.array(env.mod_dcgeometry.points)[-9:-1,1], z=np.array(env.mod_dcgeometry.points)[-9:-1,2],
      #               mode='markers',
      #               name='Last Point',
      #               marker=dict(size=2, color=('red'))
      #           )
      # fig.add_traces(data=[pltPiece,pltLastPoint])
      fig.show()
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # collision = env.check_Collision()
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            # print(action_probs)
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            if action == 2:
              angle_change = 0
            elif action == 1:
              angle_change = -angle_step
            elif action == 0:
              angle_change = angle_step

            angle = angle_change
            angletotal = angletotal + angle
            state, reward, collision = env.step(angle)
            data = {
                "episode" : episode_count,
                "step" : timestep,
                "state" : state,
                "index" : lastIndex,
                "collision" : collision,
                "reward" : reward,
                "TCP" : env.mod_TCP.reshape((4,4)).tolist() #env.mod_trajectory[0,:].tolist()
            }
            save_data2txt("training.txt", data)
            rewards_history.append(reward)
            episode_reward += reward
            # Collision condition
            if collision == 0:
              print('Collision:', collision)
              break
        print('finished for')
        # print(env.process_index)

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
    # Continue next step in process
    # Analyze next step in process
    if collision == 0:
      # Update plot
      if Plot:
        fig.update_traces(x=np.array(env.mod_dcgeometry.points)[:,0], y=np.array(env.mod_dcgeometry.points)[:,1], z=np.array(env.mod_dcgeometry.points)[:,2], selector=dict(name='Piece'))
        fig.update_traces(x=np.array(env.mod_dcgeometry.points)[-9:-1,0], y=np.array(env.mod_dcgeometry.points)[-9:-1,1], z=np.array(env.mod_dcgeometry.points)[-9:-1,2], selector=dict(name='Last Point'))
        fig.update_traces(x=np.array(env.mod_trajectory)[:,0], y=np.array(env.mod_trajectory)[:,1], z=np.array(env.mod_trajectory)[:,2], selector=dict(name='Trajectory'))
        fig.show()
      lastIndex = lastIndex + process_step
      # print('step: ',timestep,'state: ',state, 'index: ', lastIndex, 'collision: ', collision, 'reward: ', reward)
      rt = env.continue_process(lastIndex)
      data = {
                "episode" : episode_count,
                "step" : timestep,
                "state" : state,
                "index" : lastIndex,
                "collision" : collision,
                "reward" : reward,
                "TCP" : env.mod_TCP.reshape((4,4)).tolist()#env.mod_trajectory[0,:].tolist()
            }
      # save_data2txt("angles.txt", [f"{lastIndex}, {state[1]}, {env.mod_TCP.reshape(3)}"])
      save_data2txt("results.txt", data)
      print("disque no hay colision")
    else:
      env.continue_process(0)
    if rt == 0:
      break
    # state = env.reset()
    # # This needs to reset the piece on each iteration
    # env.process_index = 0
    # episode_reward = 0
    # angletotal = 0
    # process_index = 0
    # rt = 1

    # Log details
    episode_count += 1
    # Stop Training sooner
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 10:  # Condition to consider the task solved
        print("Solved at episode {} with angle = {}!".format(episode_count,angletotal))
        break

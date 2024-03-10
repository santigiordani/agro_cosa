from Area import *
from State import *
from Field import *


import pandas as pd
import numpy as np

df = pd.read_csv("simulated_dataset.csv")

# Create a new index with additional rows
new_index = np.arange(len(df) * 20) / 20

# Reindex the DataFrame with the new index
df = df.reindex(new_index)
test_df = df.reindex(new_index)

# Interpolate values quadratically
df = df.interpolate(method='linear')

# Optionally, if you want to reset the index
df = df.reset_index(drop=True)

# Now, df contains 100 additional rows interpolated quadratically between existing rows
df.drop(columns=['gps_date'], inplace=True)
df.dropna(inplace=True)

# Take only the last 4000 rows
df = df.tail(4000)



states = []
for i in df.index:
  states.append(State(df.loc[i]))




import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, (ax, dax) = plt.subplots(1, 2)

# Reset test field
test_field.reset_density()
test_field.compute_mask()

# Update function
def update(frame):

  # Clear axis
  ax.cla()

  # Plot areas
  test_area.plot(ax, color='grey', alpha=.2)
  test_forbidden.plot(ax, color='grey', alpha=.2)

  # Get state
  state = states[frame]
  nstate = states[frame + 1]

  # Add to density
  test_field.add_point(nstate, new_model)

  # Plot density
  test_field.plot(ax)

  # Plot trajectory
  for _state, _nstate in zip(states, states[1:frame + 2]):
    ax.plot([_state.x, _nstate.x], [_state.y, _nstate.y], color=_nstate.state_color, alpha=.8)

  # Compute derivas
  derivas_position = nstate.get_derivas(test_pmodel, [test_forbidden])

  # Plot new state
  nstate.plot(ax, color='black')
  nstate.plot_derivas(ax)

  # Set window size and center
  window_size = 200
  ax.set_xlim(nstate.x - window_size, nstate.x + window_size)
  ax.set_ylim(nstate.y - window_size, nstate.y + window_size)

  # For manija
  print(f"\rFrame: {frame + 1} / {len(states) - 1}", end='')

  # Just for debugging
  dax.cla()
  dax.imshow(test_field.density, origin='lower', extent=[test_field.x_min, test_field.x_max, test_field.y_min, test_field.y_max])

ani = FuncAnimation(fig, update, frames=len(states) - 1, interval=15)
ani.save("interpolated_density_animation.mp4")
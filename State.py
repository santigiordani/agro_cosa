import numpy as np

from Field import *


class State():

  width = 40
  wwidth = 6
  sample_num = 4
  sigma = 4

  def __init__(self, row):

    # Self position
    self.x = row["east"]
    self.y = row["north"]
    self.z = 3 # Unless I get a better estimate

    # Velocity
    self.x_v = row["north_v"] # Esto esta intercambiado en el csv
    self.y_v = row["east_v"] # Esto esta intercambiado en el csv
    self.speed = row["gps_vel"]
    self.course = row["gps_course"] * np.pi / 180

    # Perpendicular
    if self.speed:
      self.perp = np.array([-self.y_v, self.x_v]) / self.speed
    else:
      self.perp = np.zeros(2)

    # Wind
    self.x_w = row["east_w"]
    self.y_w = row["north_w"]

    # Caudal
    self.caudal = row["caudal"]

    # Sample
    self.sample = []

    # Deriva
    self.derivas = []

    # Colors
    self.state_color = 'grey'
    self.derivas_color = []

  def get_derivas(self, pmodel, areas, tol=5):

    # Sample
    self.sample = np.array([self.x, self.y]) \
                + np.linspace(-self.width / 2, self.width / 2, self.sample_num).reshape(-1, 1) \
                * self.perp.reshape(1, -1)

    # Deriva
    self.derivas = [pmodel(self, p) for p in self.sample]

    # Derivas color
    self.derivas_color = []
    for p in self.derivas:

      # Color rule
      if p in test_field:
        color = 'red'
      elif test_field.dist(p) < tol:
        color = 'yellow'
      else:
        color = 'green'

      self.derivas_color.append(color)

    # Point color
    if 'red' in self.derivas_color:
      color = 'red'
    elif 'yellow' in self.derivas_color:
      color = 'yellow'
    else:
      color = 'green'

    self.state_color = color

    return self.derivas
  
  def plot(self, ax, **kwargs):

    """
    Plots the vehicle.
    """

    if self.speed:
      x_v = self.x_v / self.speed
      y_v = self.y_v / self.speed
    else:
      x_v = 0
      y_v = 0

    head = ax.quiver(self.x, self.y, x_v, y_v, **kwargs)
    body = ax.plot([self.sample[0][0], self.sample[-1][0]],
                   [self.sample[0][1], self.sample[-1][1]], **kwargs)
    
    return head, body
  
  def plot_derivas(self, ax, **kwargs):

    """
    Plots the derivas.
    """

    # Circles for the derivas
    derivas = []
    lines = []
    for origin, deriva, color in zip(self.sample, self.derivas, self.derivas_color):
      derivas.append(ax.scatter(deriva[0], deriva[1], s=self.caudal / 10, color=color, alpha=.4))
      lines.append(ax.plot([origin[0], deriva[0]], [origin[1], deriva[1]], '--', color='grey'))
    
    return derivas, lines

test_state = State({"east" : 0, "north" : 0, "east_v" : 1, "north_v" : -1, # I interchanged this to be consistent with the csv
                    "east_w" : 0, "north_w" : 0, "caudal" : 2000,
                    "gps_vel" : np.sqrt(2), "gps_course" : 45})
def test_pmodel(state, p=None):

  """
  Pointwise model of deriva.
  """

  # If there is p, calculate pmodel there
  if p is None:
    x = state.x
    y = state.y
  else:
    x = p[0]
    y = p[1]

  # Velocity units change: m/s per km/h
  VUC = 1 / 3.6

  # Simple UARM model
  t = np.sqrt(2 * state.z / 9.81)
  x_d = x + VUC * (state.x_v + state.x_w) * t
  y_d = y + VUC * (state.y_v + state.y_w) * t

  return np.array([x_d, y_d])

if __name__ == '__main__':
  print("Running State as main")
import numpy as np

from Area import *
from State import *

class Field():

  sigma = 50
  pixel_side = 5
  GREEN = np.array([0.34, 0.75, 0.32])
  ANTI_GREEN = np.ones(3) - GREEN

  def __init__(self, forbidden):
    self.forbidden = forbidden

    # Calculate x_min, x_max, y_min, y_max
    self.x_min = -300
    self.x_max = 300
    self.y_min = -300
    self.y_max = 300

    # Calculate pixels for imshow
    self.width = int((self.x_max - self.x_min) / self.pixel_side)
    self.height = int((self.y_max - self.y_min) / self.pixel_side)

    # Meshgrid
    x = np.linspace(self.x_min, self.x_max, self.width)
    y = np.linspace(self.y_min, self.y_max, self.height)
    self.X, self.Y = np.meshgrid(x, y)

    # Fields
    self.mask = np.zeros(self.X.shape + (3,))
    self.density = np.zeros_like(self.X)

  def __contains__(self, p):
    return sum(p in area for area in self.forbidden) > 0

  def dist(self, p):
    return min(area.dist(p) for area in self.forbidden)

  def add_point(self, point, model):
    self.density += model(self.X, self.Y, point)

  def compute_mask(self):

    # Temp mask
    red_mask = np.zeros_like(self.mask)

    # Check for forbidden points
    for area in self.forbidden:
      for x_row, y_row in zip(self.X, self.Y):
        for x, y in zip(x_row, y_row):
          if [x, y] in area:

            # Add inverted-red gaussian
            Z = np.exp(-((self.X - x)**2 + (self.Y - y)**2) / self.sigma)
            Z = np.stack((np.zeros_like(Z), Z, Z), axis=-1)
            red_mask += Z

    # Normalize temp_mask
    red_max = np.max(red_mask)
    red_mask /= red_max

    # Compute red intensity
    green_mask = self.ANTI_GREEN * np.expand_dims(1 - red_mask[:, :, 1], -1)

    # Combine temp_mask and self.mask
    self.mask += red_mask + green_mask

    # Normalize final mask
    mask_max = np.max(self.mask)
    self.mask /= mask_max
  
  def reset_density(self):
    self.density = np.zeros_like(self.density)

  def plot(self, ax, **kwargs):

    # Normalize density
    #density_max = np.max(self.density)
    #if density_max > 0:
    #  self.density /= density_max

    # Clip density
    clipped_density = np.minimum(1, self.density)

    # Final mask
    final_mask = np.ones_like(self.mask) - self.mask * np.expand_dims(clipped_density, -1)

    # Plot
    plot = ax.imshow(final_mask, interpolation='bilinear', origin='lower', extent=[self.x_min, self.x_max, self.y_min, self.y_max])

    return plot


test_field = Field([test_forbidden])
def test_model(X, Y, state):

  """
  Recieves a meshgrid X, Y and a State.
  Returns a meshgrid Z with distribution of the point.

  This is a gaussian model 5 meters to the right of position
  """

  x = state.x
  y = state.y
  c = state.caudal / 5000

  return c * np.exp(-((X - x - 5)**2 + (Y - y)**2) / 100)

def new_model(X, Y, state):

  """
  Recieves a meshgrid X, Y and a State.
  Returns a meshgrid Z with distribution of the point.

  This is also gaussian, but elongated along the direction perpendicular to the
  vehicles velocity.
  """

  # Rotation
  cos = np.cos(state.course)
  sin = np.sin(state.course)
  A = np.array([[cos, -sin], [sin, cos]])

  # Transform domain
  domain = np.stack([X - state.x, Y - state.y], axis=-1)
  rot_domain = np.matmul(domain, A.T)
  X = rot_domain[:, :, 0]
  Y = rot_domain[:, :, 1]

  # Transform wind
  wind = np.array([state.x_w, state.y_w])
  rot_wind = np.matmul(wind, A.T)

  # Velocity units change: m/s per km/h
  VUC = 1 /3.6

  # Compute center of spray with simple UARM model
  t = np.sqrt(2 * state.z / 9.81)
  x_d = VUC * wind[0] * t
  y_d = VUC * (state.speed + wind[1]) * t
  c = state.caudal / 100000

  # Density
  Z = c * np.exp(-(((X - x_d) / (state.width / 2))**2 + ((Y - y_d) / (state.wwidth / 2))**2) / state.sigma)

  return Z


if __name__ == '__main__':

  import matplotlib.pyplot as plt

  fig, axs = plt.subplots(1, 3, figsize=(12, 4))

  # Plot area
  test_area.plot(axs[0], color='grey', alpha=.2)
  test_area.plot(axs[1], color='grey', alpha=.2)
  test_area.plot(axs[2], color='grey', alpha=.2)
  # test_area.plot(axs[3], color='grey', alpha=.2)

  # Before adding the point
  test_field.reset_density()
  test_field.compute_mask()
  test_field.plot(axs[0])

  # After adding the point
  test_field.add_point(test_state, new_model)
  test_field.plot(axs[1])

  # Full color
  test_field.density = np.ones_like(test_field.density)
  test_field.plot(axs[2])

  # Only green mask
  # axs[3].imshow(test_field.green_mask, interpolation='bilinear', origin='lower', extent=[test_field.x_min, test_field.x_max, test_field.y_min, test_field.y_max])

  plt.show()
import numpy as np


def oriented_angle(u, v):
    """
    Calculate the oriented angle between two 2D vectors in radians.
    """
    dot_product = np.dot(u, v)
    cross_product = np.cross(u, v)
    angle = np.arctan2(cross_product, dot_product)
    return angle


class Area():

  def __init__(self, vertices):
    self.vertices = np.array(vertices)
    self.loop = np.concatenate((self.vertices, self.vertices[0].reshape(1, -1)))

  def __contains__(self, p):
    winding_number = sum(oriented_angle(x - p, y - p) for x, y in zip(self.loop, self.loop[1:]))
    return abs(winding_number) > .1

  def plot(self, ax, **kwargs):
    ax.fill(self.vertices[:, 0], self.vertices[:, 1], **kwargs)

  def dist(self, p):

    # If in, return
    if p in self:
      return 0
    
    # Distances
    distances = []

    # Compute distances to vertices
    for v in self.vertices:
      distances.append(np.linalg.norm(p - v))

    # Compute distances to sides
    for P, Q in zip(self.loop, self.loop[1:]):
      
      # Aux
      A = Q - P
      B = p - P
      side_len = np.linalg.norm(A)
      
      # Check if projection of p on PQ line lies within PQ segment
      if 0 <= np.dot(A, B) / side_len**2 <= 1:
        distances.append(abs(np.cross(A, B) / side_len))

    # Take the minimuim
    return min(distances)


test_area = Area([(-248.85, 130.04), (-252.21, 140.47), (-229.2, 141.03),
                  (-203.4, 142.12), (-163.7, 182.42), (-73.59, 176.1),
                  (-51.64, 134.92), (-21.63, 120.83), (14.93, 120.72),
                  (34.49, 125.82), (92.2, 160.78), (120.17, 157.9),
                  (149.65, 131.37), (206.56, 66.11), (213.64, 59.56),
                  (229.5, 0), (163.01, -133.66), (152.57, -142.98),
                  (126.63, -156.97), (140, -202.14), (14.66, -217.12),
                  (-6.76, -234.44), (-28.62, -230.44), (-44.2, -216.79),
                  (-127.4, -171.51), (-190.25, -154.86), (-240.88, -131.67),
                  (-191.31, -72.4), (-64.29, 20.05)])
test_forbidden = Area([(225, -200), (225, 200), (300, 200), (300, -200)])


# Tests for Area class
if __name__ == '__main__':
    
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()

  # Plot areas
  test_area.plot(ax, color='gray', alpha=.3)
  test_forbidden.plot(ax, color='gray', alpha=.3)

  # Plot points for visual check of __contains__
  for p in np.random.randint(-200, 200, size=(40, 2)):
    if p in test_area:
      color = 'blue'
      marker = '+'
    else:
      color = 'red'
      marker = 'x'
    ax.scatter(p[0], p[1], color=color, marker=marker)

  # Plot some squares for visual check of dist
  for p in np.random.randint(-200, 200, size=(10, 2)):
    ax.scatter(p[0], p[1], marker='s', label=f'At {test_area.dist(p):.0f}m')

  ax.legend()
  plt.show()
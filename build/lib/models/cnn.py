from flax import linen as nn

# Standard CNN model
class CNN(nn.Module):
    """A simple CNN model."""
    def setup(self):
      self.features = Features()
      self.head = Head()
 
    def __call__(self, x):
      x = self.features(x)
      x = self.head(x)
      return x
    
    def features_transform(self, x):
      return self.features(x)

class Features(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    return x

class Head(nn.Module):
  @nn.compact
  def __call__(self,x):
    x = nn.Dense(features = 16)(x)
    x = nn.relu(x)
    x = nn.Dense(features = 1)(x) 
    return x  

# Variable projection CNN model
class VarPro_CNN(nn.Module):
    def setup(self):
      self.outer_layers = VPro_Outer_Layers()
      self.last_layer = VPro_Last_Layer()

    def __call__(self,x):
        x = self.outer_layers(x)
        x = self.last_layer(x)
        return x
    
    def apply_outer_layers(self,x):
      x = self.outer_layers(x)
      return x


class VPro_Outer_Layers(nn.Module):
        @nn.compact
        def __call__(self,x):
          x = nn.Conv(features=32, kernel_size=(3, 3))(x)
          x = nn.relu(x)
          x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
          x = nn.Conv(features=64, kernel_size=(3, 3))(x)
          x = nn.relu(x)
          x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
          x = x.reshape((x.shape[0], -1))  # flatten
          x = nn.Dense(features = 16)(x)
          return x
        
class VPro_Last_Layer(nn.Module):
    @nn.compact
    def __call__(self,x):
        x = nn.Dense(features=1)(x)
        return x
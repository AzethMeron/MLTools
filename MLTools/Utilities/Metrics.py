import zlib
import pickle

class Metrics:
  def __init__(self, path):
    self.metrics = []
    self.path = path
  def __dump_object(self, obj):
    return zlib.compress(pickle.dumps(obj))
  def __load_object(self, obj):
    return pickle.loads(zlib.decompress(obj))
  def __describe_entry(self, dict):
    output = []
    for key, value in dict.items():
      if type(value) == int:
        output.append(f"{key}: {value}")
      elif type(value) == float:
        output.append(f"{key}: {value:.4}")
      else:
        output.append(f"{key}: {value}")
    output = ", ".join(output)
    return output
  def update(self, dict):
    self.metrics.append( self.__dump_object(dict) )
    self.save()
    print(self.__describe_entry(dict))
  def save(self):
    if self.path:
      with open(self.path, "wb") as f:
        pickle.dump(self.metrics, f)
  def load(self):
    if self.path:
      with open(self.path, "rb") as f:
        self.metrics = pickle.load(f)
  def __getitem__(self, index):
    return self.__load_object(self.metrics[index])
  def __len__(self):
    return len(self.metrics)
  def __iter__(self):
    for metric in self.metrics:
      yield self.__load_object(metric)
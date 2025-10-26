import pickle
import zlib
import os

def SaveBin(path, obj):
    with open(path, "wb") as f:
        data = pickle.dumps(obj)
        data = zlib.compress(data)
        f.write(data)

def LoadBin(path):
    with open(path, "rb") as f:
        data = f.read()
        data = zlib.decompress(data)
        return pickle.loads(data)
        
def SaveDump(obj):
    data = pickle.dumps(obj)
    return zlib.compress(data)

def LoadDump(data):
    data = zlib.decompress(data)
    return pickle.loads(data)

def LoadOrCompute( path, func ):
  if os.path.exists(path):
    return LoadBin(path)
  else:
    result = func()
    SaveBin(path, result)
    return result

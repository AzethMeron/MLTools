import pickle
import zlib

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
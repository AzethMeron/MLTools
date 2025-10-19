
from MLTools.Utilities import LoadBin

from torch.utils.data import Dataset
from pathlib import Path
import urllib.request
import os

class TinyImageNetFeatures(Dataset):
    URL_ResNet50_train = "https://github.com/AzethMeron/MLTools/raw/e82ed9f07bbcac622e3955c145554fc3d1a90a7e/Dataset/TinyImageNet_ResNet50_train.bin"
    URL_ResNet50_val = "https://github.com/AzethMeron/MLTools/raw/e82ed9f07bbcac622e3955c145554fc3d1a90a7e/Dataset/TinyImageNet_ResNet50_val.bin"
    def __init__(self, path):
        data = LoadBin(path)
        self.wnid_to_id = data['wnid_to_id']
        self.id_to_wnid = data['id_to_wnid']
        self.id_to_class = data['id_to_class']
        self.encoded = data['encoded']
        self.filenames = sorted(self.encoded.keys())
    def __len__(self):
        return len(self.encoded)
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        features, onehot = self.encoded[filename]
        return features, onehot
    @staticmethod
    def download(path,
                train_url=URL_ResNet50_train,
                val_url=URL_ResNet50_val):

        path = Path(path)
        train_path = path / "train.bin"
        val_path   = path / "val.bin"
        os.makedirs(path, exist_ok=True)

        if not train_path.exists():
            print(f"[i] Downloading {train_url} ...")
            with urllib.request.urlopen(train_url) as connection:
              data = connection.read()
              with open(train_path, 'wb') as f:
                f.write(data)
            print(f"[✓] Saved to {train_path}")

        if not val_path.exists():
            print(f"[i] Downloading {val_url} ...")
            with urllib.request.urlopen(val_url) as connection:
              data = connection.read()
              with open(val_path, 'wb') as f:
                f.write(data)
            print(f"[✓] Saved to {val_path}")

        return train_path, val_path
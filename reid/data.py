from pathlib import Path
import io
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def image_from_array(x):
    if isinstance(x, bytes) or (getattr(x, 'ndim', 0) == 1):
        return Image.open(io.BytesIO(bytes(x))).convert('RGB')
    x = np.asarray(x)
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.transpose(1, 2, 0)
    return Image.fromarray(x.astype('uint8')).convert('RGB')


class Images(Dataset):
    def __init__(self, path, group='', low=False, train=False):
        self.path, self.group = str(path), group
        self.key = 'images_low_resolution' if low else 'images'
        with h5py.File(self.path, 'r') as f:
            self.length = len(f[group][self.key] if group else f[self.key])
        ops = [T.Resize((256, 128), antialias=True)]
        if train:
            ops += [T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop((256, 128))]
        ops += [T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])]
        if train:
            ops += [T.RandomErasing(p=.5, value=0)]
        self.transform = T.Compose(ops)
        self.file = None

    def __len__(self): return self.length

    def __getitem__(self, i):
        if self.file is None: self.file = h5py.File(self.path, 'r')
        g = self.file[self.group] if self.group else self.file
        return self.transform(image_from_array(g[self.key][i])), i


def metadata(path, group):
    with h5py.File(path, 'r') as f:
        return {k: np.asarray(f[group][k]) for k in ['person_ids','camera_ids','image_ids']}


def inspect(root):
    info = {}
    for path in sorted(Path(root).glob('*.h5')):
        rows = {}
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset): rows[name] = {'shape':list(obj.shape),'dtype':str(obj.dtype)}
            f.visititems(visit)
        info[path.name] = rows
    return info

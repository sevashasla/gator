import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F

# "Pair": apply a transform on a pair
# "Both": apply the exact same transform to both images

class ComposePair(T.Compose):
    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2

class NormalizeBoth(T.Normalize):
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2
class ToTensorBoth(T.Transform):
    def __init__(self):
        super().__init__()
        self.tf = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])

    def __call__(self, img1, img2):
        img1 = self.tf(img1)
        img2 = self.tf(img2)
        return img1, img2
        
class RandomCropPair(T.RandomCrop): 
    # the crop will be intentionally different for the two images with this class
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2
    
class ResizePair(T.Resize):
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2

class ColorJitterPair(T.ColorJitter): 
    # can be symmetric (same for both images) or assymetric (different jitter params for each image) depending on assymetric_prob  
    def __init__(self, assymetric_prob, **kwargs):
        super().__init__(**kwargs)
        self.assymetric_prob = assymetric_prob
        
    def forward(self, img1, img2):
        if torch.rand(1) < self.assymetric_prob: # assymetric:
            img1 = super().forward(img1)
            img2 = super().forward(img2)
        else: # symmetric
            img1, img2 = super().forward(img1, img2)

        return img1, img2

def get_pair_transforms_gator(transform_str):
    # transform_str is eg    crop224+color+norm
    trfs = []
    for s in transform_str.split('+'):
        if s.startswith('crop'):
            size = int(s[len('crop'):])
            trfs.append(RandomCropPair(size))
        elif s.startswith('resize'):
            size = int(s[len('resize'):])
            trfs.append(ResizePair(size))
        elif s=='acolor':
            trfs.append(ColorJitterPair(assymetric_prob=1.0, brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0.0))
        elif s=="norm":
            trfs.extend([
                ToTensorBoth(),
                NormalizeBoth(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif s=='': # if transform_str was ""
            pass
        else:
            raise NotImplementedError('Unknown augmentation: '+s)
    
    if "norm" not in transform_str:
        trfs.append( ToTensorBoth() )
    
    if len(trfs)==1:
        return trfs
    else:
        return ComposePair(trfs)
        
        
        
        
        

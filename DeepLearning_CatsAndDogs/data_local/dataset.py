import os
from torch.utils import data
from torchvision import transforms as T
from PIL import Image


class DogAndCat(data.Dataset):
    def __init__(self, root, transforms=None,train = True,test = False):
        imgs =[os.path.join(root,file) for file in os.listdir(root)]
        self.imgs = imgs
        if transforms is None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
            if not train or test:
                self.transforms = T.Compose([T.Resize(224),
                                            T.CenterCrop(224),
                                            T.ToTensor(),
                                            normalize])
            else:
                self.transforms = T.Compose([T.Resize(256),
                                        T.RandomResizedCrop(224),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor(),
                                        normalize])

    def __getitem__(self, item):
       img_path = self.imgs[item]
       label = 1 if "dog" in (img_path.split("/")[-1]).split(".")[-3] else 0
       data = Image.open(img_path)
       data = self.transforms(data)
       return data,label

    def __len__(self):
        return len(self.imgs)


if __name__ =="__main__":
    a = DogAndCat(root ='./a')
    print(a.__getitem__(3))
    print(a)
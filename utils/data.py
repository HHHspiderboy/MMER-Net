import torch.utils.data as Data
from torchvision import transforms
from PIL import Image,ImageOps,ImageFilter
import random
import os
class IRSTDDataSet(Data.Dataset):
    def __init__(self,args,mode='train'):
        self.dataDir=args.data_dir
        self.crop_size=args.crop_size
        self.base_size=args.base_size
        self.batch_size=args.batch_size
        self.mode=mode

        if mode=='train':
            traversalList=os.path.join(self.dataDir,'train.txt')
        else:
            traversalList=self.dataDir+'test.txt'

        #img label 
        self.imgList=self.dataDir+'imgs'
        self.labelList=self.dataDir+'labels'
        self.sequenceList=[]
        with open(traversalList,'r') as f:
            self.sequenceList+=[line.strip() for line in f.readlines()]
        
        self.BNtransform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([-0.1246], [1.0923]),#处理为灰度图像后仅需要mean+std
        ])
    def __len__(self):
        return len(self.sequenceList)

    def __getitem__(self, index):
        cur_name=self.sequenceList[index]

        cur_img=self.imgList+'/'+cur_name+'.png'
        cur_label=self.labelList+'/'+cur_name+'.png'
        #分别转化为灰度图像和二值图像进行处理
        img,label=Image.open(cur_img).convert('L'),Image.open(cur_label).convert('1')

        #如果是训练  则需要数据增强
        if self.mode=='train':
            img,label=self.dataAug(img,label)
        else:
            img,label=img.resize((self.base_size,self.base_size),Image.BILINEAR),label.resize((self.base_size,self.base_size),Image.NEAREST)
        
        img,label=self.BNtransform(img),transforms.ToTensor()(label)
        
        return img,label
        
    def dataAug(self,img,mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))
        return img, mask

            
        
        
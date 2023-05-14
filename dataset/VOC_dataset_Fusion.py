import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import  Image
import random

def flip(img_ir, img_vis, img_mask, boxes):
    img_ir = img_ir.transpose(Image.FLIP_LEFT_RIGHT)
    img_vis = img_vis.transpose(Image.FLIP_LEFT_RIGHT)
    img_mask = img_mask.transpose(Image.FLIP_LEFT_RIGHT)

    w = img_ir.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img_ir, img_vis,img_mask, boxes

class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        'Lamp', 'Car', 'Bus', 'Motorcycle', 'Truck', 'People',

    )
    def __init__(self,root_dir,resize_size=[800,1333],split='trainval',use_difficult=False,is_train = True, augment = None):
        self.root=root_dir
        self.use_difficult=use_difficult
        self.imgset=split

        self._annopath = os.path.join(self.root, "Annotation", "%s.xml")
        self._imgpath_ir = os.path.join(self.root, "Ir", "%s.png")
        self._imgpath_vis = os.path.join(self.root, "Vis", "%s.png")
        self._imgpath_mask = os.path.join(self.root, "Mask", "%s.png")
        self._imgsetpath = os.path.join(self.root, "Main", "%s.txt")
        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.name2id=dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}
        self.resize_size=resize_size
        self.mean=[0,0,0]
        self.std=[1, 1, 1]
        self.train = is_train
        self.augment = augment
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,index):

        img_id=self.img_ids[index]
        img_ir = Image.open(self._imgpath_ir%img_id).convert('L')
        img_vis = Image.open(self._imgpath_vis % img_id)
        img_mask = Image.open(self._imgpath_mask% img_id)
        anno=ET.parse(self._annopath%img_id).getroot()
        boxes=[]
        classes=[]
        for obj in anno.iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            # if not self.use_difficult and difficult:
            #     continue
            _box=obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box=[
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE=1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name=obj.find("name").text.strip()
            classes.append(self.name2id[name])

        boxes=np.array(boxes,dtype=np.float32)
        if self.train:
            if random.random() < 0.5:
                img_ir, img_vis, img_mask, boxes = flip(img_ir, img_vis,img_mask, boxes)
            if self.augment is not None:
                img_ir, img_vis, img_mask, boxes = self.augment(img_ir, img_vis, img_mask, boxes)
        img_ir = np.array(img_ir)
        img_vis = np.array(img_vis)
        img_mask = np.array(img_mask)

        # img_ir,img_vis,boxes=self.preprocess_img_boxes3(img_ir,img_vis,boxes,self.resize_size)
        img_ir,img_vis,img_mask,boxes=self.preprocess_img_boxes2(img_ir,img_vis,img_mask,boxes,self.resize_size)

        img_ir=transforms.ToTensor()(img_ir)
        img_vis = transforms.ToTensor()(img_vis)
        img_mask = transforms.ToTensor()(img_mask)
        boxes=torch.from_numpy(boxes)
        classes=torch.LongTensor(classes)

        return img_ir, img_vis, img_mask, boxes,classes


    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def preprocess_img_boxes2(self,image_ir, image_vis, image_mask,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w = image_ir.shape
        image_ir = np.expand_dims(image_ir,axis=2)
        image_ir = np.concatenate((image_ir,image_ir,image_ir),axis=2)
        image_mask = np.expand_dims(image_mask, axis=2)
        image_mask = np.concatenate((image_mask, image_mask, image_mask), axis=2)
        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_ir_resized = cv2.resize(image_ir, (nw, nh))
        image_vis_resized = cv2.resize(image_vis, (nw, nh))
        image_mask_resized = cv2.resize(image_mask, (nw, nh))
        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded_ir = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded_ir[:nh, :nw, :] = image_ir_resized

        image_paded_vis = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_vis[:nh, :nw, :] = image_vis_resized
        image_paded_mask = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_mask [:nh, :nw, :] = image_mask_resized

        if boxes is None:
            return image_paded_ir,image_paded_vis
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale

            return image_paded_ir,image_paded_vis, image_paded_mask, boxes

    def preprocess_img_boxes3(self, image_ir, image_vis, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side =400
        max_side = 600
        h, w, _ = image_ir.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        # image_ir_resized = cv2.resize(image_ir, (600, 400))
        # image_vis_resized = cv2.resize(image_vis, (600, 400))
        image_ir_resized = cv2.resize(image_ir, (nw, nh))
        image_vis_resized = cv2.resize(image_vis, (nw, nh))
        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32
        #
        image_paded_ir = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_ir[:nh, :nw, :] = image_ir_resized
        #
        image_paded_vis = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_vis[:nh, :nw, :] = image_vis_resized
        if boxes is None:
             return image_paded_ir, image_paded_vis
        else:
             boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
             boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale

        return image_ir_resized, image_vis_resized, boxes

    def collate_fn(self,data):
        imgs_list_ir, imgs_list_vis, imgs_list_mask,boxes_list,classes_list=zip(*data)
        assert len(imgs_list_ir) == len(imgs_list_vis) ==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_imgs_vis_list = []
        pad_imgs_mask_list = []
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list_ir]
        w_list = [int(s.shape[2]) for s in imgs_list_ir]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list_ir[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))
            img_vis = imgs_list_vis[i]
            pad_imgs_vis_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img_vis,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))
            img_mask = imgs_list_mask[i]
            pad_imgs_mask_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
            torch.nn.functional.pad(img_mask, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))


        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))


        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)
        batch_imgs_vis = torch.stack(pad_imgs_vis_list)
        batch_imgs_mask = torch.stack(pad_imgs_mask_list)

        return batch_imgs,batch_imgs_vis,batch_imgs_mask,batch_boxes,batch_classes


class VOCDataset_Method(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        'Lamp', 'Car', 'Bus', 'Motorcycle', 'Truck', 'People',

    )
    def __init__(self,root_dir,method='U2',resize_size=[800,1333],split='trainval',use_difficult=False,is_train = True, augment = None):
        self.root=root_dir
        self.use_difficult=use_difficult
        self.imgset=split

        # self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        # self._imgpath_ir = os.path.join(self.root, "JPEGImages_ir", "%s.png")
        # self._imgpath_vis = os.path.join(self.root, "JPEGImages_vis", "%s.png")
        # self._imgsetpath = os.path.join(self.root, "ImageSets","Main", "%s.txt")
        self._annopath = os.path.join(self.root, "Annotation", "%s.xml")
        self._imgpath_ir = os.path.join(self.root, "Ir", "%s.png")
        self._imgpath_vis = os.path.join(self.root,method, "%s.png")
        self._imgpath_mask = os.path.join(self.root, "Mask", "%s.png")
        self._imgsetpath = os.path.join(self.root, "Main", "%s.txt")
        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.name2id=dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}
        self.resize_size=resize_size
        self.mean=[0,0,0]
        self.std=[1, 1, 1]
        self.train = is_train
        self.augment = augment
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,index):

        img_id=self.img_ids[index]
        img_ir = Image.open(self._imgpath_ir%img_id).convert('L')
        img_vis = Image.open(self._imgpath_vis % img_id)
        img_mask = Image.open(self._imgpath_mask% img_id)
        anno=ET.parse(self._annopath%img_id).getroot()
        boxes=[]
        classes=[]
        for obj in anno.iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            # if not self.use_difficult and difficult:
            #     continue
            _box=obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box=[
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE=1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name=obj.find("name").text.strip()
            classes.append(self.name2id[name])

        boxes=np.array(boxes,dtype=np.float32)
        if self.train:
            if random.random() < 0.5:
                img_ir, img_vis, img_mask, boxes = flip(img_ir, img_vis,img_mask, boxes)
            if self.augment is not None:
                img_ir, img_vis, img_mask, boxes = self.augment(img_ir, img_vis, img_mask, boxes)
        img_ir = np.array(img_ir)
        img_vis = np.array(img_vis)
        img_mask = np.array(img_mask)

        # img_ir,img_vis,boxes=self.preprocess_img_boxes3(img_ir,img_vis,boxes,self.resize_size)
        img_ir,img_vis,img_mask,boxes=self.preprocess_img_boxes2(img_ir,img_vis,img_mask,boxes,self.resize_size)

        img_ir=transforms.ToTensor()(img_ir)
        img_vis = transforms.ToTensor()(img_vis)
        img_mask = transforms.ToTensor()(img_mask)
        boxes=torch.from_numpy(boxes)
        classes=torch.LongTensor(classes)

        return img_ir, img_vis, img_mask, boxes,classes


    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def preprocess_img_boxes2(self,image_ir, image_vis, image_mask,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w = image_ir.shape
        image_ir = np.expand_dims(image_ir,axis=2)
        image_ir = np.concatenate((image_ir,image_ir,image_ir),axis=2)
        image_mask = np.expand_dims(image_mask, axis=2)
        image_mask = np.concatenate((image_mask, image_mask, image_mask), axis=2)
        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_ir_resized = cv2.resize(image_ir, (nw, nh))
        image_vis_resized = cv2.resize(image_vis, (nw, nh))
        image_mask_resized = cv2.resize(image_mask, (nw, nh))
        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded_ir = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded_ir[:nh, :nw, :] = image_ir_resized

        image_paded_vis = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_vis[:nh, :nw, :] = image_vis_resized
        image_paded_mask = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_mask [:nh, :nw, :] = image_mask_resized

        if boxes is None:
            return image_paded_ir,image_paded_vis
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale

            return image_paded_ir,image_paded_vis, image_paded_mask, boxes

    def preprocess_img_boxes3(self, image_ir, image_vis, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side =400
        max_side = 600
        h, w, _ = image_ir.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        # image_ir_resized = cv2.resize(image_ir, (600, 400))
        # image_vis_resized = cv2.resize(image_vis, (600, 400))
        image_ir_resized = cv2.resize(image_ir, (nw, nh))
        image_vis_resized = cv2.resize(image_vis, (nw, nh))
        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32
        #
        image_paded_ir = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_ir[:nh, :nw, :] = image_ir_resized
        #
        image_paded_vis = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_vis[:nh, :nw, :] = image_vis_resized
        if boxes is None:
             return image_paded_ir, image_paded_vis
        else:
             boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
             boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale

        return image_ir_resized, image_vis_resized, boxes

    def collate_fn(self,data):
        imgs_list_ir, imgs_list_vis, imgs_list_mask,boxes_list,classes_list=zip(*data)
        assert len(imgs_list_ir) == len(imgs_list_vis) ==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_imgs_vis_list = []
        pad_imgs_mask_list = []
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list_ir]
        w_list = [int(s.shape[2]) for s in imgs_list_ir]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list_ir[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))
            img_vis = imgs_list_vis[i]
            pad_imgs_vis_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img_vis,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))
            img_mask = imgs_list_mask[i]
            pad_imgs_mask_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
            torch.nn.functional.pad(img_mask, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))


        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))


        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)
        batch_imgs_vis = torch.stack(pad_imgs_vis_list)
        batch_imgs_mask = torch.stack(pad_imgs_mask_list)

        return batch_imgs,batch_imgs_vis,batch_imgs_mask,batch_boxes,batch_classes




class VOCDataset2(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",'People', 'Truck', 'Lamp', 'Motorcycle', 'Bus', 'Car',
        # "color_cone",
        # "animal",
        # "car_stop",
        # "car",
        # "person",
        # "bump",
        # "bike",
        # "hole",

    )
    def __init__(self,root_dir,resize_size=[800,1333],split='trainval',use_difficult=False,is_train = True, augment = None):
        self.root=root_dir
        self.use_difficult=use_difficult
        self.imgset=split
        #
        # self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        # self._imgpath_ir = os.path.join(self.root, "JPEGImages_ir", "%s.png")
        # self._imgpath_vis = os.path.join(self.root, "JPEGImages_vis", "%s.png")
        # self._imgsetpath = os.path.join(self.root, "ImageSets","Main", "%s.txt")
        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath_ir = os.path.join(self.root, "Ir", "%s.png")
        self._imgpath_vis = os.path.join(self.root, "Vis", "%s.png")
        self._imgsetpath = os.path.join(self.root, "Main", "%s.txt")
        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.name2id=dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}
        self.resize_size=resize_size
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.train = is_train
        self.augment = augment
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,index):

        img_id=self.img_ids[index]
        img_ir = Image.open(self._imgpath_ir%img_id)
        img_vis = Image.open(self._imgpath_vis % img_id)
        anno=ET.parse(self._annopath%img_id).getroot()
        boxes=[]
        classes=[]
        for obj in anno.iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            # if not self.use_difficult and difficult:
            #     continue
            _box=obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box=[
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE=1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name=obj.find("name").text.lower().strip()
            classes.append(self.name2id[name])

        boxes=np.array(boxes,dtype=np.float32)
        if self.train:
            if random.random() < 0.5:
                img_ir, img_vis, boxes = flip(img_ir, img_vis, boxes)
            if self.augment is not None:
                img_ir, img_vis, boxes = self.augment(img_ir, img_vis, boxes)
        img_ir = np.array(img_ir)
        img_vis = np.array(img_vis)
        # img_ir,img_vis,boxes=self.preprocess_img_boxes3(img_ir,img_vis,boxes,self.resize_size)
        img_ir,img_vis,boxes=self.preprocess_img_boxes2(img_ir,img_vis,boxes,self.resize_size)

        img_ir=transforms.ToTensor()(img_ir)
        img_vis = transforms.ToTensor()(img_vis)
        boxes=torch.from_numpy(boxes)
        classes=torch.LongTensor(classes)

        return img_ir, img_vis, boxes,classes


    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def preprocess_img_boxes2(self,image_ir, image_vis,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w = image_ir.shape

        image_ir = np.expand_dims(image_ir,axis=2)
        image_ir = np.concatenate((image_ir,image_ir,image_ir),axis=2)
        # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2YUV)[:,:,0]
        # image_vis = np.expand_dims(image_vis, axis=2)
        # image_vis = np.concatenate((image_vis, image_vis, image_vis), axis=2)

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_ir_resized = cv2.resize(image_ir, (nw, nh))
        image_vis_resized = cv2.resize(image_vis, (nw, nh))

        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded_ir = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded_ir[:nh, :nw, :] = image_ir_resized

        image_paded_vis = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_vis[:nh, :nw, :] = image_vis_resized
        if boxes is None:
            return image_paded_ir,image_paded_vis
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale

            return image_paded_ir,image_paded_vis, boxes

    def preprocess_img_boxes3(self, image_ir, image_vis, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side =400
        max_side = 600
        h, w, _ = image_ir.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        # image_ir_resized = cv2.resize(image_ir, (600, 400))
        # image_vis_resized = cv2.resize(image_vis, (600, 400))
        image_ir_resized = cv2.resize(image_ir, (nw, nh))
        image_vis_resized = cv2.resize(image_vis, (nw, nh))
        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32
        #
        image_paded_ir = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_ir[:nh, :nw, :] = image_ir_resized
        #
        image_paded_vis = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded_vis[:nh, :nw, :] = image_vis_resized
        if boxes is None:
             return image_paded_ir, image_paded_vis
        else:
             boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
             boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale

        return image_ir_resized, image_vis_resized, boxes

    def collate_fn(self,data):
        imgs_list_ir, imgs_list_vis,boxes_list,classes_list=zip(*data)
        assert len(imgs_list_ir) == len(imgs_list_vis) ==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_imgs_vis_list = []
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list_ir]
        w_list = [int(s.shape[2]) for s in imgs_list_ir]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list_ir[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))
            img_vis = imgs_list_vis[i]
            pad_imgs_vis_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img_vis,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))



        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))


        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)
        batch_imgs_vis = torch.stack(pad_imgs_vis_list)
        return batch_imgs,batch_imgs_vis,batch_boxes,batch_classes
if __name__=="__main__":
    pass
    eval_dataset = VOCDataset(root_dir='/Users/VOCdevkit/VOCdevkit/VOC0712', resize_size=[800, 1333],
                               split='test', use_difficult=False, is_train=False, augment=None)
    print(len(eval_dataset.CLASSES_NAME))
    #dataset=VOCDataset("/home/data/voc2007_2012/VOCdevkit/VOC2012",split='trainval')
    # for i in range(100):
    #     img,boxes,classes=dataset[i]
    #     img,boxes,classes=img.numpy().astype(np.uint8),boxes.numpy(),classes.numpy()
    #     img=np.transpose(img,(1,2,0))
    #     print(img.shape)
    #     print(boxes)
    #     print(classes)
    #     for box in boxes:
    #         pt1=(int(box[0]),int(box[1]))
    #         pt2=(int(box[2]),int(box[3]))
    #         img=cv2.rectangle(img,pt1,pt2,[0,255,0],3)
    #     cv2.imshow("test",img)
    #     if cv2.waitKey(0)==27:
    #         break
    #imgs,boxes,classes=eval_dataset.collate_fn([dataset[105],dataset[101],dataset[200]])
    # print(boxes,classes,"\n",imgs.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,imgs.dtype)
    # for index,i in enumerate(imgs):
    #     i=i.numpy().astype(np.uint8)
    #     i=np.transpose(i,(1,2,0))
    #     i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
    #     print(i.shape,type(i))
    #     cv2.imwrite(str(index)+".jpg",i)








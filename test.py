# coding:utf-8

import torch
import torch.nn.functional as F
from torchvision import transforms

from model.fcos_fusion import Network_IJCAI as Network
from tqdm import tqdm
import os
import argparse
import numpy as np
import cv2
import  matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def YCrCb2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def RGB2YCrCb(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def val_fusion_detection(model):
    vi_root = './test_imgs/Visible/'
    ir_root = './test_imgs/Infrared/'

    img_names = os.listdir(vi_root)

    for img_name in img_names:
        print("| Total test:" + str(len(img_names)), "| Now test:" + img_name)
        
        vi_img = cv2.imread(vi_root + img_name)
        vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2RGB)
        ir_img = cv2.imread(ir_root + img_name)
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)

        vi_img1 = transforms.ToTensor()(vi_img).cuda()
        ir_img1 = transforms.ToTensor()(ir_img).cuda()
        with torch.no_grad():
            fused, scores, classes, boxes = model.forward(ir_img1.unsqueeze_(dim=0),vi_img1.unsqueeze_(dim=0),None,None)
        fusion_img = fused.cpu().numpy()
        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()

        ## save fusion image
        fusion_img = fusion_img.transpose((0,2,3,1))
        fusion_img = fusion_img[0,:,:,:]
        fusion_img = np.uint8(255.0 * fusion_img)
        fusion_path = './test_result/fusion/'
        fusion_img = cv2.cvtColor(fusion_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fusion_path + img_name, fusion_img)
        
        ## compute detection result
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 15)]
        CLASSES_NAME = (
            "__background__", "Lamp", "Car", "Bus", "Motorcycle", "Truck", "People"
        )
        for i,box in enumerate(boxes):
            if scores[i]>=0.5:
                pos1 = (int(box[0]),int(box[1]))  
                pos2 = (int(box[2]),int(box[3]))  
                b_color = colors[int(classes[i])]  
                b_color = list(b_color)
                b_color  = [ele * 255 for ele in b_color]
                
                text_color = (255, 255, 255) 
                info = CLASSES_NAME[int(classes[i])] + ' %.3f'%(scores[i]) 
                
                cv2.rectangle(fusion_img,pos1,pos2,b_color,2)
                ## text setting
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size, _ = cv2.getTextSize(info, font, font_scale, thickness)
                ## Draw a rectangle as the background of the text
                padding = 3
                rect_x, rect_y = pos1
                rect_w, rect_h = text_size[0], text_size[1]

                cv2.rectangle(fusion_img, (rect_x, rect_y - rect_h - padding), (rect_x + rect_w + padding, rect_y), b_color, -1)
                cv2.putText(fusion_img, info, (rect_x + padding, rect_y - padding), font, font_scale, text_color, thickness, cv2.LINE_AA)

            save_path = './test_result/detection/'
            img = cv2.cvtColor(fusion_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path+img_name, fusion_img)

if __name__ == '__main__':
    '''
    test fusion and detection
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()

    print('| Testing fusion model on GPU #%d with pytorch' % (args.gpu))
    class Config():
        #backbone
        pretrained=False
        freeze_stage_1=True
        freeze_bn=True

        #fpn
        fpn_out_channels=256
        use_p5=True
        
        #head
        class_num=80
        use_GN_head=True
        prior=0.01
        add_centerness=True
        cnt_on_reg=False

        #training
        strides=[8,16,32,64,128]
        limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

        #inference
        score_threshold=0.3
        nms_iou_threshold=0.4
        max_detection_boxes_num=300
    model = Network(None, mode='inference', config=Config).cuda()
    model.load_state_dict(torch.load("./checkpoint/model_meta_50.pth"),strict=False)
    print('| Model Load Done!')
    val_fusion_detection(model)
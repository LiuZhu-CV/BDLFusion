import torchvision
from torch import nn
from torchvision.models import vgg16

from model.config import DefaultConfig
import torch
from dataset.VOC_dataset_Fusion_ import VOCDataset as VOCDataset
import math, time
from dataset.augment import Transforms, Transforms_Fusion
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse

from model.fcos_fusion import Network_IJCAI
from architect_average2 import Architect as Architect1
from architect_object_detection import Architect as Architect2

from model.loss import Total_fusion_loss, Total_fusion_loss2, Fusionloss2, Fusionloss4, Fusionloss_grad2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=400, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)



transform = Transforms_Fusion()
config = DefaultConfig()
def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
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
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
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
parser = argparse.ArgumentParser("ruas")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.000000001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

if __name__ == '__main__':
    train_dataset = VOCDataset(root_dir='./M3FD_Detection/', resize_size=[512, 360],
                               split='train', use_difficult=False, is_train=True, augment=transform)
    val_dataset = VOCDataset(root_dir='./M3FD_Detection/', resize_size=[480, 360],
                               split='val', use_difficult=False, is_train=True, augment=transform)
    fusion_loss = Fusionloss_grad2().cuda()
    model = Network_IJCAI(f_loss=fusion_loss, mode="training", config=config)
    model = model.cuda()
    model.load_state_dict(torch.load('./model_ijcai_fusion_add_1.pth'),strict=False)

    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    #WARMPUP_STEPS_RATIO = 0.12
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=0)
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=0)
    print("total_images : {}".format(len(train_dataset)))
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMPUP_STEPS = 1000

    GLOBAL_STEPS = 1
    LR_INIT = 2e-4
    LR_END = 2e-5
    architect = Architect1(model, args)
    # architect1 = Architect2(model, args)

    optimizer_dis = torch.optim.SGD(model.discriminator.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)
    optimizer_detection = torch.optim.SGD(model.denoise_net.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

    model.train()
    f_loss_txt = open('loss_bilvel.txt','w')

    for epoch in range(EPOCHS):
        for epoch_step, data in enumerate(train_loader):

            batch_imgs_ir, batch_imgs_vis,mask, batch_boxes, batch_classes = data

            batch_imgs_ir_old,batch_imgs_vis_old,mask_old, batch_boxes_old,batch_classes_old= next(iter(val_dataset))

            batch_imgs_ir = batch_imgs_ir.cuda()
            batch_imgs_vis = batch_imgs_vis.cuda()
            mask = mask.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()
            if GLOBAL_STEPS < WARMPUP_STEPS:
               lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
               for param in optimizer_dis.param_groups:
                   param['lr'] = lr
            if GLOBAL_STEPS == 30001:
               lr = LR_INIT * 0.1
               for param in optimizer_dis.param_groups:
                   param['lr'] = lr
            if GLOBAL_STEPS == 40000:
               lr = LR_INIT * 0.01
               for param in optimizer_dis.param_groups:
                  param['lr'] = lr


            if GLOBAL_STEPS < WARMPUP_STEPS:
               lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
               for param in optimizer_detection.param_groups:
                   param['lr'] = lr
            if GLOBAL_STEPS == 30001:
               lr = LR_INIT * 0.1
               for param in optimizer_detection.param_groups:
                   param['lr'] = lr
            if GLOBAL_STEPS == 40000:
               lr = LR_INIT * 0.01
               for param in optimizer_detection.param_groups:
                  param['lr'] = lr

            start_time = time.time()
            if (GLOBAL_STEPS+1) % 10 == 0 and epoch <= 40:
                batch_imgs_ir_old = batch_imgs_ir_old.cuda()
                batch_imgs_vis_old = batch_imgs_vis_old.cuda()
                mask_old = mask_old.cuda()
                batch_boxes_old = batch_boxes_old.cuda()
                batch_classes_old = batch_classes_old.cuda()
                architect.step(batch_imgs_ir, batch_imgs_vis,mask, batch_boxes, batch_classes,batch_imgs_ir_old,
                               batch_imgs_vis_old,mask_old, batch_boxes_old, batch_classes_old,param['lr'], unrolled=True, lr_new=param['lr'])

            if (GLOBAL_STEPS+1) % 10 == 0 and epoch > 40:
                batch_imgs_ir_old = batch_imgs_ir_old.cuda()
                batch_imgs_vis_old = batch_imgs_vis_old.cuda()
                mask_old = mask_old.cuda()
                batch_boxes_old = batch_boxes_old.cuda()
                batch_classes_old = batch_classes_old.cuda()
                architect.step(batch_imgs_ir, batch_imgs_vis,mask, batch_boxes, batch_classes,
                               batch_imgs_ir_old,batch_imgs_vis_old,mask_old, batch_boxes_old, batch_classes_old,
                               param['lr'], unrolled=True, lr_new=param['lr'])

            optimizer_dis.zero_grad()
            loss = model._fusion_loss(batch_imgs_ir,batch_imgs_vis,mask)
            loss.mean().backward()
            optimizer_dis.step()

            optimizer_detection.zero_grad()
            loss = model._detection_loss(batch_imgs_ir, batch_imgs_vis,batch_boxes,batch_classes)
            loss.mean().backward()
            optimizer_detection.step()




            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            print(
                "global_steps:%d epoch:%d steps:%d/%d  cost_time:%dms lr=%.4e total_loss:%.4f" % \
                (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, cost_time, lr, loss.mean()))
            GLOBAL_STEPS += 1
            f_loss_txt.write('epoch:' + str(GLOBAL_STEPS) + ' loss:' + str(loss.mean()))
            if (GLOBAL_STEPS+1)% 50 ==0:
                with torch.no_grad():
                    fused,losses = model(batch_imgs_ir,batch_imgs_vis, batch_boxes, batch_classes)
                torchvision.utils.save_image(batch_imgs_ir[:2], 'input_ir_1.png')
                torchvision.utils.save_image(batch_imgs_vis[:2], 'input_vis_1.png')
                torchvision.utils.save_image(fused[:2], 'output_1.png')
                torchvision.utils.save_image(mask[:2], 'mask_1.png')
        if (epoch+1)%1==0:
            torch.save(model.state_dict(),
                       "./checkpoint/model_meta1_{}.pth".format(epoch + 1))















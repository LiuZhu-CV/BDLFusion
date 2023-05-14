from .head import ClsCntRegHead
from .fpn_neck import FPN
from .backbone.resnet import resnet50
import torch.nn as nn
from .loss import GenTargets, LOSS, coords_fmap2orig
import torch
from .config import DefaultConfig
import torch.nn.functional as F
import functools

class FCOS(nn.Module):
    
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.backbone=resnet50(pretrained=config.pretrained,if_include_top=False)
        self.fpn=FPN(config.fpn_out_channels,use_p5=config.use_p5)
        self.head=ClsCntRegHead(config.fpn_out_channels,config.class_num,
                                config.use_GN_head,config.cnt_on_reg,config.prior)
        self.config=config
    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False
        # if self.config.freeze_bn:
        #     self.apply(freeze_bn)
        #     print("INFO===>success frozen BN")
        # if self.config.freeze_stage_1:
        #     self.backbone.freeze_stages(1)
        #     print("INFO===>success frozen backbone stage1")

    def forward(self,x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3,C4,C5=self.backbone(x)
        all_P=self.fpn([C3,C4,C5])
        cls_logits,cnt_logits,reg_preds=self.head(all_P)
        return [cls_logits,cnt_logits,reg_preds]

class DetectHead(nn.Module):
    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides,config=None):
        super().__init__()
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w] 
        '''
        cls_logits,coords=self._reshape_cat_out(inputs[0],self.strides)#[batch_size,sum(_h*_w),class_num]
        cnt_logits,_=self._reshape_cat_out(inputs[1],self.strides)#[batch_size,sum(_h*_w),1]
        reg_preds,_=self._reshape_cat_out(inputs[2],self.strides)#[batch_size,sum(_h*_w),4]

        cls_preds=cls_logits.sigmoid_()
        cnt_preds=cnt_logits.sigmoid_()

        coords =coords.cuda() if torch.cuda.is_available() else coords

        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)#[batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            cls_scores = torch.sqrt(cls_scores*(cnt_preds.squeeze(dim=-1)))#[batch_size,sum(_h*_w)]
        cls_classes=cls_classes+1#[batch_size,sum(_h*_w)]

        boxes=self._coords2boxes(coords,reg_preds)#[batch_size,sum(_h*_w),4]

        #select topk
        max_num=min(self.max_detection_boxes_num,cls_scores.shape[-1])
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]#[batch_size,max_num]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])#[max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])#[max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])#[max_num,4]
        cls_scores_topk=torch.stack(_cls_scores,dim=0)#[batch_size,max_num]
        cls_classes_topk=torch.stack(_cls_classes,dim=0)#[batch_size,max_num]
        boxes_topk=torch.stack(_boxes,dim=0)#[batch_size,max_num,4]
        assert boxes_topk.shape[-1]==4
        return self._post_process([cls_scores_topk,cls_classes_topk,boxes_topk])

    def _post_process(self,preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask=cls_scores_topk[batch]>=self.score_threshold
            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
            _boxes_b=boxes_topk[batch][mask]#[?,4]
            nms_ind=self.batched_nms(_boxes_b,_cls_scores_b,_cls_classes_b,self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores,classes,boxes= torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)
        
        return scores,classes,boxes
    
    @staticmethod
    def box_nms(boxes,scores,thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.sort(0,descending=True)[1]
        keep=[]
        while order.numel()>0:
            if order.numel()==1:
                i=order.item()
                keep.append(i)
                break
            else:
                i=order[0].item()
                keep.append(i)
            
            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            idx=(iou<=thr).nonzero().squeeze()
            if idx.numel()==0:
                break
            order=order[idx+1]
        return torch.LongTensor(keep)

    def batched_nms(self,boxes, scores, idxs, iou_threshold):
        
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self,coords,offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1=coords[None,:,:]-offsets[...,:2]
        x2y2=coords[None,:,:]+offsets[...,2:]#[batch_size,sum(_h*_w),2]
        boxes=torch.cat([x1y1,x2y2],dim=-1)#[batch_size,sum(_h*_w),4]
        return boxes


    def _reshape_cat_out(self,inputs,strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            pred=pred.permute(0,2,3,1)
            coord=coords_fmap2orig(pred,stride).to(device=pred.device)
            pred=torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes
     
class DRDB(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x + F.relu(x6)
        return out

class Fusion_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.DRDB1 = DRDB()
        self.DRDB2 = DRDB()
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.PReLU()
    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        x1 = self.conv1(torch.cat([ir,vis],dim=1))
        x1 = self.relu(x1)
        f1 = self.DRDB1(x1)
        f2 = self.DRDB2(f1)
        f_final = self.relu(self.conv2(f2))
        return f_final

class Discriminator(nn.Module):
    """
    Use to discriminate fused images and source images.
    """

    def __init__(self, dim=32):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, dim, 3,padding=1),
                nn.ReLU(),
            ),

        )
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv3 = nn.Conv2d(dim, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.conv_1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, dim, 3, padding=1),
                nn.ReLU(),
            ),

        )
        self.conv_2 = nn.Conv2d(dim , dim, 3, padding=1)
        self.conv_3 = nn.Conv2d(dim, 1, 3, padding=1)

        # self.flatten = nn.Flatten()
        # self.linear = nn.Linear((size[0] // 8) * (size[1] // 8) * 128, 1)

    def forward(self, fused):
        feature = self.conv(fused)
        # x = self.flatten(x)
        x_ir = self.relu(self.conv2(feature))
        x_ir = self.conv3(x_ir)
        feature1 = self.conv_1(fused)
        # x = self.flatten(x)
        x_vis = self.relu(self.conv_2(feature1))
        x_vis = self.conv_3(x_vis)
        return x_ir, x_vis

class FCOSDetector(nn.Module):
    def __init__(self, mode="training", config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.mode = mode
        self.fcos_body = FCOS(config=config)
        if mode == "training":
            self.target_layer = GenTargets(strides=config.strides, limit_range=config.limit_range)
            self.loss_layer = LOSS()
        elif mode == "inference":
            self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                             config.max_detection_boxes_num, config.strides, config)
            self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        '''
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if self.mode == "training":
            batch_imgs, batch_boxes, batch_classes = inputs
            out = self.fcos_body(batch_imgs)
            targets = self.target_layer([out, batch_boxes, batch_classes])
            losses = self.loss_layer([out, targets])
            return losses
        elif self.mode == "inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs = inputs
            out = self.fcos_body(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes

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

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class Fusion_Network_IJ(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=64)
        self.DRDB2 = DRDB(in_ch=64)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        x1 = self.conv1(torch.cat([ir,vis],dim=1))
        x1 = self.relu(x1)
        f1 = self.DRDB1(x1)
        f2 = self.DRDB2(f1)
        f_final = self.relu(self.conv2(f2))
        f_final = self.relu(self.conv21(f_final))
        # ones = torch.ones_like(f_final)
        # zeros = torch.zeros_like(f_final)
        # f_final = torch.where(f_final > ones, ones, f_final)
        # f_final = torch.where(f_final < zeros, zeros, f_final)
        # # new encode
        # f_final = (f_final - torch.min(f_final)) / (
        #         torch.max(f_final) - torch.min(f_final)
        # )
        return f_final

class Network_IJCAI(nn.Module):

    def __init__(self, f_loss, mode="training", config=None):
        super(Network_IJCAI, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self._criterion_gan = GANLoss('wgangp')
        self.enhance_net = Fusion_Network_IJ()
        self.denoise_net = FCOSDetector(mode,config)
        self.discriminator = PixelDiscriminator(1)
        self.mode = mode
    def forward(self, ir, vis,batch_boxes, batch_calsses):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        # ir_mask, vis_mask = self.discriminator(fused)
        ##Fixed some c
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        # new encode
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        ###Add the all types
        if self.mode =="training":
            losses = self.denoise_net([fused_seg,batch_boxes,batch_calsses])
            return fused,losses
        else:
            scores, classes, boxes = self.denoise_net(fused_seg)
            return fused_seg, scores, classes, boxes
        # u_list.append(u_d)
        # return fused, output
    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        # new encode
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        # x_ir,x_vis = self.discriminator(fused)
        return fused, fused,fused
    def forward_fusion2(self, ir, vis):
        fused = self.enhance_net.forward(ir, vis)
        return fused, fused,fused
    def forward_object(self, ir, vis,batch_boxes, batch_calsses):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        # ir_mask, vis_mask = self.discriminator(fused)
        ##Fixed some c
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        # new encode
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        # x_ir,x_vis = self.discriminator(fused)
        ###Add the all types
        if self.mode =="training":
            losses = self.denoise_net([fused_seg,batch_boxes,batch_calsses])
            return fused, losses
        else:
            scores, classes, boxes = self.denoise_net(fused_seg)
            return fused_seg, scores, classes, boxes
    def forward_object2(self, fused_seg,batch_boxes, batch_calsses):

        # x_ir,x_vis = self.discriminator(fused)
        ###Add the all types
        if self.mode =="training":
            losses = self.denoise_net([fused_seg,batch_boxes,batch_calsses])
            return fused, losses
        else:
            scores, classes, boxes = self.denoise_net(fused_seg)
            return fused_seg, scores, classes, boxes
    def _loss(self, ir, vis, mask, batch_boxes, batch_calsses):
        fused_img, losses = self(ir,vis,batch_boxes, batch_calsses)
        vis = RGB2YCrCb(vis)
        mask_ = mask[:, :1, :, :]
        mask = mask_ * ir[:, :1, :, :] + (1 - mask_) * vis[:, :1, :, :]
        enhance_loss = self._criterion(ir, vis, mask, fused_img)
        denoise_loss = losses[-1]

        pred_fake = self.discriminator(fused_img)
        loss_D_fake = self._criterion_gan(pred_fake, True)
        loss_D = (loss_D_fake) * 0.5

        # vis = RGB2YCrCb(vis)
        # mask_loss = self._criterion_lower(x_ir,x_vis,mask)
        return enhance_loss + denoise_loss + loss_D*0.1

    def _enhcence_loss(self,ir,vis,mask):
        fused_img =self.enhance_net(ir,vis)

        vis = RGB2YCrCb(vis)
        mask_ = mask[:, :1, :, :]
        mask = mask_ * ir[:, :1, :, :] + (1 - mask_) * vis[:, :1, :, :]
        enhance_loss = self._criterion(ir,vis, mask, fused_img)
        return enhance_loss

    def _denoise_loss(self, input1, input2, target):
        fused_img, seg1, seg2 = self(input1, input2)
        denoise_loss = self._denoise_criterion(seg1, target) + 0.2 * self._denoise_criterion(seg2, target)
        return denoise_loss

    def _fusion_loss(self, ir, vis, mask):
        fused_img, x_ir, x_vis = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        mask_ = mask[:, :1, :, :]
        mask = mask_ * ir[:, :1, :, :] + (1 - mask_) * vis[:, :1, :, :]
        # import numpy as np
        # print('shape',np.shape(mask))
        # print('shape',np.shape(mask))
        # print('shape',np.shape(mask))
        # print('shape',np.shape(mask))

        enhance_loss = self._criterion(ir, vis, mask, fused_img)

        pred_fake = self.discriminator(fused_img)
        loss_D_fake = self._criterion_gan(pred_fake, False)
        # Real
        # real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.discriminator(mask)
        loss_D_real = self._criterion_gan(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        # vis = RGB2YCrCb(vis)
        # mask_loss = self._criterion_lower(x_ir, x_vis, mask)
        return  enhance_loss + loss_D*0.5

    def _fusion_loss_upper(self, ir, vis, mask):
        fused_img, x_ir, x_vis = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        mask_ = mask[:, :1, :, :]
        mask = mask_ * ir[:, :1, :, :] + (1 - mask_) * vis[:, :1, :, :]
        enhance_loss = self._criterion(ir, vis, mask, fused_img)

        pred_fake = self.discriminator(fused_img)
        loss_D_fake = self._criterion_gan(pred_fake, True)
        # Real
        return enhance_loss + loss_D_fake * 0.5

    def _detection_loss(self, ir, vis, batch_boxes, batch_calsses):
        fused_img, losses = self.forward_object(ir, vis, batch_boxes, batch_calsses)
        pred_fake = self.discriminator(fused_img)
        loss_D_fake = self._criterion_gan(pred_fake, True)
        denoise_loss = losses[-1]
        return  denoise_loss+ loss_D_fake * 0.01

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()





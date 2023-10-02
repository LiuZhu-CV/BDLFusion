import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def _concat2(xs,moment):
  xs_new = []
  for m,x in zip(moment,xs):
    if x is not None:
      xs_new.append(x.view(-1))
    else:
      xs_new.append(torch.zeros_like(m).view(-1))
  return torch.cat(xs_new)

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    # self.optimizer = torch.optim.Adam(self.model.enhance_net.parameters(),
    #     lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    self.optimizer = torch.optim.SGD(self.model.enhance_net.parameters(),
        lr=args.arch_learning_rate, weight_decay=args.arch_weight_decay)
  def _compute_unrolled_model(self, input1, input2, target, eta, network_optimizer):
    loss = self.model._loss( input1, input2, target)
    theta = _concat(self.model.denoise_net.parameters()).data

    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.denoise_net.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    # print('models...............')
    # print(self.model)
    # print(theta.shape)
    # Why
    dtheta = _concat2(torch.autograd.grad(loss, self.model.denoise_net.parameters(),allow_unused=True),self.model.parameters()).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self,  batch_imgs_ir,batch_imgs_vis,mask, batch_boxes, batch_classes,imgs_ir,image_vis, mask_, boxes, classes, eta, unrolled, lr_new =1e-4):
        self.optimizer.param_groups[0]['lr'] = lr_new
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled_all( batch_imgs_ir,batch_imgs_vis,mask, batch_boxes, batch_classes, imgs_ir,image_vis, mask_, boxes, classes, eta)
        else:
            self._backward_step( batch_imgs_ir,batch_imgs_vis,mask, batch_boxes, batch_classes)
        self.optimizer.step()

  def _backward_step(self,  batch_imgs_ir,batch_imgs_vis,mask, batch_boxes, batch_classes):
        loss_1= self.model._loss( batch_imgs_ir,batch_imgs_vis,mask, batch_boxes, batch_classes)
        loss_1.backward()

  def _backward_step_unrolled_all(self, batch_imgs_ir, batch_imgs_vis, mask, batch_boxes, batch_classes, imgs_ir, image_vis,
                              mask_, boxes, classes, eta):
    dalpha1 = self._backward_step_unrolled(batch_imgs_ir, batch_imgs_vis, mask, batch_boxes, batch_classes, imgs_ir, image_vis,
                              mask_, boxes, classes, eta)
    dalpha2 = self._backward_step_unrolled2(batch_imgs_ir, batch_imgs_vis, mask, batch_boxes, batch_classes, imgs_ir,
                                           image_vis,
                                           mask_, boxes, classes, eta)
    batch_weight = F.softmax(torch.randn(2), dim=-1).cuda()

    for v, g,g2 in zip(self.model.enhance_net.parameters(), dalpha1,dalpha2):
      if v.grad is None:
        v.grad = Variable((g.data*batch_weight[0]*0.5+g2.data*batch_weight[1]*0.5))
      else:
        v.grad.data.copy_((g.data*batch_weight[0]*0.5+g2.data*batch_weight[1]*0.5))

  def _backward_step_unrolled(self, batch_imgs_ir,batch_imgs_vis,mask, batch_boxes, batch_classes,imgs_ir,image_vis, mask_, boxes, classes,  eta):

    ### fusion_part
    unrolled_loss = self.model._fusion_loss(imgs_ir,image_vis, mask_)
    # upper_loss
    # lower_loss
    unrolled_loss.backward()
    dalpha = [v.grad for v in self.model.enhance_net.parameters()]
    vector = []
    for v in self.model.discriminator.parameters():
      if v.grad is not None:
        vector.append(v.grad.data)
      else:
        vector.append(torch.zeros_like(v))

    # lower_loss = self.model._loss(input_train, target_train, lamda, latency)
    lower_loss_ = self.model._fusion_loss(batch_imgs_ir,batch_imgs_vis,mask)

    # dFy = torch.autograd.grad(upper_loss, unrolled_model.parameters(),allow_unused=True)

    dfy = torch.autograd.grad(lower_loss_, self.model.discriminator.parameters(), allow_unused=True)
    gfyfy = 0
    gFyfy = 0
    for f, F in zip(dfy, vector):
      if f is  None:
        f = torch.zeros_like(F)
      gfyfy = gfyfy + torch.sum(f * f)
      gFyfy = gFyfy + torch.sum(F * f)

    lower_loss_2 = self.model._fusion_loss_upper(batch_imgs_ir,batch_imgs_vis,mask)
    GN_loss = -gFyfy.detach() / gfyfy.detach() * lower_loss_2
    implicit_grads1 = torch.autograd.grad(GN_loss, self.model.enhance_net.parameters(), allow_unused=True)
    for g, ig in zip(dalpha, implicit_grads1):
      if ig is None:
        ig = torch.zeros_like(g)
      g.data.sub_(eta, ig.data)
    return dalpha

  def _backward_step_unrolled2(self, batch_imgs_ir, batch_imgs_vis, mask, batch_boxes, batch_classes, imgs_ir,
                                image_vis, mask_, boxes, classes, eta):

      ### fusion_part
      unrolled_loss = self.model._detection_loss(imgs_ir, image_vis, boxes, classes)
      # upper_loss
      # lower_loss
      unrolled_loss.backward()
      dalpha = [v.grad for v in self.model.enhance_net.parameters()]
      vector = []
      for v in self.model.denoise_net.parameters():
        if v.grad is not None:
          vector.append(v.grad.data)
        else:
          vector.append(torch.zeros_like(v))

      # lower_loss = self.model._loss(input_train, target_train, lamda, latency)
      lower_loss_ = self.model._detection_loss(batch_imgs_ir, batch_imgs_vis, batch_boxes, batch_classes)
      dfy = torch.autograd.grad(lower_loss_, self.model.denoise_net.parameters(), allow_unused=True)
      gfyfy = 0
      gFyfy = 0
      for f, F in zip(dfy, vector):
        if f is None:
          f = torch.zeros_like(F)
        gfyfy = gfyfy + torch.sum(f * f)
        gFyfy = gFyfy + torch.sum(F * f)

      lower_loss_2 = self.model._detection_loss(batch_imgs_ir, batch_imgs_vis, batch_boxes, batch_classes)
      GN_loss = -gFyfy.detach() / gfyfy.detach() * lower_loss_2
      implicit_grads1 = torch.autograd.grad(GN_loss, self.model.enhance_net.parameters(), allow_unused=True)
      for g, ig in zip(dalpha, implicit_grads1):
        if ig is None:
          ig = torch.zeros_like(g)
        g.data.sub_(eta, ig.data)
      return dalpha


  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.denoise_net.state_dict()

    params, offset = {}, 0
    for k, v in self.model.denoise_net.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.denoide_net.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.denoise_net.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target,0,100)
    grads_p = torch.autograd.grad(loss, self.model.enhance_net.parameters())

    for p, v in zip(self.model.denoise_net.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target,0,100)
    grads_n = torch.autograd.grad(loss, self.model.encoder_net.parameters())

    for p, v in zip(self.model.denoide_net.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


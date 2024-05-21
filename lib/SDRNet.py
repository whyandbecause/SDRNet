import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from lib.pvtv2 import pvt_v2_b2
from einops import rearrange
import timm
import numbers

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x

class DB(nn.Module):
  
  def __init__(self, c1, c2, e=1, k=(3, 7, 9)):
    super(DB, self).__init__()
    c_ = int(2 * c2 * e)  # hidden channels
    
    #global branch
    self.cv1 = nn.Conv2d(c1, c_, 1, 1)
    self.cv2 = nn.Conv2d(c1, c_, 1, 1)
    self.cv3 = nn.Conv2d(c_, c_, 3, 1,1)
    self.cv4 = nn.Conv2d(c_, c_, 1, 1)
    self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
    self.cv5 = nn.Conv2d(4 * c_, c_, 1, 1)
    self.cv6 = nn.Conv2d(c_, c_, 3, 1,1)
    self.cv7 = nn.Conv2d(2 * c_, c2, 1, 1)
    
    #details branch
    self.cv2_ = nn.Conv2d(c1, c_, 1, 1)
    self.cv8 = nn.Conv2d(4 * c_, c_, 1, 1)
    self.cv9 = nn.Conv2d(c_, c_, 3, 1,1)
    self.cv10 = nn.Conv2d(2 * c_, c2, 1, 1)
    
    #self.de = DetailEnhance(c2)
    
  def forward(self, x):
    x1 = self.cv4(self.cv3(self.cv1(x)))
    y1 = self.cv2(x)
    details = [y1-x1]
    glfs = [x1]
    for m in self.m:
      x_ = m(x1) 
      details += [y1 - x_]
      glfs +=[x_]
    y2 = self.cv6(self.cv5(torch.cat(glfs, 1)))
    glf = self.cv7(torch.cat((y1, y2), dim=1))
    
    y3 = self.cv2_(x)
    y4 = self.cv9(self.cv8(torch.cat(details, 1)))
    detail = self.cv10(torch.cat((y3, y4), dim=1))
    #detail = self.de(detail)
    return [glf, detail]

class DGF(nn.Module): 
  def __init__(self, r, eps):
      super(DGF, self).__init__()
      self.r = r
      self.eps = eps
      self.mean = nn.AvgPool2d(r, 1, r//2)
      
  def forward(self, guide, img): 
    mean_G = self.mean(guide)
    mean_I = self.mean(img)
    corr_IG = self.mean(img*guide)
    corr_GG = self.mean(guide*guide)
    cov_IG = corr_IG - mean_G * mean_I
    var_G = corr_GG - mean_G * mean_G
    a = cov_IG / (var_G + self.eps)
    b = mean_I - a * mean_G
    mean_a = self.mean(a)
    mean_b = self.mean(b)
    out = mean_a*guide + mean_b
    return out 
      
class Gaussblur(nn.Module):
  def __init__(self, c):
    super(Gaussblur, self).__init__()
    self.kernel = [[0.03797616, 0.044863533, 0.03797616],
               [0.044863533, 0.053, 0.044863533],
               [0.03797616, 0.044863533, 0.03797616]]
    self.kernel = torch.FloatTensor(self.kernel).expand(c,c,3,3).cuda()
    #self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
    
  def forward(self, tensor_image):

    return F.conv2d(tensor_image,self.kernel,padding=1)

class LoG(nn.Module):
  def __init__(self, c):
    super(LoG, self).__init__()
    self.kernel = [[0,1,1,2,2,2,1,1,0],
                        [1,2,4,5,5,5,4,2,1],
                        [1,4,5,3,0,3,5,4,1],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [2,5,0,-24,-40,-24,0,5,2],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [1,4,5,3,0,3,4,4,1],
                        [1,2,4,5,5,5,4,2,1],
                        [0,1,1,2,2,2,1,1,0]]
    self.kernel = torch.FloatTensor(self.kernel).expand(c,c,9,9).cuda()
    #self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
    
  def forward(self, x):

    return F.conv2d(x,self.kernel,padding=4)

class DetailEnhance(nn.Module):
    def __init__(self, channel):
        super(DetailEnhance, self).__init__()
        
        self.b1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1)#ke = k + (k ? 1)(r ? 1)  p = (ke -1)//2
        self.b2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.b4 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=4, dilation=4)
    def forward(self, x):
      
      x1 = self.b1(x)
      x2 = self.b2(x)
      x3 = self.b4(x)
      res1 = x - x1
      res2 = x1 - x2
      res3 = x2 - x3
      
      return  x + 0.5*res1 + 0.5*res2 + 0.25*res3 #(1-0.5*sgn(res1)*res1) 0.5 0.5 0.25
              
class SHFB(nn.Module):

    def __init__(self, channel):
        super(SHFB, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fdb = DB(channel, channel)
        
        self.dfg1 = DGF(7, 1e-7)#sigma>eps get details
        self.dfg2 = DGF(15, 0.3)#sigma<eps,get structure
        self.log = LoG(channel)
        self.gaussblur = Gaussblur(channel)
        self.sig = nn.Sigmoid()
        self.conv_f= BasicConv2d(2*channel, channel, 3, padding=1)
        
        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3, x4):

        x1_1 = x1
        #print(x1.shape,x2.shape, x3.shape, x4.shape)
        x1_sd, x2_sd = self.fdb(x1), self.fdb(x2)#de the feature into structure and details
        x1 = self.upsample(x1)
        w_d1, w_d2 = self.sig(self.gaussblur(abs(self.log(x1)))), self.sig(self.gaussblur(abs(self.log(x2))))#get init weight from init feature
        x_12 = self.conv_f(torch.cat((x1, x2),1))#combine two level feature
        w11 = self.dfg1(x_12, w_d1)#get details fusion weight
        w12 = self.dfg2(x_12, 1-w_d1)#get structure fusion weight
        w21 = self.dfg1(x_12, w_d2)
        w22 = self.dfg2(x_12, 1-w_d2)
        x2_1 = (self.upsample(x1_sd[0]) * w12 + x2_sd[0] * w22) + (self.upsample(x1_sd[1]) * w11 + x2_sd[1] * w21)#fusion by details and structure
        x2_o = x2_1
        x2_1sd, x3_sd = self.fdb(x2_1), self.fdb(x3)
        x2_1 = self.upsample(x2_1)
        w2_1d, w_d3 = self.sig(self.gaussblur(abs(self.log(x2_1)))), self.sig(self.gaussblur(abs(self.log(x3))))
        x_23 = self.conv_f(torch.cat((x2_1, x3),1))
        w211 = self.dfg1(x_23, w2_1d)
        w212 = self.dfg2(x_23, 1-w2_1d)
        w31 = self.dfg1(x_23, w_d3)
        w32 = self.dfg2(x_23, 1-w_d3)
        x3_1 = (self.upsample(x2_1sd[0]) * w212 + x3_sd[0] * w32) + (self.upsample(x2_1sd[1]) * w211 + x3_sd[1] * w31)
        x3_o = x3_1
        #print(x3_1.shape, x4.shape)
        x3_1sd, x4_sd = self.fdb(x3_1), self.fdb(x4)
        x3_1 = self.upsample(x3_1)
        w3_1d, w_d4 = self.sig(self.gaussblur(abs(self.log(x3_1)))), self.sig(self.gaussblur(abs(self.log(x4))))
        x_34 = self.conv_f(torch.cat((x3_1, x4),1))
        w311 = self.dfg1(x_34, w3_1d)
        w312 = self.dfg2(x_34, 1-w3_1d)
        w41 = self.dfg1(x_34, w_d4)
        w42 = self.dfg2(x_34, 1-w_d4)
        x4_1 = (self.upsample(x3_1sd[0]) * w312 + x4_sd[0] * w42) + (self.upsample(x3_1sd[1]) * w311 + x4_sd[1] * w41)
        
        #x2_2 = torch.cat((x2_1, self.conv_upsample1(self.upsample(x1_1))), 1)
        x2_2 = torch.cat((x2_o, x1), 1)
        #print(x2_1.shape, x1.shape)
        x2_2 = self.conv_concat2(x2_2)#24
        
        #x3_2 = torch.cat((x3_1, self.conv_upsample2(self.upsample(x2_2))), 1)
        x3_2 = torch.cat((x3_o, self.upsample(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)#48
        #print(x3_2.shape, x4_1.shape)
        #x4_2 = torch.cat((x4_1, self.conv_upsample3(self.upsample(x3_2))), 1)
        x4_2 = torch.cat((x4_1,self.upsample(x3_2)), 1)
        x4_2 = self.conv_concat4(x4_2)
        x__ = self.conv4(x4_2)
        #x_s = self.conv5(x__)#96*96
        #x_d = self.conv6(x__)
        #print(x.shape)
        return  x__

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avgo = self.sigmoid(avg_out)
        maxo = self.sigmoid(max_out)
        x = self.sigmoid(x * avgo + x * maxo)
        return x

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
        #return y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, channel):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(channel, channel//4, 1))
        modules_body.append(nn.Conv2d(channel//4, channel//4, 3, padding=1))
        modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(nn.Conv2d(channel//4, channel, 1))

        self.CA = CALayer(channel, reduction=4)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class SAB(nn.Module):
    def __init__(self, channel):
        super(SAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(channel, channel//2, 1))
        modules_body.append(nn.Conv2d(channel//2, channel//2, 3, padding=1))
        modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(nn.Conv2d(channel//2, channel, 1))

        self.SA = SpatialAttention(7)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        att = self.SA(res)
        #res = res * att
        #return res + x
        return att
        
class Local(nn.Module):
    def __init__(self, channel):
        super(Local, self).__init__()
        self.ca = CAB(channel)
        self.sa = SAB(channel)
        self.conv = nn.Conv2d(channel*2, channel, 1, padding=0)
    def forward(self, x1, x2):
        x = self.conv(torch.cat((x1, x2),1))
        w1 = self.ca(x)
        w2 = self.sa(x)
        o = w1 * x1 + (1-w1) * x2 + w2 * x1 + (1-w2) * x2
        return o
        
class Interaction(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Interaction, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.compress = nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, bias=bias)
        
        self.norm = LayerNorm(dim)
        
        
    def forward(self, x):
        b,c,h,w = x.shape
        x = self.norm(x)
        
        
        q=self.qkv1conv(self.qkv_0(x))
       
        k=self.qkv2conv(self.qkv_1(x))
        v=self.qkv3conv(self.qkv_2(x))
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature#b head c (h w)   b head  (h w) c
        attn = attn.softmax(dim=-1)#b head c c
        out = (attn @ v) #b head c c    b head c (h w)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out) + x
        out = self.compress(out)
        return out
                
#RestoreFormer                
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.norm = LayerNorm(dim)
        
        
    def forward(self, xx, q__):
        b,c,h,w = xx.shape
        x = self.norm(xx)
        
        q_ = self.norm(q__)
        q=self.qkv1conv(self.qkv_0(q_))
       
        k=self.qkv2conv(self.qkv_1(x))
        v=self.qkv3conv(self.qkv_2(x))
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature#b head c (h w)   b head  (h w) c
        attn = attn.softmax(dim=-1)#b head c c
        out = (attn @ v) #b head c c    b head c (h w)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out) + q__
        
        return out
        
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
                  
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c -> b c h w',h=h,w=w) 
        return x
               
      
class DFusion(nn.Module):
    def __init__(self, in_c, out_c):#320 512 512===>320
        super(DFusion, self).__init__()
        
        
        self.conv_d1 = nn.Sequential(
                                    nn.Conv2d(out_c+in_c, out_c, 1, padding=0),
                                    BasicConv2d(out_c, out_c, 3, padding=1))
                                    
        self.conv_d2 = nn.Sequential(
                                    nn.Conv2d(in_c+in_c, out_c, 1, padding=0),
                                    BasicConv2d(out_c, out_c, 3, padding=1))
                                    
        self.conv_d3 = nn.Sequential(
                                    nn.Conv2d(out_c+in_c, out_c, 1, padding=0),
                                    BasicConv2d(out_c, out_c, 3, padding=1))
                                    
        self.conv_d4 = nn.Sequential(
                                    nn.Conv2d(out_c+in_c, out_c, 1, padding=0),
                                    BasicConv2d(out_c, out_c, 3, padding=1))
                                    
        self.conv_dist1 = BasicConv2d(out_c, out_c//4, 3, padding=1)
        self.conv_dist2 = BasicConv2d(out_c, out_c//4, 3, padding=1)
        self.conv_dist3 = BasicConv2d(out_c, out_c//4, 3, padding=1)
        self.conv_dist4 = BasicConv2d(out_c, out_c//4, 3, padding=1)
        self.conv_dist5 = BasicConv2d(out_c, out_c//4, 3, padding=1)
        self.conv_dist6 = BasicConv2d(out_c, out_c//4, 3, padding=1)
        
        self.conv_ds1 = nn.Conv2d(out_c//2, out_c//4, 1, padding=0)
        self.conv_ds2 = nn.Conv2d(out_c//2, out_c//4, 1, padding=0)
        
        self.rcab = CAB(out_c)                               
        
        
    def forward(self, d1, d2, s):#320 512 512===>320
    
      d1 = self.conv_d1(torch.cat((d1, s), dim=1))
      d2 = self.conv_d2(torch.cat((d2, s), dim=1))
      dist1 = self.conv_dist1(d1)
      dist2 = self.conv_dist2(d2)
      d1 = self.conv_d3(torch.cat((d1, s), dim=1))
      d2 = self.conv_d4(torch.cat((d2, s), dim=1))
      dist3 = self.conv_dist3(d1)
      dist4 = self.conv_dist4(d2)
      
      dist12 = self.conv_ds1(torch.cat((dist1, dist2), dim=1))
      dist34 = self.conv_ds2(torch.cat((dist3, dist4), dim=1))
      dist5  = self.conv_dist5(d1)
      dist6  = self.conv_dist6(d2)
      d12 = self.rcab(torch.cat((dist12, dist34, dist5, dist6), dim=1))
      
      return d12

class SFusion(nn.Module):
    def __init__(self, in_c, out_c):#320 512 320===>320
        super(SFusion, self).__init__()
        
        self.conv_s1 = nn.Conv2d(out_c, out_c*2, kernel_size=1, padding=0)
        self.conv_s2 = nn.Conv2d(in_c, out_c*2, kernel_size=1, padding=0)
        self.conv_sepc1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv_sepc2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv_comm = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv_alpha = nn.Sequential(
                                        nn.Conv2d(out_c, out_c//2, 1, padding=0),
                                        nn.Conv2d(out_c//2, 1, 3, padding=1))
        self.conv_beta = nn.Sequential(
                                        nn.Conv2d(out_c, out_c//2, 1, padding=0),
                                        nn.Conv2d(out_c//2, 1, 3, padding=1))
                                        
        self.relu = nn.ReLU(inplace=True)
        
        self.w = nn.Sequential(
            nn.Conv2d(out_c*2, out_c//2, 1, padding=0),
            nn.Conv2d(out_c//2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, s1, s2, d):#320 512 320===>320
    
      s1_cx2 = self.conv_s1(s1)
      s2_cx2 = self.conv_s2(s2)
      s1_spec, s1_comm = s1_cx2.chunk(2, 1)
      s2_spec, s2_comm = s2_cx2.chunk(2, 1)#320
      
      s1_spec = self.relu(self.conv_sepc1(s1_spec))
      s2_spec = self.relu(self.conv_sepc2(s2_spec))#320
      s12_common = self.relu(self.conv_comm(s1_comm+s2_comm))#320
      w12 = self.w(torch.cat((s1_spec, s2_spec), dim=1))
      s12_spec = s1_spec*w12 + s2_spec*(1-w12)
      s12 = s1_spec + s12_common
      
      alpha = self.conv_alpha(d)
      beta = self.conv_beta(d)
      s12 = s12 * alpha + beta
      
      return s12
 
class AFF(nn.Module):
   
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()
        self.conv_l = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv_g = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        x1 = self.conv_l(x)
        x2 = self.conv_g(x)
        xl = self.local_att(x1)
        xg = self.global_att(x2)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x1 * wei + 2 * x2 * (1 - wei)
        return xo
        
class Network(nn.Module):
    def __init__(self, channel=64):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        #self.encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        path = './pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        #self.Translayer1 = BasicConv2d(256, channel, 1)
        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(320, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)
        
        self.intra = Interaction(channel*4, 4, False)
        
        self.decom4 = DB(channel*8, channel*8)#, k=(3, 7, 9))
        self.decom3 = DB(channel*5, channel*5)#, k=(5, 9, 13))
        self.decom2 = DB(channel*2, channel*2)#, k=(7, 13, 17))
        self.decom1 = DB(channel, channel)#, k=(9, 17, 21))
        
        self.share_att4 = Attention(channel*8, 8, False)
        self.share_att3 = Attention(channel*5, 8, False)
        self.share_att2 = Attention(channel*2, 8, False)
        self.share_att1 = Attention(channel, 8, False)
        
        self.d_f34 = DFusion(channel*8, channel*5)
        self.d_f234 = DFusion(channel*5, channel*2)
        self.d_f1234 = DFusion(channel*2, channel)
        
        self.s_f34 = SFusion(channel*8, channel*5)
        self.s_f234 = SFusion(channel*5, channel*2)
        self.s_f1234 = SFusion(channel*2, channel)
        
        self.de4 = DetailEnhance(channel*8)
        self.de3 = DetailEnhance(channel*5)
        self.de2 = DetailEnhance(channel*2)
        self.de1 = DetailEnhance(channel)
        
        self.se4 = AFF(channel*8)
        self.se3 = AFF(channel*5)
        self.se2 = AFF(channel*2)
        self.se1 = AFF(channel)
        
        self.linearrd1 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearrd2 = nn.Conv2d(channel*2, 1, kernel_size=3, stride=1, padding=1)
        self.linearrd3 = nn.Conv2d(channel*5, 1, kernel_size=3, stride=1, padding=1)
        
        self.linearrs1 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearrs2 = nn.Conv2d(channel*2, 1, kernel_size=3, stride=1, padding=1)
        self.linearrs3 = nn.Conv2d(channel*5, 1, kernel_size=3, stride=1, padding=1)
        
        self.DID = SHFB(channel)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.convs_share1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.convs_share2 = nn.Conv2d(channel, channel*2, kernel_size=1, stride=1, padding=0)
        self.convs_share3 = nn.Conv2d(channel, channel*5, kernel_size=1, stride=1, padding=0)
        self.convs_share4 = nn.Conv2d(channel, channel*8, kernel_size=1, stride=1, padding=0)
        
        self.convs = nn.Conv2d(channel, 1, 1)
        self.convd = nn.Conv2d(channel, 1, 1)
        
    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x)
        #x0, x1, x2, x3, x4 = self.encoder(x)#[64, 192, 192][256, 96, 96][512, 48, 48])[1024, 24, 24][2048, 12, 12]
        #print(x0.shape,x1.shape,x2.shape,x3.shape,x4.shape)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
        #x1_t = self.Translayer1(x1)
        x2_t = self.Translayer2(x2)
        x3_t = self.Translayer3(x3)
        x4_t = self.Translayer4(x4)
        x_qkv = self.intra(torch.cat((x1, F.interpolate(x2_t, size=x1.size()[2:], mode='bilinear'), 
                                              F.interpolate(x3_t, size=x1.size()[2:], mode='bilinear'), 
                                              F.interpolate(x4_t, size=x1.size()[2:], mode='bilinear')),1))
        x_share= self.DID(x4_t, x3_t, x2_t, x1)
        x_share = x_qkv + x_share
        s_pre = F.interpolate(self.convs(x_share), size=image_shape, mode='bilinear')
        d_pre = F.interpolate(self.convd(x_share), size=image_shape, mode='bilinear')
        
        x_share1 = F.interpolate(self.convs_share1(x_share), size=x1.shape[2:], mode='bilinear')
        x_share2 = F.interpolate(self.convs_share2(x_share), size=x2.shape[2:], mode='bilinear')
        x_share3 = F.interpolate(self.convs_share3(x_share), size=x3.shape[2:], mode='bilinear')
        x_share4 = F.interpolate(self.convs_share4(x_share), size=x4.shape[2:], mode='bilinear')
        
        x4_s, x4_d = self.decom4(x4)#512
        x3_s, x3_d = self.decom3(x3)#320
        x2_s, x2_d = self.decom2(x2)#128
        x1_s, x1_d = self.decom1(x1)#64
        
        pp_xs, pp_xd = x1_s, x1_d###########
        
        s1_spec, d1_spec = self.share_att1(x_share1, x1_s), self.share_att1(x_share1, x1_d)
        s2_spec, d2_spec = self.share_att2(x_share2, x2_s), self.share_att2(x_share2, x2_d)
        s3_spec, d3_spec = self.share_att3(x_share3, x3_s), self.share_att3(x_share3, x3_d)
        s4_spec, d4_spec = self.share_att4(x_share4, x4_s), self.share_att4(x_share4, x4_d)
        
        x4_d = self.de4(x4_d)+d4_spec
        x3_d = self.de3(x3_d)+d3_spec
        x2_d = self.de2(x2_d)+d2_spec
        x1_d = self.de1(x1_d)+d1_spec
        
        x4_s = self.se4(x4_s)+s4_spec
        x3_s = self.se3(x3_s)+s3_spec
        x2_s = self.se2(x2_s)+s2_spec
        x1_s = self.se1(x1_s)+s1_spec
        
        x4_s = F.interpolate(x4_s, size=x3_d.shape[2:], mode='bilinear')
        x4_d = F.interpolate(x4_d, size=x3_d.shape[2:], mode='bilinear')
        d34 = self.d_f34(x3_d, x4_d, x4_s)#320 512 512===>320    x3
        s34 = self.s_f34(x3_s, x4_s, d34)#320 512 320===>320   x3
        
        s34_ = F.interpolate(s34, size=x2_d.shape[2:], mode='bilinear')
        d34_ = F.interpolate(d34, size=x2_d.shape[2:], mode='bilinear')
        d234 = self.d_f234(x2_d, d34_, s34_)#128 320  320--->128  x2
        s234 = self.s_f234(x2_s, s34_, d234)#128 320 128--->128   x2
        
        s234_ = F.interpolate(s234, size=x1_d.shape[2:], mode='bilinear')
        d234_ = F.interpolate(d234, size=x1_d.shape[2:], mode='bilinear')
        d1234 = self.d_f1234(x1_d, d234_, s234_)#64 128 128-->64  x1
        s1234 = self.s_f1234(x1_s, s234_, d1234)#64 128 64--->64  x1
        
        d_3 = self.linearrd3(d34)
        d3 = F.interpolate(d_3, size=d234.shape[2:], mode='bilinear')
        d_2 = self.linearrd2(d234) + d3
        d2 = F.interpolate(d_2, size=d1234.shape[2:], mode='bilinear')
        d_1 = self.linearrd1(d1234) + d2
        
        s_3 = self.linearrs3(s34)
        s3 = F.interpolate(s_3, size=s234.shape[2:], mode='bilinear')
        s_2 = self.linearrs2(s234) + s3
        s2 = F.interpolate(s_2, size=s1234.shape[2:], mode='bilinear')
        s_1 = self.linearrs1(s1234) + s2
        #print(s_1.shape, s_2.shape, s_3.shape)

        s_1 = F.interpolate(s_1, size=image_shape, mode='bilinear')
        s_2 = F.interpolate(s_2, size=image_shape, mode='bilinear')
        s_3 = F.interpolate(s_3, size=image_shape, mode='bilinear')
        
        
        d_1 = F.interpolate(d_1, size=image_shape, mode='bilinear')
        d_2 = F.interpolate(d_2, size=image_shape, mode='bilinear')
        d_3 = F.interpolate(d_3, size=image_shape, mode='bilinear')
        

        return s_1, s_2, s_3, s_pre, d_1, d_2, d_3, d_pre
        #return s_1, d_1







if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Decoder(4,4).cuda()
    net.eval()

    dump_x1 = torch.randn(2, 4, 2, 2).cuda()
    dump_x2 = torch.randn(2, 4, 2, 2).cuda()
    dump_x3 = torch.randn(2, 4, 2, 2).cuda()
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        torch.cuda.synchronize()
        start = time()
        y = net(dump_x1,dump_x2,dump_x3)
        torch.cuda.synchronize()
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        # print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    # print(np.mean(frame_rate))
    # print(y)
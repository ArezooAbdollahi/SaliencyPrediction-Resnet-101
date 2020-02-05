import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# import pytorch_msssim 
class Arezoo(nn.Module):
	"""Atrous Spatial Pyramid Pooling"""

	def __init__(self,num_classes = 1):
		super(Arezoo, self).__init__()

		self.upsample_1 = nn.Sequential(
			nn.Conv2d(2048,1024,kernel_size = 1,stride = 1,padding = 0, dilation = 2),
			nn.BatchNorm2d(1024),
			Interpolate(),
			)

		
		self.upsample_2 = nn.Sequential(
			nn.Conv2d(1024,512,kernel_size = 1,stride = 1,padding = 0, dilation = 2),
			nn.BatchNorm2d(512),
			Interpolate(),
			)

		self.upsample_3 = nn.Sequential(
			nn.Conv2d(512,256,kernel_size = 1,stride = 1,padding = 0, dilation = 2),
			nn.BatchNorm2d(256),
			Interpolate(),
			)

		self.upsample_4 = nn.Sequential(
			nn.Conv2d(256,4,kernel_size = 1,stride = 1,padding = 0, dilation = 2),
			Interpolate(),
			)



		#self.upsample_5 = nn.Sequential(
		#	nn.Conv2d(128,1,kernel_size = 1,stride = 1,padding = 0),
		#	Interpolate(),
		#	)
		self.weight_init()
		###wight_init() bayad injaa bashe, ke faghat laaye haayi ke khodemoon ezafe kardim upsample haa ro biaad beheshoon weight initialize bokone va na kollelaaye haaye ghabli ro 
		resnet = models.resnet101(pretrained=True)
		self.num_classes = num_classes
		self.conv1 = resnet.conv1
		self.bn1 = resnet.bn1
		self.relu = resnet.relu
		self.maxpool = resnet.maxpool
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4

		# resnet2 = model ....
		# self.conv1_rn2 = resnet2.conv1
		for n, m in self.layer4.named_modules():
			if 'conv2' in n:
				m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)
			if 'conv1' in n or 'conv3' in n:
				m.dilation, m.stride = (2, 2), (1, 1)
		
		# self.L1_loss = nn.L1Loss()
		#self.BCELoss= nn.BCELoss(size_average=True)
		self.MSELoss = nn.MSELoss()

	def weight_init(self):

		# for n, m in self.layer4.named_modules():
		# 	if 'conv2' in n:
		# 		print("weight.max = ", m.weight.data.max(), "n = ", n)

		# print('====' * 50)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight.data)
		
		# for n, m in self.layer4.named_modules():
		# 	if 'conv2' in n:
		# 		print("weight.max = ", m.weight.data.max(), "n = ", n)

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()

	def forward(self, x):
		# import ipdb; ipdb.set_trace()
		# y = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		
		x = self.layer3(x)
		x = self.layer4(x)
		x = F.relu(self.upsample_1(x))
		x = F.relu(self.upsample_2(x))
		x = F.relu(self.upsample_3(x))
		# x = F.relu(self.upsample_4(x))
		x = torch.sigmoid(self.upsample_4(x))  ### F bood, 9 oct kardamesh torch.sigmoid




		# y = self.conv1(y)
		# y = self.bn1(x)
		# y = self.relu(x)
		# y = self.maxpool(x)
		# y = self.layer1(x)
		# x = self.layer2(x)
		# x = self.layer3(x)
		# x = self.layer4(x)
		# x = F.relu(self.upsample_1(x))
		# x = F.relu(self.upsample_2(x))
		# x = F.relu(self.upsample_3(x))
		# # x = F.relu(self.upsample_4(x))
		# x = F.sigmoid(self.upsample_4(x))
		# z = torch.concat(x,y)
		# return z 
		return x

	def loss(self,output, target):
		# return self.BCELoss(output,target)
		#return self.L1_loss(output, target) + (1 - pytorch_msssim.msssim(output,target))/2
		# return (1 - pytorch_msssim.msssim(output,target))/2
		# return self.L1_loss(output, target) 
		return self.MSELoss(output, target)
		



class Interpolate(nn.Module):
	"""Atrous Spatial Pyramid Pooling"""
	def __init__(self,scale_factor = 2,):
		super(Interpolate, self).__init__()
		self.interp = F.upsample
		self.scale_factor = scale_factor

	def forward(self,x):
		return self.interp(x,scale_factor = self.scale_factor,mode = "bilinear", align_corners=True) ####### 9 oct, align_corners ro ezafe kardam
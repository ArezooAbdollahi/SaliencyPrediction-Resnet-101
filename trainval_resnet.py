from __future__ import absolute_import, division, print_function
import random
import numpy as np
import os.path 
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.backends import cudnn
import torchvision.utils
from dataset import Inpainting
import datetime
import time

import torch.nn.init as init
from tqdm import tqdm
from model import *

from utils.salgan_utils import load_image, postprocess_prediction
from utils.salgan_utils import normalize_map

from IPython import embed
from evaluation.metrics_functions import AUC_Judd, AUC_Borji, AUC_shuffled, CC, NSS, SIM, EMD




cudnn.benchmark = True
manual_seed=627937
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)

def PSNR(mse):
	psnr = 20 * (255.0 /torch.sqrt(mse)).log10()
	return psnr

def CalculateMetrics(fground_truth, mground_truth, predicted_map):


	predicted_map = normalize_map(predicted_map)
	predicted_map = postprocess_prediction(predicted_map, (predicted_map.shape[0], predicted_map.shape[1]))
	predicted_map = normalize_map(predicted_map)
	predicted_map *= 255

	fground_truth = cv2.resize(fground_truth, (0,0), fx=0.5, fy=0.5)
	predicted_map = cv2.resize(predicted_map, (0,0), fx=0.5, fy=0.5)
	mground_truth = cv2.resize(mground_truth, (0,0), fx=0.5, fy=0.5)

	fground_truth = fground_truth.astype(np.float32)/255
	predicted_map = predicted_map.astype(np.float32)
	mground_truth = mground_truth.astype(np.float32)

	AUC_judd_answer = AUC_Judd(predicted_map, fground_truth)
	AUC_Borji_answer = AUC_Borji(predicted_map, fground_truth)
	nss_answer = NSS(predicted_map, fground_truth)
	cc_answer = CC(predicted_map, mground_truth)
	sim_answer = SIM(predicted_map, mground_truth)

	return AUC_judd_answer, AUC_Borji_answer, nss_answer, cc_answer, sim_answer



def main():

	writer = SummaryWriter(comment='Multi-scale-resnet-Train, OriginalImg(input)=Cat2000, GT:4 seperated Heatmap, 20 Categories, split:1800,200, resnet101. 150epoch, lr=1e-5, Adam, loss:MSE')

	model=Arezoo().to('cuda:0') 
	model.train()
	mode='train' # name of file train.txt
	

	val_dataset = Inpainting(
		root='/home/arezoo/5-DataSet/',
		mode='val' 
	)

	train_dataset = Inpainting(
		root='/home/arezoo/5-DataSet/',
		mode='train' 
	)

	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=8,
		num_workers=32,
		shuffle=True,
	)

	val_loader = torch.utils.data.DataLoader(
		dataset=val_dataset,
		batch_size=8,
		num_workers=32,
		shuffle=True,
	)
	print(val_loader.__len__())
	print(train_loader.__len__())
	optimizer = torch.optim.Adam(model.parameters(), lr= 1e-5)
	global_train_iter = 0
	global_val_iter = 0
	for epoch in tqdm(range(150),desc='epoch: '):
		folderpath=os.path.join('./outputImages', str(epoch))
		if not os.path.exists(folderpath):
			os.makedirs(folderpath)

		print("epoch training: " , epoch)
		model= train(model,train_loader,optimizer,epoch,writer,global_train_iter)

		modelpath=os.path.join('./models',str("model-")+str(epoch) +".pt")
		if (epoch+1) % 5 ==0 :
			torch.save(model.state_dict(), modelpath)

		global_train_iter +=1

		if (epoch+1)%5==0:

			folderpath=os.path.join('./outputImagesVal', str(epoch))
			if not os.path.exists(folderpath):
				os.makedirs(folderpath)
			print("epoch validation: " , epoch)
			Val(model,val_loader,epoch,writer,global_val_iter)
			global_val_iter += 1


def Val(model,val_loader,epoch,writer,global_val_iter):
	model.eval()
	length=val_loader.__len__()
	sum_psnr=0.0
	sum_loss=0.0
	sum_l1_loss = 0.0

	sum_AUC_Judd_16= 0.0
	sum_AUC_Borji_answer_16=0.0
	sum_nss_16=0.0
	sum_cc_16=0.0
	sum_sim_16= 0.0

	sum_AUC_Judd_32= 0.0
	sum_AUC_Borji_answer_32=0.0
	sum_nss_32=0.0
	sum_cc_32=0.0
	sum_sim_32= 0.0

	sum_AUC_Judd_64= 0.0
	sum_AUC_Borji_answer_64=0.0
	sum_nss_64=0.0
	sum_cc_64=0.0
	sum_sim_64= 0.0

	sum_AUC_Judd_128= 0.0
	sum_AUC_Borji_answer_128=0.0
	sum_nss_128=0.0
	sum_cc_128=0.0
	sum_sim_128= 0.0



	folderpath=os.path.join('./outputImagesVal', str(epoch))
	for i, data in tqdm(enumerate(val_loader), desc = "validating: "):
		with torch.no_grad():
			image, nameImg,target = data
			lenInputImgs = image.shape[0]

			image = Variable(image).to('cuda:0')
			output=model(image)
			target = Variable(target).to('cuda:0')
			
			loss = model.loss(output, target)
			PSNRValue=PSNR(loss)
			sum_psnr += PSNRValue.cpu().item()
			sum_loss += loss.item()

			for i in range(0, lenInputImgs):

				if (epoch+1)%5==0 :  
					##############################################################  
					Dirname= nameImg[i].split('-')[0]
					fground_truth_name= nameImg[i].split('-')[-1]+'.jpg'
					mground_truth_name= nameImg[i].split('-')[-1].split('.')[0]+'_SaliencyMap.jpg'
					##############################################################
					imagepath_128=os.path.join(folderpath, nameImg[i]+"-128.png")
					imagepath_64=os.path.join(folderpath, nameImg[i]+"-64.png")
					imagepath_32=os.path.join(folderpath, nameImg[i]+"-32.png")
					imagepath_16=os.path.join(folderpath, nameImg[i]+"-16.png")
					torchvision.utils.save_image(output[i,0,:,:], imagepath_128, normalize=True) ###change this to save all of them! 
					torchvision.utils.save_image(output[i,1,:,:], imagepath_64, normalize=True)
					torchvision.utils.save_image(output[i,2,:,:], imagepath_32, normalize=True)
					torchvision.utils.save_image(output[i,3,:,:], imagepath_16, normalize=True)



					fground_truth=cv2.imread(os.path.join('/home/arezoo/5-DataSet/FIXATIONMAPSresize/', Dirname, fground_truth_name),cv2.IMREAD_GRAYSCALE)

					mground_truth=cv2.imread(os.path.join('/home/arezoo/5-DataSet/cat2000saliencybenchmark/Cat2000Saliencymap/', Dirname, mground_truth_name),cv2.IMREAD_GRAYSCALE)
					
					################################################16
					predicted_map_16=cv2.imread(os.path.join(imagepath_16),cv2.IMREAD_GRAYSCALE)
					AUC_Judd_16, AUC_Borji_answer_16, nss_16, cc_16, sim_16 = CalculateMetrics(fground_truth, mground_truth, predicted_map_16)
					sum_AUC_Judd_16 +=AUC_Judd_16
					sum_AUC_Borji_answer_16+= AUC_Borji_answer_16
					sum_nss_16 += nss_16
					sum_cc_16 += cc_16
					sum_sim_16 += sim_16
					################################################16


					################################################32
					predicted_map_32=cv2.imread(os.path.join(imagepath_32),cv2.IMREAD_GRAYSCALE)
					AUC_Judd_32, AUC_Borji_answer_32, nss_32, cc_32, sim_32 = CalculateMetrics(fground_truth, mground_truth, predicted_map_32)
					sum_AUC_Judd_32 +=AUC_Judd_32
					sum_AUC_Borji_answer_32+= AUC_Borji_answer_32
					sum_nss_32 += nss_32
					sum_cc_32 += cc_32
					sum_sim_32 += sim_32
					################################################32 end

					################################################64
					predicted_map_64=cv2.imread(os.path.join(imagepath_64),cv2.IMREAD_GRAYSCALE)
					AUC_Judd_64, AUC_Borji_answer_64, nss_64, cc_64, sim_64 = CalculateMetrics(fground_truth, mground_truth, predicted_map_64)
					sum_AUC_Judd_64 +=AUC_Judd_64
					sum_AUC_Borji_answer_64+= AUC_Borji_answer_64
					sum_nss_64 += nss_64
					sum_cc_64 += cc_64
					sum_sim_64 += sim_64
					################################################64 end


					################################################128
					predicted_map_128=cv2.imread(os.path.join(imagepath_128),cv2.IMREAD_GRAYSCALE)
					AUC_Judd_128, AUC_Borji_answer_128, nss_128, cc_128, sim_128 = CalculateMetrics(fground_truth, mground_truth, predicted_map_128)
					sum_AUC_Judd_128 +=AUC_Judd_128
					sum_AUC_Borji_answer_128+= AUC_Borji_answer_128
					sum_nss_128 += nss_128
					sum_cc_128 += cc_128
					sum_sim_128 += sim_128
					################################################128 end


	if (epoch+1)%5==0:
	# if epoch>=0:
		#####################################################################################16
		avg_AUC_Judd_16= sum_AUC_Judd_16 / length
		avg_AUC_Borji_16= sum_AUC_Borji_answer_16 / length
		avg_nss_16= sum_nss_16 / length
		avg_cc_16= sum_cc_16 / length
		avg_sim_16= sum_sim_16 / length
		writer.add_scalar('eval-val16/avg_AUC_Judd',avg_AUC_Judd_16,global_val_iter)
		writer.add_scalar('eval-val16/avg_AUC_Borji',avg_AUC_Borji_16,global_val_iter)	
		writer.add_scalar('eval-val16/avg_nss',avg_nss_16,global_val_iter)
		writer.add_scalar('eval-val16/avg_cc',avg_cc_16,global_val_iter)
		writer.add_scalar('eval-val16/avg_sim',avg_sim_16,global_val_iter)	
		######################################################################################16 end

		#####################################################################################32
		avg_AUC_Judd_32= sum_AUC_Judd_32 / length
		avg_AUC_Borji_32= sum_AUC_Borji_answer_32 / length
		avg_nss_32= sum_nss_32 / length
		avg_cc_32= sum_cc_32 / length
		avg_sim_32= sum_sim_32 / length
		writer.add_scalar('eval-val32/avg_AUC_Judd',avg_AUC_Judd_32,global_val_iter)
		writer.add_scalar('eval-val32/avg_AUC_Borji',avg_AUC_Borji_32,global_val_iter)	
		writer.add_scalar('eval-val32/avg_nss',avg_nss_32,global_val_iter)
		writer.add_scalar('eval-val32/avg_cc',avg_cc_32,global_val_iter)
		writer.add_scalar('eval-val32/avg_sim',avg_sim_32,global_val_iter)	
		######################################################################################32 end


		#####################################################################################64
		avg_AUC_Judd_64= sum_AUC_Judd_64 / length
		avg_AUC_Borji_64= sum_AUC_Borji_answer_64 / length
		avg_nss_64= sum_nss_64 / length
		avg_cc_64= sum_cc_64 / length
		avg_sim_64= sum_sim_64 / length
		writer.add_scalar('eval-val64/avg_AUC_Judd',avg_AUC_Judd_64,global_val_iter)
		writer.add_scalar('eval-val64/avg_AUC_Borji',avg_AUC_Borji_64,global_val_iter)	
		writer.add_scalar('eval-val64/avg_nss',avg_nss_64,global_val_iter)
		writer.add_scalar('eval-val64/avg_cc',avg_cc_64,global_val_iter)
		writer.add_scalar('eval-val64/avg_sim',avg_sim_64,global_val_iter)	
		#####################################################################################64 end


		#####################################################################################128
		avg_AUC_Judd_128= sum_AUC_Judd_128 / length
		avg_AUC_Borji_128= sum_AUC_Borji_answer_128 / length
		avg_nss_128= sum_nss_128 / length
		avg_cc_128= sum_cc_128 / length
		avg_sim_128= sum_sim_128 / length
		writer.add_scalar('eval-val128/avg_AUC_Judd',avg_AUC_Judd_128,global_val_iter)
		writer.add_scalar('eval-val128/avg_AUC_Borji',avg_AUC_Borji_128,global_val_iter)	
		writer.add_scalar('eval-val128/avg_nss',avg_nss_128,global_val_iter)
		writer.add_scalar('eval-val128/avg_cc',avg_cc_128,global_val_iter)
		writer.add_scalar('eval-val128/avg_sim',avg_sim_128,global_val_iter)	
		#####################################################################################128 end




	avg_psnr = sum_psnr / length
	avg_loss = sum_loss / length
	writer.add_scalar('psnr/val_psnr',avg_psnr,global_val_iter)
	writer.add_scalar('loss/val_loss',avg_loss,global_val_iter)
	print("val_psnr: ", avg_psnr, "val_loss: ", avg_loss)


def train(model,train_loader,optimizer,epoch,writer,global_train_iter):

	folderpath=os.path.join('./outputImages', str(epoch))
	model.train()

	length=train_loader.__len__()
	sum_psnr=0.0
	sum_loss = 0.0
	sum_l1_loss = 0.0



	sum_AUC_Judd_16= 0.0
	sum_AUC_Borji_answer_16=0.0
	sum_nss_16=0.0
	sum_cc_16=0.0
	sum_sim_16= 0.0

	sum_AUC_Judd_32= 0.0
	sum_AUC_Borji_answer_32=0.0
	sum_nss_32=0.0
	sum_cc_32=0.0
	sum_sim_32= 0.0

	sum_AUC_Judd_64= 0.0
	sum_AUC_Borji_answer_64=0.0
	sum_nss_64=0.0
	sum_cc_64=0.0
	sum_sim_64= 0.0

	sum_AUC_Judd_128= 0.0
	sum_AUC_Borji_answer_128=0.0
	sum_nss_128=0.0
	sum_cc_128=0.0
	sum_sim_128= 0.0


	for i, data in tqdm(enumerate(train_loader), desc= "training"):

		image, nameImg, target = data

		lenInputImgs = image.shape[0]

		sizebatch=image.size()
		NumBatch= sizebatch[0]
		optimizer.zero_grad()
		model.zero_grad()
		image = Variable(image).to('cuda:0')
		output = model(image)

		target = Variable(target).to('cuda:0')
		loss = model.loss(output, target)
		loss.backward()
		optimizer.step()
		PSNRValue=PSNR(loss)
		sum_psnr += PSNRValue.cpu().item()
		sum_loss += loss.item()

		for i in range(0, lenInputImgs):
			if (epoch+1)%5==0: 
			
				##############################################################  
				Dirname= nameImg[i].split('-')[0]
				fground_truth_name= nameImg[i].split('-')[-1]+'.jpg'
				mground_truth_name= nameImg[i].split('-')[-1].split('.')[0]+'_SaliencyMap.jpg'
				##############################################################
			  
			  	# import ipdb; ipdb.set_trace()
				imagepath_128=os.path.join(folderpath, nameImg[i]+"-128.png")
				imagepath_64=os.path.join(folderpath, nameImg[i]+"-64.png")
				imagepath_32=os.path.join(folderpath, nameImg[i]+"-32.png")
				imagepath_16=os.path.join(folderpath, nameImg[i]+"-16.png")
				torchvision.utils.save_image(output[i,0,:,:], imagepath_128, normalize=True) ###change this to save all of them! 
				torchvision.utils.save_image(output[i,1,:,:], imagepath_64, normalize=True)
				torchvision.utils.save_image(output[i,2,:,:], imagepath_32, normalize=True)
				torchvision.utils.save_image(output[i,3,:,:], imagepath_16, normalize=True)


				fground_truth=cv2.imread(os.path.join('/home/arezoo/5-DataSet/FIXATIONMAPSresize/', Dirname, fground_truth_name),cv2.IMREAD_GRAYSCALE)

				mground_truth=cv2.imread(os.path.join('/home/arezoo/5-DataSet/cat2000saliencybenchmark/Cat2000Saliencymap/', Dirname, mground_truth_name),cv2.IMREAD_GRAYSCALE)
				
				################################################16
				predicted_map_16=cv2.imread(os.path.join(imagepath_16),cv2.IMREAD_GRAYSCALE)
				AUC_Judd_16, AUC_Borji_answer_16, nss_16, cc_16, sim_16 = CalculateMetrics(fground_truth, mground_truth, predicted_map_16)
				sum_AUC_Judd_16 +=AUC_Judd_16
				sum_AUC_Borji_answer_16+= AUC_Borji_answer_16
				sum_nss_16 += nss_16
				sum_cc_16 += cc_16
				sum_sim_16 += sim_16
				################################################16


				################################################32
				predicted_map_32=cv2.imread(os.path.join(imagepath_32),cv2.IMREAD_GRAYSCALE)
				AUC_Judd_32, AUC_Borji_answer_32, nss_32, cc_32, sim_32 = CalculateMetrics(fground_truth, mground_truth, predicted_map_32)
				sum_AUC_Judd_32 +=AUC_Judd_32
				sum_AUC_Borji_answer_32+= AUC_Borji_answer_32
				sum_nss_32 += nss_32
				sum_cc_32 += cc_32
				sum_sim_32 += sim_32
				################################################32 end

				################################################64
				predicted_map_64=cv2.imread(os.path.join(imagepath_64),cv2.IMREAD_GRAYSCALE)
				AUC_Judd_64, AUC_Borji_answer_64, nss_64, cc_64, sim_64 = CalculateMetrics(fground_truth, mground_truth, predicted_map_64)
				sum_AUC_Judd_64 +=AUC_Judd_64
				sum_AUC_Borji_answer_64+= AUC_Borji_answer_64
				sum_nss_64 += nss_64
				sum_cc_64 += cc_64
				sum_sim_64 += sim_64
				################################################64 end


				################################################128
				predicted_map_128=cv2.imread(os.path.join(imagepath_128),cv2.IMREAD_GRAYSCALE)
				AUC_Judd_128, AUC_Borji_answer_128, nss_128, cc_128, sim_128 = CalculateMetrics(fground_truth, mground_truth, predicted_map_128)
				sum_AUC_Judd_128 +=AUC_Judd_128
				sum_AUC_Borji_answer_128+= AUC_Borji_answer_128
				sum_nss_128 += nss_128
				sum_cc_128 += cc_128
				sum_sim_128 += sim_128
				################################################128 end

	if (epoch+1)%5==0:

		#####################################################################################16
		avg_AUC_Judd_16= sum_AUC_Judd_16 / length
		avg_AUC_Borji_16= sum_AUC_Borji_answer_16 / length
		avg_nss_16= sum_nss_16 / length
		avg_cc_16= sum_cc_16 / length
		avg_sim_16= sum_sim_16 / length
		writer.add_scalar('eval16/avg_AUC_Judd',avg_AUC_Judd_16,global_train_iter)
		writer.add_scalar('eval16/avg_AUC_Borji',avg_AUC_Borji_16,global_train_iter)	
		writer.add_scalar('eval16/avg_nss',avg_nss_16,global_train_iter)
		writer.add_scalar('eval16/avg_cc',avg_cc_16,global_train_iter)
		writer.add_scalar('eval16/avg_sim',avg_sim_16,global_train_iter)	
		######################################################################################16 end

		#####################################################################################32
		avg_AUC_Judd_32= sum_AUC_Judd_32 / length
		avg_AUC_Borji_32= sum_AUC_Borji_answer_32 / length
		avg_nss_32= sum_nss_32 / length
		avg_cc_32= sum_cc_32 / length
		avg_sim_32= sum_sim_32 / length
		writer.add_scalar('eval32/avg_AUC_Judd',avg_AUC_Judd_32,global_train_iter)
		writer.add_scalar('eval32/avg_AUC_Borji',avg_AUC_Borji_32,global_train_iter)	
		writer.add_scalar('eval32/avg_nss',avg_nss_32,global_train_iter)
		writer.add_scalar('eval32/avg_cc',avg_cc_32,global_train_iter)
		writer.add_scalar('eval32/avg_sim',avg_sim_32,global_train_iter)	
		######################################################################################32 end


		#####################################################################################64
		avg_AUC_Judd_64= sum_AUC_Judd_64 / length
		avg_AUC_Borji_64= sum_AUC_Borji_answer_64 / length
		avg_nss_64= sum_nss_64 / length
		avg_cc_64= sum_cc_64 / length
		avg_sim_64= sum_sim_64 / length
		writer.add_scalar('eval64/avg_AUC_Judd',avg_AUC_Judd_64,global_train_iter)
		writer.add_scalar('eval64/avg_AUC_Borji',avg_AUC_Borji_64,global_train_iter)	
		writer.add_scalar('eval64/avg_nss',avg_nss_64,global_train_iter)
		writer.add_scalar('eval64/avg_cc',avg_cc_64,global_train_iter)
		writer.add_scalar('eval64/avg_sim',avg_sim_64,global_train_iter)	
		#####################################################################################64 end


		#####################################################################################128
		avg_AUC_Judd_128= sum_AUC_Judd_128 / length
		avg_AUC_Borji_128= sum_AUC_Borji_answer_128 / length
		avg_nss_128= sum_nss_128 / length
		avg_cc_128= sum_cc_128 / length
		avg_sim_128= sum_sim_128 / length
		writer.add_scalar('eval128/avg_AUC_Judd',avg_AUC_Judd_128,global_train_iter)
		writer.add_scalar('eval128/avg_AUC_Borji',avg_AUC_Borji_128,global_train_iter)	
		writer.add_scalar('eval128/avg_nss',avg_nss_128,global_train_iter)
		writer.add_scalar('eval128/avg_cc',avg_cc_128,global_train_iter)
		writer.add_scalar('eval128/avg_sim',avg_sim_128,global_train_iter)	
		#####################################################################################128 end


	avg_psnr = sum_psnr / length
	avg_loss = sum_loss / length
	writer.add_scalar('psnr/train_psnr',avg_psnr,global_train_iter)
	writer.add_scalar('loss/train_loss',avg_loss,global_train_iter)
	print("train_psnr: ", avg_psnr, "train_loss: ", avg_loss)


	return model

if __name__ == '__main__':
	main()

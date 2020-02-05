import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage.io as sio
# import skimage.io as imsave
from skimage.color import rgb2gray
import os
from skimage.measure import compare_ssim as ssim
from scipy.ndimage import gaussian_filter

from skimage import measure as mymeasure

from skimage import color

from scipy import signal
import gauss
import cv2

import matplotlib.image as mpimg
import glob

from matplotlib.pyplot import imsave
from matplotlib.pyplot import cm

import torch
import torch.nn.functional as F

'''
This code generates heatmap from the inpainted images. 
It goes through all the images, and for each image, it goes through all the inpainted images with their corresponding masks
and it measures the difference between the inpainted image and the original image. 

The intuiton is that if the difference between the original image and the inpainted image be high, it means that 
there is s.th that the network couldn't predict it


'''

# Reading images from the data-set text files
fileOriginal = open("Allimages-9Categories.txt", "r") 
file_listOriginal=fileOriginal.readlines()
file_listOriginal = [x.strip() for x in file_listOriginal]

# Reading mask images
fileMask = open("Masks128.txt", "r")
file_listMask=fileMask.readlines()
file_listMask = [x.strip() for x in file_listMask]


# going through all images in the data-set 
for g in range(0, 900):	

	OriginalIndex= file_listOriginal[g]
	print(OriginalIndex)
	ref=cv2.imread(os.path.join('/home/arezoo/5-DataSet/Saliencyresize-total', OriginalIndex))

	weighted_output = []

	# going through all the inpainted image related to an image
	# here, for each image we have 16 inpainted images
	for k in range(0,16):	

		# getting the name of the inpainted images
		MaskIndex= file_listMask[k]
		nameofDir= OriginalIndex.split('/')[-2]
		str1= OriginalIndex.split('/')[-1].split('.')[0]
	
		str2=MaskIndex
		str3=str1+'-'+str2
		InpaintIndex=str3


		inpainted=cv2.imread(os.path.join('/home/arezoo/5-DataSet/SaliencyInpaint/Imgs512-Masks128', nameofDir, InpaintIndex))
		mask=cv2.imread(os.path.join('/home/arezoo/5-DataSet/SaliencyMasks/SqaureSaliencyMasks/128', MaskIndex))
	

		gray = 1 - rgb2gray(mask) # getting the area that is not masked 
		score = 1 - ssim(ref, inpainted, data_range=ref.max() - ref.min(), multichannel=True) # getting the difference between the original image and the inpainted one
		weight_mask = gray * score # getting the contribution percentage of each area that make the heatmap 
		weighted_output.append(weight_mask)
	
	weighted_ans = np.zeros(shape=gray.shape, dtype=np.float32) 
	
	# merging the score together to generate the final image
	for ans in weighted_output:
		weighted_ans = weighted_ans + ans
	#import ipdb; ipdb.set_trace()
	weighted_ans = gaussian_filter(weighted_ans, sigma=25)
	weighted_ans -= weighted_ans.min()
	weighted_ans /= weighted_ans.max()

	
	weighted_ans_pytorch = torch.from_numpy(weighted_ans)
	weighted_ans_pytorch = F.softmax(weighted_ans_pytorch.view(-1)).view(512, 512)
	weighted_ans_normal = weighted_ans_pytorch.numpy()


	# saving images with their corresponding original names
	plt.imshow(weighted_ans)
	indexsave=OriginalIndex.split('/')[-1].split('.')[0]
	mpimg.imsave(os.path.join('/home/arezoo/5-DataSet/HeatmapGT/SaliencyHeatmap/Imgs512-Masks128' , nameofDir, indexsave +'.png'), weighted_ans)
	
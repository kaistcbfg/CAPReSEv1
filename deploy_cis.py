
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from densenet import *

from xgboost import XGBClassifier
from sklearn.externals import joblib

import imgutils

import sys
import os
import copy

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

targetsample = sys.argv[1]
targetchr = sys.argv[2]
inputfile = sys.argv[3]
saveimg = sys.argv[4]

savedir = "/home/sillo/pytorch/SV_classify/data/testoutput/"
resolution = 20000
boxsize1 = 16
boxsize2 = 2
bkptsizelimit = 320000

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, 10)
	#

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)
	#

	def feature(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
	#	
#

model = Net()
model.load_state_dict(torch.load('/home/sillo/pytorch/SV_classify/finetuning/mnist/mnist_cnn.pt'))
model.train(False)
model = model.cuda()

xgclf = joblib.load('/home/sillo/pytorch/SV_classify/model/xg_model_mnisttf2.pkl')
#"""
def classify_image_DLv2(crop, DLmodel, XGmodel):

	rgbcropPIL = Image.fromarray(crop)

	trans = transforms.Compose([transforms.Scale((28,28)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
	croptensor = trans(rgbcropPIL)
	croptensor = croptensor.unsqueeze(0)
	cropvar = Variable(croptensor).cuda()

	enc_output = DLmodel.feature(cropvar)
	enc_raw =  enc_output.cpu().data.numpy()[0]

	enclist = np.array([enc_raw])
	y_pred = XGmodel.predict(enclist)

	return int(y_pred[0])
#
#if not os.path.exists(savedir + targetsample): os.mkdir(savedir + targetsample)

SVfilename = "/home/sillo/pytorch/hicgrad_tflearn_trans/data/11.Total_SV_region_bed/" + targetsample + "_paired_total.bed"

WGS_bkpt_list = []
"""
f = open(SVfilename)
for line in f:
	line = line.rstrip()
	linedata = line.split("\t")
	
	chr1 = linedata[0]
	chr2 = linedata[2]
	bkpt1 = int(linedata[1])
	bkpt2 = int(linedata[3])

	bkptsize = np.abs(bkpt1 - bkpt2)

	if chr1 == chr2 and chr1 == targetchr and bkptsize >= bkptsizelimit:
		WGS_bkpt_list.append((bkpt1, bkpt2))
	#
#
f.close()
#"""

#imgfilename = "/home/sillo/pytorch/hic_svdetect/data/img20k40sample/" + targetsample + "_" + targetchr + "_cis_20000_panznormdiv.png"
#imgfilename = '/home/sillo/pytorch/SV_classify/dil/chr8_dil_1.0.png'
img = cv2.imread(inputfile)
outimg = copy.copy(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = np.tril(img)
img_raw = copy.copy(img)
img = cv2.medianBlur(img, 3)
img = cv2.bilateralFilter(img,16,75,75)
img_y_len, img_x_len = img.shape
cis_axis = (0, 0, img_x_len ,img_y_len)

img_cont = copy.copy(img)
img_cont = cv2.Canny(img_cont, 51, 255)
ret, img_cont = cv2.threshold(img_cont, 51, 255, cv2.THRESH_BINARY)
img_cont = cv2.dilate(img_cont, None, iterations=1)
img_cont = cv2.erode(img_cont, None, iterations=1)
#img_cont = cv2.dilate(img_cont, None, iterations=1)

contimg = copy.copy(img_cont)
cont, h = cv2.findContours(img_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
jointcont = []
for i in cont:
	x, y, w, h = cv2.boundingRect(i)
	x2 = x + w
	y2 = y + h
	jointcont.append((x, y, x2, y2))
#

jointbox = []
for i in jointcont:
	if not imgutils.Liang_Barsky_line_rect_collision(i, cis_axis):
		jointbox.append(i)
	#
#

#jointbox = imgutils.non_max_suppression_fast(np.array(jointbox), 0)
#jointbox = jointcont

def refine_box_coord(bkptcoord, imgshape, boxsize):

	bkptx, bkpty = bkptcoord

	boxx = bkptx - boxsize
	boxy = bkpty - boxsize
	boxx2 = bkptx + boxsize
	boxy2 = bkpty + boxsize

	width, height = imgshape
	if boxx < 0: boxx = 0
	if boxx2 > width: boxx2 = width
	if boxy < 0: boxy = 0
	if boxy2 > height: boxy2 = height

	return (boxx, boxy, boxx2, boxy2)
#

def check_SV_overlap():

	return
#

def print_log(datalist):

	outstr = ''
	for i in datalist[:-1]:
		outstr += str(i)
		outstr += "\t"
	#
	outstr += str(datalist[-1])

	return outstr
#

cnt = 0
valid_cont_list = []
for contbox in jointbox:

	contx,conty,contx2,conty2 = contbox
	contcrop = img_raw[conty:conty2,contx:contx2]
	grad_x, grad_y, gradpattern = imgutils.calc_gradCenter(contcrop,contbox)

	gradbox = refine_box_coord((grad_x,grad_y),(img_x_len, img_y_len), boxsize1)		
	gradx, grady, gradx2, grady2 = gradbox

	crop = img_raw[grady:grady2, gradx:gradx2]

	pred = -1
	if not imgutils.Liang_Barsky_line_rect_collision(gradbox, cis_axis):
		try: pred = classify_image_DLv2(crop, model, xgclf)
		except:pass
	#	

	if pred == 1:
		cv2.rectangle(outimg, (contx, conty), (contx2, conty2), (0,255,0), 10)
		#cv2.rectangle(outimg, (gradx, grady), (gradx2, grady2), (0,0,255), 1)
		if not contbox in valid_cont_list: valid_cont_list.append(contbox)
	#

	else:
		for bkpt in WGS_bkpt_list:
			bkpt_x_orig, bkpt_y_orig = bkpt
			bkptx = bkpt_x_orig/resolution
			bkpty = bkpt_y_orig/resolution

			wgsbox1 = refine_box_coord((bkptx,bkpty),(img_x_len, img_y_len), boxsize1)
			wgsbox2 = refine_box_coord((bkptx,bkpty),(img_x_len, img_y_len), boxsize2)

			if imgutils.bb_intersection_over_union(contbox, wgsbox2) > 0:
				if not imgutils.Liang_Barsky_line_rect_collision(wgsbox1, cis_axis):
					wgsx, wgsy, wgsx2, wgsy2 = wgsbox1
					crop = img_raw[wgsy:wgsy2,wgsx:wgsx2]
					pred = -1
					try: pred = classify_image_DLv2(crop, model, xgclf)
					except:pass
					if pred == 1:
						cv2.rectangle(outimg, (contx, conty), (contx2, conty2), (0,255,0), 10)
						#cv2.rectangle(outimg, (wgsx, wgsy), (wgsx2, wgsy2), (0,255,255), 3)
						if not contbox in valid_cont_list: valid_cont_list.append(contbox)
					#
				#
			#
		# 
	#
#			

WGS_bkpt_match = []

for box in valid_cont_list:
	contx, conty, contx2, conty2 = box
	outdata = ['dummy', targetchr,  contx, contx2, targetchr, conty, conty2, targetsample]
	matchSV = []

	for bkpt in WGS_bkpt_list:
		bkpt_x_orig, bkpt_y_orig = bkpt
		bkptx = bkpt_x_orig/resolution
		bkpty = bkpt_y_orig/resolution

		wgsbox2 = refine_box_coord((bkptx,bkpty),(img_x_len, img_y_len), boxsize2)
		if imgutils.bb_intersection_over_union(box, wgsbox2) > 0: 
			WGS_bkpt_match.append(bkpt)
			matchSV.append(bkpt)
		#
	#
	outdata.append(len(matchSV))
	for i in matchSV:
		bx, by = i
		outdata.append(bx)
		outdata.append(by)
	#

	print print_log(outdata)
#

for bkpt in WGS_bkpt_list:
	bkpt_x_orig, bkpt_y_orig = bkpt

	if bkpt not in WGS_bkpt_match:

		bkptx = bkpt_x_orig/resolution
		bkpty = bkpt_y_orig/resolution

		wgsbox = refine_box_coord((bkptx,bkpty),(img_x_len, img_y_len), boxsize1)

		if not imgutils.Liang_Barsky_line_rect_collision(wgsbox, cis_axis):
			pred = -1
			wgsx, wgsy, wgsx2, wgsy2 = wgsbox
			crop = img_raw[wgsy:wgsy2,wgsx:wgsx2]
			try: pred = classify_image_DLv2(crop, model, xgclf)
			except:pass
			if pred == 1:
				wgsx, wgsy, wgsx2, wgsy2 = wgsbox
				cv2.rectangle(outimg, (wgsx, wgsy), (wgsx2, wgsy2), (255,0,255), 10)
				outdata = ['dummy', targetchr, -1, -1, targetchr, -1, -1, targetsample, bkpt_x_orig, bkpt_y_orig]	
				print print_log(outdata)
		#	
	#
#

outfilename = savedir = "/home/sillo/pytorch/SV_classify/data/testoutput/" + targetsample + "_" + targetchr + "_dloutput.png"
#cv2.imwrite(outfilename, outimg)
cv2.imwrite(saveimg, outimg)

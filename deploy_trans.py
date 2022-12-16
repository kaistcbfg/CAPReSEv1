
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from xgboost import XGBClassifier
from sklearn.externals import joblib

import imgutils
import custom_utils

import sys
import os
import copy

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

inputfile = sys.argv[1]
saveimg = sys.argv[2]

savedir = "FIX PATH"
resolution = 500000
boxsize1 = 16
boxsize2 = 0

allchrsize_file = "FIX PATH"
centromere_file = "FIX PATH"
startcoord_dict, chrsize_dict, chrlist = custom_utils.gen_chr_cumsum_dict(allchrsize_file, resolution)
chrlist2 = copy.copy(chrlist)

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
model.load_state_dict(torch.load("FIX PATH"))
model.train(False)
model = model.cuda()

xgclf = joblib.load("FIX PATH")

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

SVfilename = "FIX PATH"
WGS_bkpt_list = [] #clear this list to use Hi-C only
f = open(SVfilename)
for line in f:
	line = line.rstrip()
	linedata = line.split("\t")
	
	chr1 = linedata[0]
	chr2 = linedata[2]
	bkpt1 = int(linedata[1])
	bkpt2 = int(linedata[3])

	if chr1 != chr2:
		WGS_bkpt_list.append((chr1, bkpt1, chr2, bkpt2))
	#
#
f.close()

def transregion_preproc(transregion):

	transregion2 = copy.copy(transregion)

	img = cv2.medianBlur(transregion2, 3)
	img = cv2.bilateralFilter(img,8,75,75)

	img_cont = cv2.Canny(img, 17, 255)
	#ret, img_cont = cv2.threshold(img_cont, 17, 255, cv2.THRESH_BINARY)
	img_cont = cv2.dilate(img_cont, None, iterations=1)
	#img_cont = cv2.erode(img_cont, None, iterations=1)

	cont, h = cv2.findContours(img_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	jointcont = []
	for i in cont:
		x, y, w, h = cv2.boundingRect(i)
		x2 = x + w
		y2 = y + h
		jointcont.append((x, y, x2, y2))
	#

	return jointcont
#

imgfilename = "FIX PATH"
allchrimg = cv2.imread(inputfile)
#allchrimg = cv2.imread(imgfilename)
outimg = copy.copy(allchrimg)
allchrimg = cv2.cvtColor(allchrimg, cv2.COLOR_RGB2GRAY)

transregionlist = []
for chr1 in chrlist:
	for chr2 in chrlist2:
		if chr1 != chr2 and chrlist.index(chr1) < chrlist.index(chr2): 
			start1 = startcoord_dict[chr1] 
			end1 = start1 + chrsize_dict[chr1]
			start2 =  startcoord_dict[chr2] 
			end2 = start2 + chrsize_dict[chr2]
			transregionlist.append((chr1, chr2, start1, end1, start2, end2))
	#
#

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

for i in transregionlist:

	valid_cont_list = []
	chr1, chr2, start1, end1, start2, end2 = i
	transregion = allchrimg[start2:end2, start1:end1]
	contlist = transregion_preproc(transregion)
	ylen, xlen = transregion.shape

	for j in contlist:

		contx,conty,contx2,conty2 = j
		contstart1 = contx+start1
		contend1 = contx2+start1
		contstart2 = conty+start2
		contend2 = conty2+start2

		contcrop = transregion[conty:conty2,contx:contx2]
		grad_x, grad_y, gradpattern = imgutils.calc_gradCenter(contcrop,j)

		gradbox = refine_box_coord((grad_x,grad_y),(xlen, ylen), boxsize1)		
		gradx, grady, gradx2, grady2 = gradbox

		crop = transregion[grady:grady2, gradx:gradx2]

		pred = -1
		try: pred = classify_image_DLv2(crop, model, xgclf)
		except:pass

		if pred == 1:
			cv2.rectangle(outimg, (contstart1, contstart2), (contend1, contend2), (0,0,255), 1)
			if not j in valid_cont_list: valid_cont_list.append(j)
		#

		else:
			for bkpt in WGS_bkpt_list:
				wgschr1, bkpt_x_orig, wgschr2, bkpt_y_orig = bkpt
				if wgschr1 == chr1 and wgschr2 == chr2:
					bkptx = bkpt_x_orig/resolution 
					bkpty = bkpt_y_orig/resolution

					wgsbox1 = refine_box_coord((bkptx,bkpty),(xlen, ylen), boxsize1)
					wgsbox2 = refine_box_coord((bkptx,bkpty),(xlen, ylen), boxsize2)

					wgsx, wgsy, wgsx2, wgsy2 = wgsbox1
					crop = transregion[wgsy:wgsy2,wgsx:wgsx2]
					pred = -1
					try: pred = classify_image_DLv2(crop, model, xgclf)
					except:pass
					if pred == 1:
						cv2.rectangle(outimg, (contstart1, contstart2), (contend1, contend2), (0,0,255), 1)
						if not j in valid_cont_list: valid_cont_list.append(j)
					#
				#
			#
		# 
	#

	WGS_bkpt_match = []

	for box in valid_cont_list:
		contx, conty, contx2, conty2 = box
		outdata = ['dummy', chr1,  contx, contx2, chr2, conty, conty2, targetsample]
		matchSV = []

		for bkpt in WGS_bkpt_list:
			wgschr1, bkpt_x_orig, wgschr2, bkpt_y_orig = bkpt
			if wgschr1 == chr1 and wgschr2 == chr2:
				bkptx = bkpt_x_orig/resolution
				bkpty = bkpt_y_orig/resolution

				wgsbox2 = refine_box_coord((bkptx,bkpty),(xlen, ylen), boxsize2)
				if imgutils.bb_intersection_over_union(box, wgsbox2) > 0: 
					WGS_bkpt_match.append(bkpt)
					matchSV.append(bkpt)
				#
			#
		outdata.append(len(matchSV))
		for i in matchSV:
			wgschr1, bx, wgschr2, by = i
			outdata.append(bx)
			outdata.append(by)
		#

		print print_log(outdata)
	#

	for bkpt in WGS_bkpt_list:
		wgschr1, bkpt_x_orig, wgschr2, bkpt_y_orig = bkpt

		if bkpt not in WGS_bkpt_match and wgschr1 == chr1 and wgschr2 == chr2:

			bkptx = bkpt_x_orig/resolution
			bkpty = bkpt_y_orig/resolution

			wgsbox = refine_box_coord((bkptx,bkpty),(xlen, ylen), boxsize1)

			pred = -1
			wgsx, wgsy, wgsx2, wgsy2 = wgsbox
			crop = transregion[wgsy:wgsy2,wgsx:wgsx2]
			try: pred = classify_image_DLv2(crop, model, xgclf)
			except:pass
			if pred == 1:
				wgsx, wgsy, wgsx2, wgsy2 = wgsbox
				cv2.rectangle(outimg, (wgsx+start1, wgsy+start2), (wgsx2+start1, wgsy2+start2), (255,0,255), 1)
				outdata = ['dummy', chr1, -1, -1, chr2, -1, -1, targetsample, bkpt_x_orig, bkpt_y_orig]	
				print print_log(outdata)
			#
		#
	#
#

#outfilename = savedir + targetsample + "_trans_dloutput.png"
cv2.imwrite(saveimg, outimg)


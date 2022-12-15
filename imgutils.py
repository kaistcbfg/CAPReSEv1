import numpy as np
import cv2
from scipy import stats

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	if len(boxes) == 0:
		return []

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []

	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / area[idxs[:last]]

		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	return boxes[pick].astype("int")
#

def Liang_Barsky_line_rect_collision(boxcoord, linecoord):

	line_x_start, line_y_start, line_x_end, line_y_end = linecoord
	x, y, x2, y2 = boxcoord

	p = [-(line_x_end - line_x_start), (line_x_end - line_x_start), -(line_y_end - line_y_start), (line_y_end - line_y_start)]
	q = [line_x_start - x, x2 - line_x_start, line_y_start - y, y2 - line_y_start ]

	u1 = -np.inf
	u2 = np.inf

	for i in range(4):
		t = float(q[i])/p[i]
		if (p[i] < 0 and u1 < t): u1 = t
		elif (p[i] > 0 and u2 > t): u2 = t
	#

	if (u1 > u2 or u1 > 1 or u1 < 0):
		collision = False
	else:
		collision = True

	return collision
#

def bb_intersection_over_union(boxA, boxB):

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou
#

def mark_Box_to_Img(outimg, boxlist, color):

	for i in boxlist:
		if len(i) == 4:
			x, y, x2, y2 = i
		else:
			x, y, x2, y2, origx, origy = i
		cv2.rectangle(outimg, (x,y), (x2, y2), color, 2)
	#
	return outimg
#

def mark_Dot_to_Img(outimg, boxlist, color):

	for i in boxlist:
		if len(i) == 4:
			x, y, x2, y2 = i
		else:
			x, y, x2, y2, origx, origy = i
	
		center_x = x + (x2 - x)/2
		center_y = y + (y2 - y)/2

		cv2.circle(outimg,  (center_x, center_y), 2, color, -1)
	#
	return outimg
#

def calc_gradCenter(grey_patch, boxcoord):

	gradcenter = (-1, -1)
	x, y, x2, y2 = boxcoord
	h = y2 - y
	w = x2 - x

	img_h, img_w = grey_patch.shape
	if not img_h%2 == 0: img_h += 1
	if not img_w%2 == 0: img_w += 1
	grey_patch = cv2.resize(grey_patch, (img_w, img_h))

	crop_up_half = np.hsplit(np.vsplit(grey_patch, 2)[0], 2)
	crop_down_half =  np.hsplit(np.vsplit(grey_patch, 2)[-1], 2)

	crop_up_left = crop_up_half[0]
	crop_up_right = crop_up_half[-1]
	crop_down_left = crop_down_half[0]
	crop_down_right = crop_down_half[-1]

	score_list = [np.sum(crop_up_left),np.sum(crop_up_right),np.sum(crop_down_left),np.sum(crop_down_right)]
	#score_list = [stats.mode(crop_up_left,axis=None),stats.mode(crop_up_right,axis=None),stats.mode(crop_down_left,axis=None),stats.mode(crop_down_right,axis=None)]
	seq = sorted(score_list)
	score_index = [seq.index(v) for v in score_list]
	
	maxindex = score_list.index(max(score_list))
	minindex = score_list.index(min(score_list))	
	#secondindex = score_list.index(score_list[score_index.index(1)])
	if maxindex == 0:
		#if minindex == 3: gradcenter = (x, y)
		#elif secondindex == 3: gradcenter = (x + w/2, y + h/2)			
		gradcenter = (x, y)

	elif maxindex == 1:
		#if minindex == 2: gradcenter = (x2, y)
		#elif secondindex == 2: gradcenter = (x + w/2, y + h/2)			
		gradcenter = (x2, y)	

	elif maxindex == 2:
		#if minindex == 1: gradcenter = (x, y2)
		#elif secondindex == 1: gradcenter = (x + w/2, y + h/2)			
		gradcenter = (x, y2)

	elif maxindex == 3:
		#if minindex == 0: gradcenter = (x2, y2)
		#elif secondindex == 0: gradcenter = (x + w/2, y + h/2)			
		gradcenter = (x2, y2)
	#	

	return gradcenter[0], gradcenter[1], maxindex
#

def save_Boxlist_to_Img(img, samplename, chrname, boxlist, savepath, patch_class=0, padding=0, resize=None):

	for idx,box in enumerate(boxlist):
		x, y, x2, y2 = box
		imgpatch = img[y-padding:y2+padding, x-padding:x2+padding]
	
		imgsavename = samplename + "_" + chrname + "_" + str(patch_class) + "_" + str(idx) + ".png"
		fullimgsavename = savepath + "/" + imgsavename
		if resize == None:
			cv2.imwrite(fullimgsavename, imgpatch)
		else:
			imgpatch = cv2.resize(imgpatch, (resize, resize))
			cv2.imwrite(fullimgsavename, imgpatch)
		#
	#	
	return
#



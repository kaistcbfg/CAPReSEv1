import numpy as np
import cv2
import os

def get_genedata(gencode_gtf_file, targetchr):

	gencode_dict = {}
	genename_dict = {}
	gene_list = []

	f = open(gencode_gtf_file)
	f.readline()
	for line in f:
		line = line.rstrip()
		linedata = line.split("\t")

		strand = linedata[3]
		if strand == "+": tss = int(linedata[1])
		elif strand == "-": tss = int(linedata[2])

		genetype = linedata[5]
		geneid = linedata[4]
		genename = linedata[6][1:]
		genestart = int(linedata[1])
		geneend = int(linedata[2])

		if genetype == " protein_coding":
			if linedata[0] == targetchr:
				gencode_dict[geneid] = (tss, (int(linedata[1]), int(linedata[2])))
				genename_dict[geneid] = genename
				gene_list.append((genename, geneid, tss, strand, (genestart, geneend) ))
			#
		#
	#
	f.close()

	return (gencode_dict, genename_dict, gene_list) 
#

def get_HiC_bkptdata(hic_dlresult_file, targetsamplename, targetchr):

	cis_bkpt_list = []
	f = open(hic_dlresult_file)
	for line in f:
		line = line.rstrip()
		linedata = line.split("\t")

		chrname1 = linedata[0]
		chrname2 = linedata[2]
		samplename = linedata[4]
		svtype = linedata[5]
		hictype = linedata[6]
	
		bkpt1 = int(linedata[1])
		bkpt2 = int(linedata[3])
	
		if chrname1 == chrname2:
			if chrname1 == targetchr:
				if samplename == targetsamplename:
					cis_bkpt_list.append((bkpt1, bkpt2))
				#
			#
		#	
	#
	f.close()

	return cis_bkpt_list
#

def get_pannorm_SE_data(SE_file, targetchrname):

	selist = []

	f = open(SE_file)
	for line in f:
		line = line.rstrip()
		linedata = line.split("\t")

		chrname = linedata[0]
		sestart = int(linedata[1])
		seend = int(linedata[2])

		if chrname == targetchrname:
			selist.append((sestart, seend))

	#

	f.close()

	return selist
#

def get_mindist(bkptlist, tss, genebody):

	start, end = genebody
	mindist = 1000000000
	for bkpt in bkptlist:
		if bkpt > start and bkpt < end: return -1
		else:
			if np.abs(bkpt - tss) < mindist: mindist = np.abs(bkpt - tss)
		#
	#

	return mindist
#

def calc_hic_SVtype(hicsv_coord, hicimg, resolution):
	
	bkpty, bkptx = hicsv_coord
	bkptx = bkptx/resolution
	bkpty = bkpty/resolution

	imgX, imgY = hicimg.shape
	
	x = bkptx - 6
	x2 = bkptx + 6
	y = bkpty - 6
	y2 = bkpty + 6

	if x < 0: x = 0
	if y < 0: y = 0
	if x2 > imgX: x2 = imgX
	if y2 > imgY: y2 = imgY

	#print y,y2,x,x2,bkptx,bkpty,imgX,imgY
	imgcrop = hicimg[y:y2, x:x2]
	if imgcrop.shape[0] != 16: imgcrop = cv2.resize(imgcrop, (16,16))
	if imgcrop.shape[1] != 16: imgcrop = cv2.resize(imgcrop, (16,16))
		
	crop_up_half = np.hsplit(np.vsplit(imgcrop, 2)[0], 2)
	crop_down_half =  np.hsplit(np.vsplit(imgcrop, 2)[-1], 2)

	crop_up_left = crop_up_half[0]
	crop_up_right = crop_up_half[-1]
	crop_down_left = crop_down_half[0]
	crop_down_right = crop_down_half[-1]

	score_list = [np.sum(crop_up_left),np.sum(crop_up_right),np.sum(crop_down_left),np.sum(crop_down_right)]
	#print score_list
	seq = sorted(score_list)
	score_index = [seq.index(v) for v in score_list]
	
	maxindex = score_list.index(max(score_list))

	return maxindex
#

def gen_INVlist(invlistfile):
	
	invlist = []

	f = open(invlistfile)
	for line in f:
		line = line.rstrip()
		linedata = line.split("\t")
		
		samplename = linedata[0]
		chrname = linedata[1]
		bkpt1 = int(linedata[2])
		bkpt2 = int(linedata[4])

		invlist.append((samplename, chrname, bkpt1, bkpt2))
	#
	f.close()

	return invlist
#

def fix_buff_size(imgsize, contcoord, buffsize):

	x,y,x2,y2 = contcoord

	outlist = []
	vallist = [x-buffsize, x2+buffsize, y-buffsize, y2+buffsize]
	for i in vallist:
		if i < 0: outlist.append(0)
		elif i > imgsize: outlist.append(imgsize)
		else: outlist.append(i)
	#

	return tuple(outlist)
#

def fix_buff_size_trans(imgsize, contcoord, buffsize):

	w, h = imgsize
	x,y,x2,y2 = contcoord

	outlist = []
	vallist = [x-buffsize, x2+buffsize]
	vallist2 = [y-buffsize, y2+buffsize]
	for i in vallist:
		if i < 0: outlist.append(0)
		elif i > w: outlist.append(w)
		else: outlist.append(i)
	#
	for i in vallist2:
		if i < 0: outlist.append(0)
		elif i > h: outlist.append(h)
		else: outlist.append(i)
	#

	return tuple(outlist)
#

def gen_chr_cumsum_dict(allchrsize_file, resolution):

	chr_cumsum_dict = {}

	chrlist = []
	for i in range(22):
		chrlist.append('chr' + str(i+1))
	#
	chrlist += ['chrX', 'chrY']

	chrsizelist = []
	chrsize_dict = {}
	f = open(allchrsize_file)
	for line in f:
		line  = line.rstrip()
		linedata = line.split()
		chrsize = int(linedata[1])/resolution
		chrsizelist.append(chrsize)
		chrsize_dict[linedata[0]] = chrsize
		#print linedata[0], int(linedata[1])/resolution
	#
	chrsizelist = [0] + chrsizelist[:-1]
	#print chrsizelist

	chrcumlist = list(np.cumsum(np.array(chrsizelist)))
	for i in range(len(chrlist)):
		chr_cumsum_dict[chrlist[i]] = chrcumlist[i]
	#
		
	return chr_cumsum_dict, chrsize_dict, chrlist
#

def proc_SV_bedfile(bedfile):
	
	sample = os.path.basename(bedfile).split("_")[0]

	translist = []
	f = open(bedfile)
	for line  in f:
		line = line.rstrip()
		linedata = line.split("\t")
	
		chr1 = linedata[0]
		chr2 = linedata[2]
		bkpt1 = int(linedata[1])
		bkpt2 = int(linedata[3])

		if chr1 != chr2: translist.append((sample, chr1, bkpt1, chr2, bkpt2))
	#
	f.close()

	return translist
#


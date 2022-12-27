
import numpy as np
import pandas as pd

import glob
import sys
import os

targetsamplename = sys.argv[1]

sigint_file_list = glob.glob('/home/sillo/CRC2019/SE_hijack/trans/output/01_sigint/*.csv.gz')

gencode_gtf_file = "/home/CBFG/Colorectal_2019/annotation/gencode.v27.protein_coding_lincRNA"
gene_dict = {}
f = open(gencode_gtf_file)
f.readline()
for line in f:
	line = line.rstrip()
	linedata = line.split("\t")

	chrname = linedata[0]

	strand = linedata[3]
	if strand == "+": 
		tss1 = (int(linedata[1])/40000) * 40000
		tss2 = tss1 - 40000
		if tss2 < 0: tss2 = 0
		binid = ".".join([chrname, str(tss2), str(tss1)])

	elif strand == "-":
		tss1 = (int(linedata[2])/40000) * 40000
		tss2 = tss1 + 40000
		binid = ".".join([chrname, str(tss1), str(tss2)])
	#

	genetype = linedata[5]
	geneid = linedata[4]
	genename = linedata[6]

	if genetype == " protein_coding":
		if binid in gene_dict.keys():
			gene_dict[binid] += ("," + geneid)	
		else:
			gene_dict[binid] = geneid
		#
	#
#
f.close()

def gen_se_dict():

	se_dict = {}
	f = open("/home/CBFG/Colorectal_2019/ChIP/all_SE.normal.merge.bed")
	for line in f:
		line = line.rstrip()
		[chrname, start, end] = line.split('\t')
		if chrname in se_dict.keys(): se_dict[chrname].append((int(start), int(end)))
		else: se_dict[chrname] = [(int(start), int(end))]
	#

	f.close()

	return se_dict
#
se_dict = gen_se_dict()

def check_SE_in_cont(bincoord, chrname, se_dict):

	flag = np.nan
	selist = se_dict[chrname]
	for start, end in selist:
		if start <= bincoord <= end or start <= bincoord + 40000 <= end:
			flag = ".".join([chrname, str(start), str(end)]) 
			break
		#
	#

	return flag
#

outfile_path = '/home/sillo/CRC2019/SE_hijack/trans/output/02_geneSEpair'
outfile_noSEgradgene  = open('{}/{}_noSEgradgene.txt'.format(outfile_path, targetsamplename),'w')
outfile_noSEotherside = open('{}/{}_noSEotherside.txt'.format(outfile_path, targetsamplename),'w')
outfile_geneSE_nonsig = open('{}/{}_geneSE_nonsig.txt'.format(outfile_path, targetsamplename),'w')
outfile_geneSE_sig    = open('{}/{}_geneSE_sig.txt'.format(outfile_path, targetsamplename),'w')

for filepath in sigint_file_list:
	filename = os.path.basename(filepath)
	[samplename, chrname1, bkpt1, contx, contx2, chrname2, bkpt2, conty, conty2] = filename.split(".")[0].split("_")
	
	if samplename == targetsamplename:

		df = pd.read_csv(filepath, sep='\t',error_bad_lines=False, index_col=False ,compression='gzip')
		df['xgene'] = df['frag1'].map(gene_dict)
		df['xgene'] = df['xgene'].fillna(0)
		df['yse']   = df['ybin'].apply(check_SE_in_cont, args=(chrname2, se_dict))
		df['yse'] = df['yse'].fillna(0)
	
		df['ygene'] = df['frag2'].map(gene_dict)
		df['ygene'] = df['ygene'].fillna(0)
		df['xse']   = df['xbin'].apply(check_SE_in_cont, args=(chrname1, se_dict))
		df['xse'] = df['xse'].fillna(0)

		x_side_SE_list = df['xse'].to_list()
		blank_x = [0 for i in range(len(x_side_SE_list))]
		y_side_SE_list = df['yse'].to_list()
		blank_y = [0 for i in range(len(y_side_SE_list))]

		if x_side_SE_list == blank_x and y_side_SE_list == blank_y: #No SE in region

			x_noSE_gene_df = df[(df['xgene'] != 0) & (df['yse'] == 0)]
			y_noSE_gene_df = df[(df['ygene'] != 0) & (df['xse'] == 0)]

			outfile_noSEgradgene.write(x_noSE_gene_df['xgene'].to_csv(sep='\t', index=False, header=False))
			outfile_noSEgradgene.write(y_noSE_gene_df['ygene'].to_csv(sep='\t', index=False, header=False))

		elif x_side_SE_list == blank_x and  y_side_SE_list != blank_y: #Only yside SE 

			x_geneSE_nonsig_df = df[(df['xgene'] != 0) & (df['yse'] != 0) & (df['exp_fc'] < 2)]
			x_geneSE_sig_df    = df[(df['xgene'] != 0) & (df['yse'] != 0) & (df['exp_fc'] >= 2)]
			y_noSE_gene_df = df[(df['ygene'] != 0) & (df['xse'] == 0)]

			outfile_geneSE_nonsig.write(x_geneSE_nonsig_df[['xgene','yse','exp_fc']].to_csv(sep='\t', index=False, header=False))
			outfile_geneSE_sig.write(x_geneSE_sig_df[['xgene','yse']].to_csv(sep='\t', index=False, header=False))
			outfile_noSEotherside.write(y_noSE_gene_df['ygene'].to_csv(sep='\t', index=False, header=False))

		elif x_side_SE_list != blank_x and  y_side_SE_list == blank_y: #Only xside SE

			y_geneSE_nonsig_df = df[(df['ygene'] != 0) & (df['xse'] != 0) & (df['exp_fc'] < 2)]
			y_geneSE_sig_df    = df[(df['ygene'] != 0) & (df['xse'] != 0) & (df['exp_fc'] >= 2)]
			x_noSE_gene_df = df[(df['xgene'] != 0) & (df['yse'] == 0)]

			outfile_geneSE_nonsig.write(y_geneSE_nonsig_df[['ygene','xse','exp_fc']].to_csv(sep='\t', index=False, header=False))
			outfile_geneSE_sig.write(y_geneSE_sig_df[['ygene','xse']].to_csv(sep='\t', index=False, header=False))
			outfile_noSEotherside.write(x_noSE_gene_df['xgene'].to_csv(sep='\t', index=False, header=False))

		elif x_side_SE_list != blank_x and  y_side_SE_list != blank_y: #Both side SE
	
			x_geneSE_nonsig_df = df[(df['xgene'] != 0) & (df['yse'] != 0) & (df['exp_fc'] < 2)]
			x_geneSE_sig_df    = df[(df['xgene'] != 0) & (df['yse'] != 0) & (df['exp_fc'] >= 2)]
			y_geneSE_nonsig_df = df[(df['ygene'] != 0) & (df['xse'] != 0) & (df['exp_fc'] < 2)]
			y_geneSE_sig_df    = df[(df['ygene'] != 0) & (df['xse'] != 0) & (df['exp_fc'] >= 2)]

			outfile_geneSE_nonsig.write(x_geneSE_nonsig_df[['xgene','yse','exp_fc']].to_csv(sep='\t', index=False, header=False))
			outfile_geneSE_sig.write(x_geneSE_sig_df[['xgene','yse']].to_csv(sep='\t', index=False, header=False))
			outfile_geneSE_nonsig.write(y_geneSE_nonsig_df[['ygene','xse','exp_fc']].to_csv(sep='\t', index=False, header=False))
			outfile_geneSE_sig.write(y_geneSE_sig_df[['ygene','xse']].to_csv(sep='\t', index=False, header=False))
		#
	#
#
outfile_noSEotherside.close()
outfile_noSEgradgene.close()
outfile_geneSE_nonsig.close()
outfile_geneSE_sig.close()

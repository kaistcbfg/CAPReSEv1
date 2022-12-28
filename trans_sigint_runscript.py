
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import sys

targetsamplename = sys.argv[1]

contlist = []
hicsv_dlresult_file = "/home/sillo/scripts/proc/SV_gene_enh_trans/trans_40kb_all_finemap_v3.txt"
#hicsv_dlresult_file = '/home/sillo/CRC2019/SE_hijack/trans/test/test.txt'
f = open(hicsv_dlresult_file)
f.readline()
for line in f:
	line = line.rstrip()
	linedata = line.split()

	chrname2 = linedata[0]
	chrname1 = linedata[2]

	samplename = linedata[4]

	iswgs = linedata[10]
	if iswgs == 'yes':
		bkpt2 = int(linedata[11])
		bkpt1 = int(linedata[12])
	else:
		bkpt2 = int(linedata[13])
		bkpt1 = int(linedata[14])
	#
	
	marker = linedata[19]
	if not marker == 'NA':

		conty  = int(linedata[19])
		conty2 = int(linedata[20])
		contx  = int(linedata[21])
		contx2 = int(linedata[22])

		if samplename == targetsamplename:
			contbox = (chrname1, contx, contx2, chrname2, conty, conty2, bkpt1, bkpt2)
			contlist.append(contbox)
		#
	#
#
f.close()

pannorm_meandist_file = "/home/sillo/CRC2019/coverage_calc/trans_data/Pannorm/meandist4Mb.txt"
pannorm_dist_exp_profile = {}
f = open(pannorm_meandist_file)
f.readline()
f.readline()
for line in f:
	line = line.rstrip()
	dist,exp = line.split("\t")
	pannorm_dist_exp_profile[int(dist)] = float(exp)/12.1
#
pannorm_dist_exp_profile[0] = pannorm_dist_exp_profile[1] 
maxdist = max(pannorm_dist_exp_profile.keys())

def grep_sigints(df, contbox, pannorm_dist_exp_profile):

	chrnamex, contx, contx2, chrnamey, conty, conty2, bkpt1, bkpt2 = contbox

	grad_df = df[(df['chrname1'] == chrnamex) & (df['xbin'].between(contx,contx2)) & (df['chrname2'] == chrnamey) & (df['ybin'].between(conty,conty2))]
	grad_df['recomb_dist'] = ((grad_df['xbin']-bkpt1).abs().floordiv(40000) + (grad_df['ybin']-bkpt2).abs().floordiv(40000))
	grad_df = grad_df[(grad_df['recomb_dist'] <= 50)]
	grad_df_200kb_mean = grad_df[grad_df['recomb_dist']==5]['capture_res'].mean()
	grad_df['norm200kb'] = (grad_df['capture_res'].divide(grad_df_200kb_mean))
	grad_df['exp_200kb'] = grad_df['recomb_dist'].map(pannorm_dist_exp_profile)
	grad_df['exp_fc'] = grad_df['norm200kb']/grad_df['exp_200kb']

	return grad_df
#

if not len(contlist) == 0:

	covnorm_file = '/home/sillo/CRC2019/coverage_calc/trans_data/Normalized2/{}.trans.40kb.normalized.gz'.format(targetsamplename)
	df = pd.read_csv(covnorm_file, compression='gzip', sep='\t', names=["frag1", "frag2", "cov_frag1", "cov_frag2", "freq", "dist", "rand", "exp_value_capture", "capture_res"], error_bad_lines=False, index_col=False)
	df['chrname1'] = df.frag1.str.split(".").str[0]
	df['xbin'] = df.frag1.str.split(".").str[1].astype(int)
	df['chrname2'] = df.frag2.str.split(".").str[0]
	df['ybin'] = df.frag2.str.split(".").str[1].astype(int)

	for contbox in contlist:
		chrname1, contx, contx2, chrname2, conty, conty2, bkpt1, bkpt2 = contbox
		max_gradsize = max([np.abs(contx - bkpt1), np.abs(contx2 - bkpt1)])/40000 + max([np.abs(conty - bkpt2), np.abs(conty2 - bkpt2)])/40000
		if max_gradsize >= 5:
			outfilename = '/home/sillo/CRC2019/SE_hijack/trans/output/01_sigint/{}_{}_{}_{}_{}_{}_{}_{}_{}.csv.gz'.format(targetsamplename, chrname1, bkpt1, contx, contx2, chrname2, bkpt2, conty, conty2)
			grad_df = grep_sigints(df, contbox, pannorm_dist_exp_profile)
			grad_df.to_csv(outfilename, compression='gzip', sep='\t')
		#
	#
else: pass
#



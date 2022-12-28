
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import sys

targetsamplename = sys.argv[1]
chrlist = ['chr' + str(i+1) for i in range(22)]
querydict = {}
for i in chrlist: querydict[i] = []

hicsv_dlresult_file = "/home/martyr/result/Colon_cancer/chr16_blacklist_region/finemapping_cis.v2_over2.remove.blacklist.txt"
#hicsv_dlresult_file = "./test/test.txt"
f = open(hicsv_dlresult_file)
f.readline()
for line in f:
	line = line.rstrip()
	linedata = line.split("\t")
		
	samplename = linedata[4]
	chrname = linedata[0]

	marker = linedata[5]
	sv_num = linedata[10]
	
	if samplename == targetsamplename and marker != 'NA':

		contx  = int(linedata[5])
		contx2 = int(linedata[6])
		conty  = int(linedata[7])
		conty2 = int(linedata[8])

		if sv_num == 'yes':

			bkpt1 = int(linedata[11])
			bkpt2 = int(linedata[12])
	
		else:
			bkpt1 = int(linedata[13])
			bkpt2 = int(linedata[14])
		#
		
		contbox = (chrname, contx, contx2, chrname, conty, conty2, bkpt1, bkpt2)
		querydict[chrname].append(contbox)
	#
#
f.close()

pannorm_meandist_file = "/home/sillo/CRC2019/coverage_calc/Pannorm/20kb/meandist2Mb.txt"
pannorm_dist_exp_profile = {}
f = open(pannorm_meandist_file)
f.readline()
f.readline()
for line in f:
	line = line.rstrip()
	dist,exp = line.split("\t")
	pannorm_dist_exp_profile[int(dist)] = float(exp)/4.69
#
pannorm_dist_exp_profile[0] = pannorm_dist_exp_profile[1] 
maxdist = max(pannorm_dist_exp_profile.keys())

def check_pattern_condition(contbox):

	flag = True

	chrnamex, contx, contx2, chrnamey, conty, conty2, bkpt1, bkpt2 = contbox
	if (np.abs(contx2 - contx) + np.abs(conty2 - conty)) <= 200000: flag = False

	return flag
#

def grep_sigints(df, contbox, pannorm_dist_exp_profile):

	chrnamex, contx, contx2, chrnamey, conty, conty2, bkpt1, bkpt2 = contbox

	grad_df = df[(df['chrname1'] == chrnamex) & (df['xbin'].between(contx,contx2)) & (df['chrname2'] == chrnamey) & (df['ybin'].between(conty,conty2))]
	grad_df['recomb_dist'] = ((grad_df['xbin']-bkpt1).abs().floordiv(20000) + (grad_df['ybin']-bkpt2).abs().floordiv(20000))
	grad_df = grad_df[(grad_df['recomb_dist'] <= 100)]
	grad_df_200kb_mean = grad_df[grad_df['recomb_dist']==10]['capture_res'].mean()
	grad_df['norm200kb'] = (grad_df['capture_res'].divide(grad_df_200kb_mean))
	grad_df['exp_200kb'] = grad_df['recomb_dist'].map(pannorm_dist_exp_profile)
	grad_df['exp_fc'] = grad_df['norm200kb']/grad_df['exp_200kb']
	#grad_sig_df = grad_df[(grad_df['exp_fc'] >= 2)]

	return grad_df
#

for targetchrname in chrlist:
	contlist = querydict[targetchrname]
	if not len(contlist) == 0:

		covnorm_file = '/home/sillo/CRC2019/coverage_calc/Normalized/{}.{}.20kb.normalized.gz'.format(targetsamplename, targetchrname)
		df = pd.read_csv(covnorm_file, compression='gzip', sep='\t', names=["frag1", "frag2", "cov_frag1", "cov_frag2", "freq", "dist", "rand", "exp_value_capture", "capture_res"], error_bad_lines=False, index_col=False)
		df['chrname1'] = df.frag1.str.split(".").str[0]
		df['xbin'] = df.frag1.str.split(".").str[1].astype(int)
		df['chrname2'] = df.frag2.str.split(".").str[0]
		df['ybin'] = df.frag2.str.split(".").str[1].astype(int)

		for contbox in contlist:
			chrname1, contx, contx2, chrname2, conty, conty2, bkpt1, bkpt2 = contbox
			max_gradsize = max([np.abs(contx - bkpt1), np.abs(contx2 - bkpt1)])/20000 + max([np.abs(conty - bkpt2), np.abs(conty2 - bkpt2)])/20000
			if max_gradsize >= 10:
				outfilename = '/home/sillo/CRC2019/SE_hijack/cis/output/01_sigint/{}_{}_{}_{}_{}_{}_{}_{}_{}.csv.gz'.format(targetsamplename, chrname1, bkpt1, contx, contx2, chrname2, bkpt2, conty, conty2)
				grad_df = grep_sigints(df, contbox, pannorm_dist_exp_profile)
				grad_df.to_csv(outfilename, compression='gzip', sep='\t')
			#
		#

	else: pass
#



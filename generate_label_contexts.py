import tensorflow as tf
import os
from os.path import join
import numpy as np
from util import *

def convolve_context_window(gt, wsize):
	'''
	Convolve the context window over the given ground truth mask.

	:param gt: The ground truth mask.
	:param wsize: The size of the context window.
	
	:returns An array of tuples of the form (given_label, context_label)
	'''
	assert wsize % 2 == 1
	assert wsize < gt.shape[0] and wsize < gt.shape[1]

	wrad = int(np.floor(wsize/2))
	context_pairs = []

	for i in range(wrad, gt.shape[0] - wrad):
		for j in range(wrad, gt.shape[1] - wrad):
			given_lab = gt[i, j]
			if given_lab == 0 or given_lab == 255: continue

			reg = gt[i-wrad:i+wrad+1, j-wrad:j+wrad+1].ravel()
			reg = reg[fg_mask_for(reg)]

			diff_labs = reg[reg != given_lab]
			uniq_diff_labs = np.unique(diff_labs)

			for lab in uniq_diff_labs:
				context_pairs.append((given_lab, lab]))

	return context_pairs

def get_context_for_imset(imset, writer, wsize=151):
	'''
	Generates the context pairs for all foreground pixels in the given image set.

	:param imset: The image set to generate the context pairs for.
	:param writer: The TFRecordWriter to write the serialized context pairs to.
	:param wsize: The context window size.
	'''
	for idx in range(1, num_img_for(imset)+1):
		gt = load_gt(imset, idx, reshape=False)
		pairs = convolve_context_window(gt, wsize)

		for given_lab, context_labs in pairs:
			tf_features = {
				given_tf_key: int64_feature(given_lab),
				context_tf_key: bytes_feature(context_labs.tostring())
			}
			tf_ex = tf.train.Example(features=tf_features)
			writer.write(tf_ex.SerializeToString())

if __name__ == '__main__':
	tfrecord_fname = 'context.tfrecord'
	writer = tf.python_io.TFRecordWriter(join(data_dir, tfrecord_fname))

	get_context_for_imset('val', writer)
	get_context_for_imset('test', writer)

	write.close()

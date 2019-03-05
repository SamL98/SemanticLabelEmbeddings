import tensorflow as tf
from hdf5storage import loadmat
from os.path import join

ds_path = 'D:/datasets/processed/voc2012'
nc = 20

data_dir = 'data'
embeddings_model_dir = 'models/embeddings_model'

given_tf_key = 'given_label'
context_tf_key = 'context_label'

def num_img_for(imset):
	val_size = 724
	if imset == 'val':
		return val_size
	else:
		return 1449-val_size

def load_gt(imset, idx, reshape=False):
	global ds_path

	gt = loadmat(join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat') % idx)['truth_img']

	if reshape:
		gt = gt.ravel()

	return gt

def fg_mask_for(gt):
	return ((gt > 0) & (gt < 255))

def bytes_feature(val):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def int64_feature(val):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))
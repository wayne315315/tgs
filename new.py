from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from skimage.transform import resize, rotate

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, concatenate, BatchNormalization, AveragePooling2D, UpSampling2D, Activation, Dropout, Input, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam



### utils.py ###

IMG_SIZE_ORI = 101
IMG_SIZE_TAR = 128


train_csv_path = "./input/train.csv"
depth_csv_path = "./input/depths.csv"
image_path = "./input/train/images/{}.png"
mask_path = "./input/train/masks/{}.png"


def coverage_to_class(val):    
	for i in range(0, 11):
		if val * 10 <= i :
			return i

def iou_metric(y_true_in, y_pred_in, print_table=False):
	labels = np.squeeze(y_true_in, axis=-1)
	y_pred = np.squeeze(np.round(y_pred_in), axis=-1)

	true_objects = 2
	pred_objects = 2

	intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

	# Compute areas (needed for finding the union between all objects)
	area_true = np.histogram(labels, bins=true_objects)[0]
	area_pred = np.histogram(y_pred, bins=pred_objects)[0]
	area_true = np.expand_dims(area_true, -1) # (2,1)
	area_pred = np.expand_dims(area_pred, 0) # (1,2)

	# Compute union
	union = area_true + area_pred - intersection

	# Exclude background from the analysis
	intersection = intersection[1:,1:]
	union = union[1:,1:]
	union[union == 0] = 1e-9

	# Compute the intersection over union
	iou = intersection / union

    # Precision helper function
	def precision_at(threshold, iou):
		matches = iou > threshold
		true_positives = np.sum(matches, axis=1) == 1   # Correct objects
		false_positives = np.sum(matches, axis=0) == 0  # Missed objects
		false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
		tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
		return tp, fp, fn

    # Loop over IoU thresholds
	prec = []
	if print_table:
		print("Thresh\tTP\tFP\tFN\tPrec.")
	for t in np.arange(0.5, 1.0, 0.05):
		tp, fp, fn = precision_at(t, iou)
		if (tp + fp + fn) > 0:
			p = tp / (tp + fp + fn)
		else:
			p = 1.0
		if print_table:
			print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
		prec.append(p)

	if print_table:
		print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
	return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

### data.py ###
def read_dataset(n_splits=5, index=0, shuffle=True, seed=1234567890):

	assert index in list(range(n_splits))

	# Loading of training/testing ids and depths
	# Reading the training data and the depths, store them in a DataFrame. 
	# Also create a test DataFrame with entries from depth not in train.
	train_df = pd.read_csv(train_csv_path, index_col="id", usecols=[0])
	depths_df = pd.read_csv(depth_csv_path, index_col="id")
	train_df = train_df.join(depths_df)
	test_df = depths_df[~depths_df.index.isin(train_df.index)]

	# Read images and masks
	# Load the images and masks into the DataFrame and divide the pixel values by 255.
	train_df["images"] = [np.asarray(load_img(image_path.format(idx), grayscale=True)) / 255 
		for idx in tqdm(train_df.index)]
	train_df["masks"] = [np.asarray(load_img(mask_path.format(idx), grayscale=True)) / 255 
		for idx in tqdm(train_df.index)]

	# Calculating the salt coverage and salt coverage classes
	# Counting the number of salt pixels in the masks and dividing them by the image size. 
	# Also create 11 coverage classes, -0.1 having no salt at all to 1.0 being salt only. 
	# Plotting the distribution of coverages and coverage classes, and the class against the raw coverage.
	train_df["coverage"] = train_df.masks.map(np.sum) / pow(IMG_SIZE_ORI, 2)
	train_df["coverage_class"] = train_df.coverage.map(coverage_to_class)


	kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

	images = np.array(train_df.images.values.tolist()).reshape(-1, IMG_SIZE_ORI, IMG_SIZE_ORI, 1)
	masks = np.array(train_df.masks.values.tolist()).reshape(-1, IMG_SIZE_ORI, IMG_SIZE_ORI, 1)
	classes = train_df.coverage_class

	indices = [(train_index, valid_index) for train_index, valid_index in kf.split(masks, classes)]

	train_index, valid_index = indices[index]

	x_train, x_valid = images[train_index], images[valid_index]
	y_train, y_valid = masks[train_index], masks[valid_index]

	# x_train : (3200, 101, 101, 1)
	# y_train : (3200, 101, 101, 1)
	# x_test : (800, 101, 101, 1)
	# y_test : (800, 101, 101, 1)
	return x_train, x_valid, y_train, y_valid

# on-the-fly transformation 
def random_rotate(image, mask, rotate_nb):
	rotate_nb = tf.cast(rotate_nb, tf.int32)
	return tf.image.rot90(image, k=rotate_nb), tf.image.rot90(mask, k=rotate_nb)

class Bilinear(Layer):
    def __init__(self, size, **kwargs):
        self.size = size
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return tf.image.resize_bilinear(x, self.size)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size[0], self.size[1], input_shape[3])

    def get_config(self):
        config = {'size' : self.size}
        base_config = super().get_config()
        config.update(base_config)
        return config

def build_densemask(input_layer, growth_nb=8, drop_ratio=0.25):

	### conv1 (output: 128 * 128 * (2 * growth_nb))
	input_128 = Bilinear([IMG_SIZE_TAR, IMG_SIZE_TAR])(input_layer)
	conv1 = Conv2D(growth_nb * 2, (3, 3), padding="same")(input_128)
	conv1_ = conv1

	# conv1 (output: 128 * 128 * (8 * growth_nb))
	for _ in range(6):
		fconv1 = BatchNormalization()(conv1)
		fconv1 = Activation('relu')(fconv1)
		fconv1 = Conv2D(growth_nb * 4, (1, 1), padding="same")(fconv1)
		fconv1 = Dropout(drop_ratio)(fconv1)
		fconv1 = BatchNormalization()(fconv1)
		fconv1 = Activation('relu')(fconv1)
		fconv1 = Conv2D(growth_nb, (3, 3), padding="same")(fconv1)
		fconv1 = Dropout(drop_ratio)(fconv1)
		conv1 = concatenate([conv1, fconv1])

	#### conv2 (output : 64 * 64 * (4 * growth_nb))
	trans1 = BatchNormalization()(conv1)
	trans1 = Activation('relu')(trans1)
	trans1 = Conv2D(growth_nb * 4, (1, 1), padding="same")(trans1)
	conv2 = AveragePooling2D()(trans1)
	conv2_ = conv2


	# conv2 (output: 64 * 64 * (16 * growth_nb))
	for _ in range(12):
		fconv2 = BatchNormalization()(conv2)
		fconv2 = Activation('relu')(fconv2)
		fconv2 = Conv2D(growth_nb * 4, (1, 1), padding="same")(fconv2)
		fconv2 = Dropout(drop_ratio * 2)(fconv2)
		fconv2 = BatchNormalization()(fconv2)
		fconv2 = Activation('relu')(fconv2)
		fconv2 = Conv2D(growth_nb, (3, 3), padding="same")(fconv2)
		fconv2 = Dropout(drop_ratio * 2)(fconv2)
		conv2 = concatenate([conv2, fconv2])

	#### conv3 (output : 32 * 32 * (8 * growth_nb))
	trans2 = BatchNormalization()(conv2)
	trans2 = Activation('relu')(trans2)
	trans2 = Conv2D(growth_nb * 8, (1, 1), padding="same")(trans2)
	conv3 = AveragePooling2D()(trans2)
	conv3_ = conv3

	# conv3 (output: 32 * 32 * (32 * growth_nb))
	for _ in range(24):
		fconv3 = BatchNormalization()(conv3)
		fconv3 = Activation('relu')(fconv3)
		fconv3 = Conv2D(growth_nb * 4, (1, 1), padding="same")(fconv3)
		fconv3 = Dropout(drop_ratio * 2)(fconv3)
		fconv3 = BatchNormalization()(fconv3)
		fconv3 = Activation('relu')(fconv3)
		fconv3 = Conv2D(growth_nb, (3, 3), padding="same")(fconv3)
		fconv3 = Dropout(drop_ratio * 2)(fconv3)
		conv3 = concatenate([conv3, fconv3])

	#### conv4 (output : 16 * 16 * (16 * growth_nb))
	trans3 = BatchNormalization()(conv3)
	trans3 = Activation('relu')(trans3)
	trans3 = Conv2D(growth_nb * 16, (1, 1), padding="same")(trans3)
	conv4 = AveragePooling2D()(trans3)
	conv4_ = conv4

	# conv4 (output: 16 * 16 * (64 * growth_nb))
	for _ in range(48):
		fconv4 = BatchNormalization()(conv4)
		fconv4 = Activation('relu')(fconv4)
		fconv4 = Conv2D(growth_nb * 4, (1, 1), padding="same")(fconv4)
		fconv4 = Dropout(drop_ratio * 2)(fconv4)
		fconv4 = BatchNormalization()(fconv4)
		fconv4 = Activation('relu')(fconv4)
		fconv4 = Conv2D(growth_nb, (3, 3), padding="same")(fconv4)
		fconv4 = Dropout(drop_ratio * 2)(fconv4)
		conv4 = concatenate([conv4, fconv4])

	#### mid (output : 8 * 8 * (32 * growth_nb))
	trans4 = BatchNormalization()(conv4)
	trans4 = Activation('relu')(trans4)
	trans4 = Conv2D(growth_nb * 32, (1, 1), padding="same")(trans4)
	mid = AveragePooling2D()(trans4)

	"""
	conv1_ : 128 * 128 * (2 * growth_nb)
	conv2_ : 64 * 64 * (4 * growth_nb)
	conv3_ : 32 * 32 * (8 * growth_nb)
	conv4_ : 16 * 16 * (16 * growth_nb)
	mid : 8 * 8 * (32 * growth_nb)
	"""

	# transition up
	tconv4 = Conv2DTranspose(growth_nb * 16, (3, 3), strides=(2, 2), padding="same")(mid)
	conv4_ = concatenate([conv4_, tconv4]) # (output : 16 * 16 * (32 * growth_nb))

	# trainsition up
	tconv3 = Conv2DTranspose(growth_nb * 8, (3, 3), strides=(2, 2), padding="same")(conv4_)
	conv3_ = concatenate([conv3_, tconv3]) # (output : 32 * 32 * (16 * growth_nb))

	# trainsition up
	tconv2 = Conv2DTranspose(growth_nb * 4, (3, 3), strides=(2, 2), padding="same")(conv3_)
	conv2_ = concatenate([conv2_, tconv2]) # (output : 64 * 64 * (8 * growth_nb))

	# trainsition up
	tconv1 = Conv2DTranspose(growth_nb * 2, (3, 3), strides=(2, 2), padding="same")(conv2_)
	feature = concatenate([conv1_, tconv1]) # (output : 128 * 128 * (4 * growth_nb))

	print("feature shape : ", feature.shape) # (output : 128 * 128 * (4 * growth_nb))

	feature = BatchNormalization()(feature)
	feature = Activation('relu')(feature)
	logit = Conv2D(1, (1, 1), padding="same")(feature) # 128 * 128 * 1
	prob = Activation('sigmoid')(logit) # 128 * 128 * 1
	output_layer = Bilinear([IMG_SIZE_ORI, IMG_SIZE_ORI])(prob) # 101 * 101 * 1

	return output_layer


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
	y_true = tf.reshape(y_true, [-1])
	y_pred = tf.reshape(y_pred, [-1])
	epsilon = 1e-08
	loss_1 = - alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.log(tf.clip_by_value(y_pred, epsilon, 1))
	loss_2 = - (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.log(tf.clip_by_value(1 - y_pred, epsilon, 1))
	loss = loss_1 + loss_2
	loss = tf.reduce_mean(loss)

	return loss

def lovasz_loss(y_true, y_pred):
	labels = tf.squeeze(y_true, [-1])
	logits = tf.squeeze(y_pred, [-1])
	epsilon = 1e-07
	logits = - tf.log((1 / tf.clip_by_value(logits, epsilon, 1 - epsilon)) - 1)

	return lovasz_hinge(logits, labels, per_image=False, ignore=None)

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    # handle zero division
    shape = jaccard.shape
    mask_nan = tf.equal(jaccard, np.nan)
    mask_inf = tf.equal(jaccard, np.inf) | tf.equal(jaccard, - np.inf) 
    ones = tf.ones_like(jaccard, dtype=tf.float32)
    zeros = tf.zeros_like(jaccard, dtype=tf.float32)
    jaccard = tf.where(mask_nan, zeros, jaccard)
    jaccard = tf.where(mask_inf, ones, jaccard)

    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


###############################################################################
pretrainpath = "pretrain.h5"
densepath = "dense/lovasz.h5"
sesspath = "dense/lovasz.ckpt"
index = 4

x_train, x_valid, y_train, y_valid = read_dataset(n_splits=5, index=index, shuffle=True, seed=1234567890)

flip_x_train = np.flip(x_train, axis=2)
flip_y_train = np.flip(y_train, axis=2)

x_train = np.vstack((x_train, flip_x_train))
y_train = np.vstack((y_train, flip_y_train))


input_layer = Input(shape=(IMG_SIZE_ORI, IMG_SIZE_ORI, 1), name='input')
output_layer = build_densemask(input_layer)
dense = Model(input_layer, output_layer)


### Configuration
epochs = 10000
batch_size = 4
val_batch_size = 4
steps = (len(x_train) // batch_size) + 1
val_steps = (len(x_valid) // val_batch_size) + 1
init_lr = 0.001
min_lr = 1e-07
earlystopping_patience = 10
reducelr_patience = 2

### Dataset preparation
data_x_train = tf.placeholder(tf.float32, [None, None, None, 1])
data_y_train = tf.placeholder(tf.float32, [None, None, None, 1])
data_x_valid = tf.placeholder(tf.float32,[None, None, None, 1])
data_y_valid = tf.placeholder(tf.float32, [None, None, None, 1])

train_dataset = tf.data.Dataset.from_tensor_slices((data_x_train, data_y_train)).batch(batch_size).shuffle(len(x_train)).repeat()
valid_dataset = tf.data.Dataset.from_tensor_slices((data_x_valid, data_y_valid)).batch(val_batch_size).shuffle(len(x_valid)).repeat()
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

random_dataset = tf.data.Dataset.range(4).shuffle(4).repeat()
random_iterator = random_dataset.make_one_shot_iterator()
rotate_nb = random_iterator.get_next()

next_element = iterator.get_next()
next_element = random_rotate(*next_element, rotate_nb)
x, y = next_element
x = tf.to_float(x)
y = tf.to_float(y)

y_pred = dense(x)
loss = lovasz_loss(y, y_pred)

with tf.name_scope('adam_optimizer'):
	learning_rate = tf.placeholder(tf.float32, shape=[])
	adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = adam.minimize(loss)

init_op = tf.global_variables_initializer()
train_init_op = iterator.make_initializer(train_dataset)
valid_init_op = iterator.make_initializer(valid_dataset)

with tf.Session() as sess:

	sess.run(init_op)
	sess.run(train_init_op, feed_dict={data_x_train: x_train, data_y_train: y_train})
	sess.run(valid_init_op, feed_dict={data_x_valid: x_valid, data_y_valid: y_valid})

	print("Start loading weights....")
	dense.load_weights(pretrainpath)
	print("Complete loading weights....")

	best_val_loss = float('inf')
	index_es = index_lr = 0
	lr = init_lr
	train_focal_losses = []
	val_focal_losses = []
	saver = tf.train.Saver()

	for epoch in tqdm(range(epochs)):
		print("### Epoch : ", epoch)
		
		print("### Train phase")
		train_losses = []
		for _ in tqdm(range(steps)):
			train_loss = sess.run((train_op, loss), feed_dict={learning_rate: lr})[1]
			train_loss = np.mean(train_loss) ###
			print(train_loss)
			train_losses.append(train_loss)
		train_focal_loss = np.mean(train_losses)
		train_focal_losses.append(train_focal_loss)
		print("Training loss : ", train_focal_loss)
		

		print("### Validation phase")
		val_loss = [sess.run(loss) for _ in tqdm(range(val_steps))]
		val_loss = np.mean(val_loss)
		val_focal_losses.append(val_loss)
		print("Validation loss : ", val_loss)

		# Only save the best model
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			save_model(dense, densepath, include_optimizer=False)
			print("Save model into %s" % densepath)
			saver.save(sess, sesspath)
			print("Save session into %s" % sesspath)
			index_es = index_lr =  0
			continue

		# Early stopping
		if index_es > earlystopping_patience:
			print("*** Early stopping after %s epochs without improving ***" % index_es)
			print("*** Best validation loss : ", best_val_loss)
			break

		# Reduce lr
		if index_lr > reducelr_patience:
			lr = max(0.1 * lr, min_lr)
			index_lr = 0
			print("*** Reduce learning rate to : ", lr)
			continue

		index_es += 1
		index_lr += 1
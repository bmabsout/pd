import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import os
import csv

import dataset
import model

dataset = dataset.preprocess_dataset()
# cd Documents/
# cd pd_ICLR_rebuttal/
# cd Benchmark_Tests
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m KT_RandomSearch_CIFAR100_Benchmark_with_regularization

# grad_steps = 25 trials * 2 executions each trial * 782 batches per execution + (5 * 782) for final training = 43000 steps



# CIFAR100 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()








# seed:
def set_seeds(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	tf.random.set_seed(seed)
	np.random.seed(seed)

def set_global_determinism(seed):
	set_seeds(seed=seed)

	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

	tf.config.threading.set_inter_op_parallelism_threads(1)
	tf.config.threading.set_intra_op_parallelism_threads(1)



SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]
SEED = [24, 34, 49, 60, 74, 89, 97, 100]

for seed in SEED:  

	set_global_determinism(seed=seed)
	print(seed), print("") 

	import time
	start_time = time.time()  


	max_trials = 25
	model_num = "1 with reg"


	# define tuner
	print("random search")
	tuner = keras_tuner.RandomSearch(
	    hypermodel=build_model,
	    objective="val_accuracy",
	    max_trials=max_trials,
	    executions_per_trial=2,
	    overwrite=True,
	    project_name="CIFAR100: %s" % SEED
	)


	# search
	tuner.search(train_images, train_labels, validation_data=(validation_images, validation_labels), batch_size=64)

	# retrieve and train best model
	best_hps = tuner.get_best_hyperparameters(5)
	model = build_model(best_hps[0])


	# Use early stopping
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


	# TRAIN Model
	print("")
	print("TRAINING")
	train_epochs = 20
	model.fit(train_images, train_labels, batch_size=64, validation_data=(validation_images, validation_labels), epochs=train_epochs, callbacks=[callback])


	time_lapsed = time.time() - start_time


	# evaluating model on test and train data
	batch_size = 64

	np.random.seed(0)
	eIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
	random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = train_images[eIndices], train_labels[eIndices], test_images[eIndices], test_labels[eIndices]

	print(""), print(""), print("Evaluating models on test data after randomization")

	# evaluating on train, test images
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy()
	# train_loss = lossfn(random_batch_train_labels, model(random_batch_train_images))
	# test_loss = lossfn(random_batch_test_labels, model(random_batch_test_images))

	train_loss = model.evaluate(random_batch_train_images, random_batch_train_labels)[0]
	test_loss = model.evaluate(random_batch_test_images, random_batch_test_labels)[0]

	print("unnormalized train loss: %s" % train_loss)
	print("unnormalized test loss: %s" % test_loss)


	def graph_history(history):
		integers = [i for i in range(1, (len(history))+1)]

		ema = []
		avg = history[0]

		ema.append(avg)

		for loss in history:
			avg = (avg * 0.9) + (0.1 * loss)
			ema.append(avg)


		x = [j * rr * (batches * pop_size) for j in integers]
		y = history

		# plot line
		plt.plot(x, ema[:len(history)])
		# plot title/captions
		plt.title("Keras Tuner CIFAR100")
		plt.xlabel("Gradient Steps")
		plt.ylabel("Validation Loss")
		plt.tight_layout()


		print("ema:"), print(ema), print("")
		print("x:"), print(x), print("")
		print("history:"), print(history), print("")


		
		# plt.savefig("TEST_DATA/PD_trial_%s.png" % trial)
		def save_image(filename):
		    p = PdfPages(filename)
		    fig = plt.figure(1)
		    fig.savefig(p, format='pdf') 
		    p.close()

		filename = "KerasTuner_CIFAR100_progress_with_reg_line.pdf"
		save_image(filename)

		# plot points too
		plt.scatter(x, history, s=20)

		def save_image(filename):
		    p = PdfPages(filename)
		    fig = plt.figure(1)
		    fig.savefig(p, format='pdf') 
		    p.close()

		filename = "KerasTuner_CIFAR100_progress_with_reg__with_points.pdf"
		save_image(filename)


		plt.show(block=True), plt.close()
		plt.close('all')




	model_num = "1_with_reg"

	# writing data to excel file
	data = [[test_loss, train_loss, model_num, max_trials, time_lapsed, seed]]

	with open('../KT_RandomSearch_CIFAR100_Benchmark_with_regularization.csv', 'a', newline = '') as file:
	    writer = csv.writer(file)
	    writer.writerows(data)





# use argparse to parse if with regularization or not
# python3 -m KT_RandomSearch_CIFAR100_Benchmark_with_regularization --with_reg

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--with_reg", action="store_true")
	args = parser.parse_args()
	model = model.build_model(with_reg=args.with_reg)
	train_model(model)

if __name__ == "__main__":
	main()




# model
def build_model(hp, with_reg):
	hp_reg = hp.Float("reg_term", min_value=1e-5, max_value=1e-1)
    regularizer = tf.keras.regularizers.l2(l=hp_reg) if with_reg else None

	model = keras.Sequential()
	model.add(tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3), kernel_regularizer=regularizer))
	model.add(tf.keras.layers.Conv2D(64,  kernel_size = 3, activation='relu', kernel_regularizer=regularizer))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(128,  kernel_size = 3, activation='relu', kernel_regularizer=regularizer))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64,  kernel_size = 3, activation='relu', kernel_regularizer=regularizer))
	model.add(tf.keras.layers.MaxPooling2D((4, 4)))

	model.add(tf.keras.layers.Flatten())


	model.add(tf.keras.layers.Dense(256, activation = "relu", kernel_regularizer=regularizer))
	model.add(tf.keras.layers.Dense(100, activation = "softmax"))

	hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

	model.compile(
	    optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
	    loss=keras.losses.SparseCategoricalCrossentropy(),
	    metrics=["accuracy"],
	)

	return model
 
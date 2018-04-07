import tensorflow as tf

reader=tf.train.NewCheckpointReader("save/model.ckpt")

variables = reader.get_variable_to_shape_map()

for element in variables:
    print(element)

import tensorflow as tf

def load_dataset(path, img_size=(64, 64), batch_size=32):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )
    class_names = dataset.class_names
    return dataset, class_names

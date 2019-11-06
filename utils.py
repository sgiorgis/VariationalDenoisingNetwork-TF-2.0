import tensorflow as tf


def manage_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.debugging.set_log_device_placement(True)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                          tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

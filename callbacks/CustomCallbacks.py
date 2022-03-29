import tensorflow as tf


class CustomCallbacks(tf.keras.callbacks.Callbackcallback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:  # @KEEP
            print("\nReached 99% accuracy so cancelling training!")

            # Stop training once the above condition is met
            self.model.stop_training = True
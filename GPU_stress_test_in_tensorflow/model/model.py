import tensorflow as tf


class Tyan_korea:
    """
    A trainable version VGG19.
    """

    def __init__(self,num_gpu,batch_size):
        self.batch_size=batch_size
        self.num_gpu=num_gpu

    def inference(self,x):
        x=tf.layers.conv2d(x,512,(3,3),padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same')
        return x
    
    

    def test(self, intput):


        self.mb_per_gpu = int(self.batch_size / self.num_gpu)

        output_list=[]

        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_idex in range(self.num_gpu):
                with tf.device('/gpu:' + str(gpu_idex)):
                    mb_start = self.mb_per_gpu * gpu_idex
                    mb_end = self.mb_per_gpu * (gpu_idex + 1)
                    input_img = intput[mb_start:mb_end]
                    output = self.inference(input_img)
                    output_list.append(output)

        return output_list



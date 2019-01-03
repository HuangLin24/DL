import tensorflow as tf
import numpy as np

class VGG16:
    def __init__(self,imgsPlaceHolder):
        self.imgs=imgsPlaceHolder
        self.convLayer()
        self.fcLayer()
        self.outPut=self,fc_3



    def w_variable(shape):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    def b_variable(shape):
        return tf.Variable(tf.constant(0.1,shape=shape,dtype=tf.float32))

    def convLayer(self):
        with tf.name_scope('conv1_1') as scope:
            conv = tf.nn.conv2d(imgs,w_variable([3,3,3,64]),[1,1,1,1],padding='SAME')
            biases=b_variable([64])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv1_1=tf.nn.relu(linearout,name=scope)

        
        with tf.name_scope('conv1_2') as scope:
            conv=tf.nn.conv2d(self.conv1_1,w_variable([3,3,64,64]),[1,1,1,1],padding='SAME')
            biases=b_variable([64])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv1_2=tf.nn.relu(linearout,name=scope)

        self.pool_1=tf.nn.max_pool(self.conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool_1')




        with tf.name_scope('conv2_1') as scope:
            conv=tf.nn.conv2d(self.pool_1,w_variable([3,3,64,128],[1,1,1,1],padding='SAME'))
            biases=b_variable([128])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv2_1=tf.nn.relu(linearout,name=scope)

        with tf.name_scope('conv2_2') as scope:
            conv=tf.nn.conv2d(self.conv2_1,w_variable([3,3,128,128],[1,1,1,1],padding='SAME'))
            biases=b_variable([128])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv2_2=tf.nn.relu(linearout,name=scope)

        self.pool_2=tf.nn.max_pool(self.conv2_2,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name='pool_2')




        with tf.name_scope('conv3_1') as scope:
            conv=tf.nn.conv2d(self.pool_2,w_variable([3,3,128,256],[1,1,1,1],padding='SAME'))
            biases=b_variable([256])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv3_1=tf.nn.relu(linearout,name=scope)

        with tf.name_scope('conv3_2') as scope:
            conv=tf.nn.conv2d(self.pool_1,w_variable([3,3,256,256],[1,1,1,1],padding='SAME'))
            biases=b_variable([256])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv3_2=tf.nn.relu(linearout,name=scope)

        self.pool_3=tf.nn.max_pool(self.conv3_2,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name='pool_3')




        with tf.name_scope('conv4_1') as scope:
            conv=tf.nn.conv2d(self.pool_3,w_variable([3,3,256,512],[1,1,1,1],padding='SAME'))
            biases=b_variable([512])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv4_1=tf.nn.relu(linearout,name=scope)

        with tf.name_scope('conv4_2') as scope:
            conv=tf.nn.conv2d(self.conv4_1,w_variable([3,3,512,512],[1,1,1,1],padding='SAME'))
            biases=b_variable([512])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv4_2=tf.nn.relu(linearout,name=scope)

        with tf.name_scope('conv4_3') as scope:
            conv=tf.nn.conv2d(self.conv4_2,w_variable([3,3,512,512]),padding='SAME')
            biases=b_variable([512])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv4_3=tf.nn.relu(linearout,name=scope)

        self.pool_4=tf.nn.max_pool(self.conv4_3,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name='pool_4')





        with tf.name_scope('conv5_1') as scope:
            conv=tf.nn.conv2d(self.pool_3,w_variable([3,3,512,512],[1,1,1,1],padding='SAME'))
            biases=b_variable([512])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv5_1=tf.nn.relu(linearout,name=scope)

        with tf.name_scope('conv5_2') as scope:
            conv=tf.nn.conv2d(self.conv5_1,w_variable([3,3,512,512],[1,1,1,1],padding='SAME'))
            biases=b_variable([512])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv5_2=tf.nn.relu(linearout,name=scope)

        with tf.name_scope('conv5_3') as scope:
            conv=tf.nn.conv2d(self.conv5_2,w_variable([3,3,512,512]),padding='SAME')
            biases=b_variable([512])
            linearout=tf.nn.bias_add(conv,biases)
            self.conv5_3=tf.nn.relu(linearout,name=scope)

        self.pool_5=tf.nn.max_pool(self.conv5_3,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name='pool_5')


    def fcLayer(self):
        with tf.name_scope('fc1') as scope:
            inputShape=int(np.prod(self.pool_5.get_shap()[1:]))
            W=w_variable([inputShape,4096])
            B=b_variable([4096])
            pool_5_flatten=tf.reshape(self.pool_5[-1:inputShape])
            linearout=tf.nn.bias_add(tf.matmul(pool_5_flatten,W),B)
            self.fc_1=tf.nn.relu(linearout)





        with tf.name_scope('fc2') as scope:
            W=w_variable([4096,4096])
            B=b_variable([4096])
            linearout=tf.nn.bias_add(tf.matmul(self.fc_1,W),B)
            self.fc_2=tf.nn.relu(linearout)



        with tf.name_scope('fc3') as scope:
            W=w_variable([4096,1000])
            B=b_variable([4096])
            linearout=tf.nn.bias_add(tf.matmul(self.fc_2,W),B)
            self.fc_3=tf.nn.relu(linearout)



if __name__ == '__main__':

    imgInput=None
    labelInput=None

    imgsDim=224
    labelDim=10

    learningRate=0.01
    traningNum=1024



    sess=tf.Session()
    imgsHolder=tf.placeholder("float",[None,imgsDim,imgsDim,3])
    labelHolder=tf.placeholder("float",shape=[None,labelDime])


    vgg=VGG16(imgsHolder)


    W=tf.Variable(tf.truncated_normal([1000],stddev=0.1))
    B=tf.Variable(tf.constant(0.1,shape=[labelDim],dtype=tf.float32))


    sess.run(tf.initialize_all_variables())

    label = tf.nn.softmax(tf.matmul(vgg.outPut,W)+B)

    #crossEntropy=-tf.reduce_sum(labelHolder*tf.log(label))
    #y_*log(y)+(1-y)*log(1-y_)
    #this function may be better
    crossEntropy = -tf.reduce_sum(labelHolder*tf.log(label)+(1-labelHolder)*tf.log(1-label))


    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)

    for i in range(traningNum):
        train_step.run(feed_dict={imgsHolder:imgInput,labelHolder:labelInput})













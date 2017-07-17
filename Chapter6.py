import tensorflow as tf

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

X = tf.placeholder('float',[None,4])
Y = tf.placeholder('float',[None,3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

#Cross-Entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer,feed_dict={X: x_data, Y:y_data})
        if step % 200 == 0:
            print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}))

    all = sess.run(hypothesis,feed_dict={X : [[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
    print(all,sess.run(tf.arg_max(all,1)))



##########동물 데이터 분류##########################
import numpy as np
xy = np.loadtxt(r'C:\Users\stu\Downloads\zoo2.csv',delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
nb_classes = 7 #0~6이니까 7개
X = tf.placeholder(tf.float32,[None,16])
Y = tf.placeholder(tf.int32,[None,1])

Y_one_hot = tf.one_hot(Y,nb_classes)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes])

W = tf.Variable(tf.random_normal([16,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

pred = tf.argmax(hypothesis,1)
correct_pred = tf.equal(pred,tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer,feed_dict={X : x_data, Y : y_data})
        if step % 100 == 0:
            loss,acc = sess.run([cost,accuracy],feed_dict={X:x_data, Y:y_data})
            print('Step : {:5}\t Loss:{:.3f} \t Acc:{:.2%}'.format(step,loss,acc))

    predict = sess.run(pred,feed_dict={X:x_data})
    for p,y in zip(predict,y_data.flatten()):
        print('[{}] Prediction : {} True Y : {}'.format(p==int(y),p,int(y)))
#!/usr/bin/env python3
"""Trains a loaded neural network model using mini-batch gradient descent"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent
    """
    # Start a TensorFlow session to run the training operations
    with tf.Session() as sess:
        # Load the model graph and restore the session
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        m, nx = X_train.shape

        # Loop over epochs
        for epoch in range(epochs):
            # Shuffle data
            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            # Loop over batches
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = shuffled_X[i:i + batch_size]
                Y_batch = shuffled_Y[i:i + batch_size]

                # Train your model
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # Print cost and accuracy on current
                # mini-batch every 100 steps
                if i % 100 == 0:
                    step_cost, step_accuracy = sess.run(
                        [loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {0}: ".format(i//batch_size+1))
                    print("\t\tCost={0:.4f}".format(step_cost))
                    print("\t\tAccuracy={0:.4f}".format(step_accuracy))
            # Print cost and accuracy on entire training and validation sets after each epoch
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={
                                                  x: shuffled_X, y: shuffled_Y})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print("After {0} epochs:".format(epoch+1))
            print("\tTraining Cost={0:.4f}".format(train_cost))
            print("\tTraining Accuracy={0:.4f}".format(train_accuracy))
            print("\tValidation Cost={0:.4f}".format(valid_cost))
            print("\tValidation Accuracy={0:.4f}".format(valid_accuracy))

        # Save session
        save_path = saver.save(sess, save_path)
        print("Model saved in path: {0}".format(save_path))

    return save_path

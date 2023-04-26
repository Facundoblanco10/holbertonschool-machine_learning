#!/usr/bin/env python3
"""Trains a loaded neural network model using mini-batch gradient descent"""
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

            # Print cost and accuracy on entire training and
            # validation sets after each epoch
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={
                x: shuffled_X, y: shuffled_Y})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print("After {0} epochs:".format(epoch))
            print("\tTraining Cost={0:.4f}".format(train_cost))
            print("\tTraining Accuracy={0:.4f}".format(train_accuracy))
            print("\tValidation Cost={0:.4f}".format(valid_cost))
            print("\tValidation Accuracy={0:.4f}".format(valid_accuracy))

            # Calculate the number of batches
            if m % batch_size == 0:
                n_batches = m // batch_size
            else:
                n_batches = m // batch_size + 1
            # Train the model using mini-batches
            for b in range(n_batches):
                start = b * batch_size
                end = (b + 1) * batch_size
                if end > m:
                    end = m
                X_mini_batch = shuffled_X[start:end]
                Y_mini_batch = shuffled_Y[start:end]

                # Define the next mini-batch
                next_train = {x: X_mini_batch, y: Y_mini_batch}

                # Run a training step with the mini-batch
                sess.run(train_op, feed_dict=next_train)

                # Print the mini-batch results every 100 batches
                if (b + 1) % 100 == 0 and b != 0:
                    loss_mini_batch = sess.run(loss, feed_dict=next_train)
                    acc_mini_batch = sess.run(accuracy,
                                              feed_dict=next_train)
                    print("\tStep {}:".format(b + 1))
                    print("\t\tCost: {}".format(loss_mini_batch))
                    print("\t\tAccuracy: {}".format(acc_mini_batch))

        # Save session
        save_path = saver.save(sess, save_path)

    return save_path

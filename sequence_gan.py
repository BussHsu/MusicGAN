import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT_OLD
import os
from config import *


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, gen_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(gen_lstm.pretrain_loss, {gen_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    # vocab_size = 97
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,learning_rate=0.01)


    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution

    gen_data_loader.create_batches(positive_file)


    #  pre-train generator
    print 'Start pre-training...'

    early_stop_buffer = [10.]*5
    for e_cnt, epoch in enumerate(xrange(PRE_EPOCH_NUM)):
        loss = pre_train_epoch(sess, generator, gen_data_loader)

        if epoch % 2 == 0:
            # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_real_file)
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            early_stop_buffer = early_stop_buffer[1:]
            early_stop_buffer.append(test_loss)
            if all(early_stop_buffer[0] < np.asarray(early_stop_buffer[1:])):
                break

    print 'Start pre-training discriminator...'
    # Train 3 epoch on the generated data and do this for 50 times
    for e in range(50):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)
        print 'Epoch {}'.format(e)
    rollout = ROLLOUT_OLD(generator, 0.8)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    model_idx = 1
    fname = 'model' + str(model_idx)
    model_save_path = './Model/' + fname + '/'

    while os.path.exists(model_save_path):
        model_idx += 1
        fname = 'model' + str(model_idx)
        model_save_path = './Model/' + fname + '/'

    os.makedirs(model_save_path)
    os.makedirs(os.path.join('./log', fname))


    early_stop_buffer = [10.] * 4
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, SAMP_NUM, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 3 == 0 or total_batch == TOTAL_BATCH - 1:
            # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_real_file)
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss

            early_stop_buffer = early_stop_buffer[1:]
            early_stop_buffer.append(test_loss)
            if all(early_stop_buffer[0] < np.asarray(early_stop_buffer[1:])):
                break

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(1):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(1):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

        saver.save(sess, os.path.join(model_save_path, fname), global_step=total_batch, write_meta_graph=False)

        metagraph_filename = os.path.join(model_save_path, fname + '.meta')

        if not os.path.exists(metagraph_filename):
            saver.export_meta_graph(metagraph_filename)



if __name__ == '__main__':
    main()

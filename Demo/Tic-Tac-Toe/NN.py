import tensorflow as tf
import numpy as np
import csv
import random
import argparse
import sys
import os

batch_size = 100
trans_type = ['rotate180','rotate90','rotate270','flip_v','flip_h']
data_file = 'base_tic_tac_toe_moves.csv'
model_dir = 'Models'
ckpt_file = '{}/model.ckpt'.format(model_dir)

# print board (1D array with 9 elements)
def print_board(board):
    symbols = [' ','X','O'] # 0:blank   -1:O    +1:X
    print(' ' + symbols[board[0]] + '|' + symbols[board[1]] + '|' + symbols[board[2]])
    print('-------')
    print(' ' + symbols[board[3]] + '|' + symbols[board[4]] + '|' + symbols[board[5]])
    print('-------')
    print(' ' + symbols[board[6]] + '|' + symbols[board[7]] + '|' + symbols[board[8]])

# generate symmetry board
def get_symmetry(board,response,transformation):
    if transformation == trans_type[0]:
        return (board[::-1],8-response)
    elif transformation == trans_type[1]:
        new_response = [6,3,0,7,4,1,8,5,2].index(response)
        tuple_board = zip(*[board[6:9],board[3:6],board[0:3]])
        return ([value for item in tuple_board for value in item],new_response)
    elif transformation == trans_type[2]:
        new_response = [2,5,8,1,4,7,0,3,6].index(response)
        tuple_board = zip(*[board[0:3][::-1],board[3:6][::-1],board[6:9][::-1]])
        return ([value for item in tuple_board for value in item],new_response)
    elif transformation == trans_type[3]:
        new_response = [6,7,8,3,4,5,0,1,2].index(response)
        return (board[6:9]+board[3:6]+board[0:3],new_response)
    elif transformation == trans_type[4]:
        new_response = [2,1,0,5,4,3,8,7,6].index(response)
        new_board = board[::-1]
        return (new_board[6:9]+new_board[3:6]+new_board[0:3],new_response)
    else:
        raise ValueError("Not defined transformation type!")

# get tic tac toe moves
def get_moves_from_csv():
    moves = []
    with open(data_file,'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        # this csv file doesn't have csv header
        for row in csvreader:
            moves.append((row[0:9],int(row[9])))
    return moves

# get symmetry with a random transformation type
def get_move_via_rand_trans(moves):
    (board,response) = random.choice(moves)
    transformation = random.choice(trans_type)
    return get_symmetry(board,response,transformation)

# init variable
def init_var(shape):
    return tf.Variable(tf.random_normal(shape))

# build model
def build_model(X,Y,keep_prob):
    # first layer
    A1 = init_var(shape=[9,64])
    B1 = init_var(shape=[64])
    layer1 = tf.nn.sigmoid(tf.matmul(X,A1)+B1)

    # second layer
    A2 = init_var(shape=[3,3,1,16])
    B2 = init_var(shape=[16])
    layer1_reshape = tf.reshape(layer1,shape=[-1,8,8,1])
    layer2_conv = tf.nn.relu(tf.nn.conv2d(layer1_reshape,A2,strides=[1,1,1,1],padding='SAME')+B2)
    layer2 = tf.nn.max_pool(layer2_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # third layer
    layer2_reshape = tf.reshape(layer2,shape=[-1,4*4*16])
    layer3 = tf.nn.dropout(layer2_reshape,keep_prob)

    # fourth layer
    A4 = init_var(shape=[4*4*16,9])
    B4 = init_var(shape=[9])
    model_out = tf.nn.relu(tf.matmul(layer3,A4)+B4)
    return model_out

# fight
def fight(model_out,sess,X,keep_prob):
    game_tracker = [0] * 9
    is_win = False
    num_moves = 0
    print_board(game_tracker)
    while not is_win:
        # move by player
        player_index = input('Input index of your move(0-8)(indicate by X): ')
        allowed_list = [ix for ix,x in enumerate(game_tracker) if x == 0]
        if player_index not in allowed_list:
            print("Invalid move!")
            continue
        num_moves += 1
        game_tracker[int(player_index)] = 1
        print("You have moved!")

        # move by machine
        result_list = sess.run(model_out,feed_dict={X:[game_tracker],keep_prob:1.0})[0]
        allowed_list = [ix for ix,x in enumerate(game_tracker) if x == 0]
        machine_index = np.argmax([x if ix in allowed_list else -999 for ix,x in enumerate(result_list)])
        game_tracker[int(machine_index)] = -1
        print('Machine has moved!')

        # print board
        print_board(game_tracker)
        if check(game_tracker) or num_moves>=5:
            print("Game over! You Win!")
            is_win = True

# main
def main(_):
    # get moves data
    moves = get_moves_from_csv()

    # create totalset
    totalset = [] + moves
    totalsize = 500
    extrasize = totalsize - len(totalset)
    for i in range(extrasize):
        totalset.append(get_move_via_rand_trans(moves))

    # create trainset from totalset
    trainset = totalset[0:int(0.8*len(totalset))]

    # create testset from totalset
    testset = totalset[int(0.8*len(totalset)):]

    with tf.Graph().as_default():
        # place holders
        X = tf.placeholder(shape=[None,9],dtype=tf.float32)
        Y = tf.placeholder(shape=[None],dtype=tf.int64)
        keep_prob = tf.placeholder(tf.float32)

        # build model
        model_out = build_model(X,Y,keep_prob)

        # loss function
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_out,labels=Y))

        # build train_step
        train_step = tf.train.GradientDescentOptimizer(0.8).minimize(loss)

        # build accuracy
        prediction =tf.argmax(model_out,1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,Y),tf.float32))

        with tf.Session() as sess:
            # init
            init = tf.global_variables_initializer()
            sess.run(init)

            # train model
            if not FLAGS.restore:
                if os.path.exists(model_dir):
                    try:
                        os.rmdir(model_dir)
                    except OSError:
                        tf.gfile.DeleteRecursively(model_dir)
                os.mkdir(model_dir)
                for i in range(FLAGS.max_step):
                    # get batch data
                    rand_indices = np.random.choice(len(trainset),batch_size,replace=False)
                    batchx = [trainset[x][0] for x in rand_indices]
                    batchy = [trainset[x][1] for x in rand_indices]
                    sess.run(train_step,feed_dict={X:batchx,Y:batchy,keep_prob:0.5})
                    if (i+1)%1000 == 0:
                        loss_value = sess.run(loss,feed_dict={X:batchx,Y:batchy,keep_prob:1.})
                        accuracy_value = sess.run(accuracy,feed_dict={X:batchx,Y:batchy,keep_prob:1.})
                        print("Step:{} loss:{} accuracy:{}".format(i,loss_value,accuracy_value))
                # save model
                tf.train.Saver().save(sess,ckpt_file,FLAGS.max_step)

            # restore
            else:
                ckpt = tf.train.get_checkpoint_state(model_dir)
                tf.train.Saver().restore(sess,ckpt.model_checkpoint_path)

            # test
            testx = [x[0] for x in testset]
            testy = [x[1] for x in testset]
            test_accuracy = sess.run(accuracy,feed_dict={X:testx,Y:testy,keep_prob:1.})
            print("Final test accuracy:{}".format(test_accuracy))

            # prediction
            predictx = [[-1,0,0,1,-1,-1,0,0,1]]
            truthy = 6
            predicty = sess.run(prediction,feed_dict={X:predictx,keep_prob:1.})
            print("prediction:{}".format(predicty))

            # fight
            if FLAGS.fight:
                fight(model_out,sess,X,keep_prob)

# check if board is win
def check(board):
    wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for i in range(len(wins)):
        if board[wins[i][0]] == board[wins[i][1]] == board[wins[i][2]] == 1:
            return True
        elif board[wins[i][0]] == board[wins[i][1]] == board[wins[i][2]] == -1:
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--restore',
            type=bool,
            default=False,
            help='restore model.ckpt')
    parser.add_argument(
            '--max_step',
            type=int,
            default=20000,
            help='max train step')
    parser.add_argument(
            '--fight',
            type=bool,
            default=False,
            help='fight with machine')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)






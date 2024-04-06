import numpy as np
import tensorflow as tf
from numpy import random
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.models import Model
from GoEnvironment import GoGame
elo_ratings1 = 1035
elo_ratings2 = 1000


def count_territory(board):
    """计算各玩家领地"""
    player1_territory = 0
    player2_territory = 0
    neutral = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 1:
                player1_territory += 1
            elif board[i, j] == 2:
                player2_territory += 1
            else:
                neutral += 1
    return neutral, player1_territory, player2_territory

# 定义神经网络模型
def create_model(board_size):
    # 输入层
    input_shape = (board_size, board_size, 1)
    inputs = Input(shape=input_shape)

    # 卷积层堆叠
    conv1 = Conv2D(256, kernel_size=(1, 1), activation='relu')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(256, kernel_size=(1, 1), activation='relu')(bn1)
    bn2 = BatchNormalization()(conv2)
    flatten = Flatten()(bn2)

    # 分离出策略网络和价值网络
    policy_head = Dense(board_size * board_size, activation='softmax')(flatten)  # 输出下一步落子概率分布
    value_head = Dense(1, activation='tanh')(flatten)  # 输出当前局面的胜率估计（归一化至[-1, 1]）

    model = Model(inputs=inputs, outputs=[policy_head, value_head])

    return model

# 定义全局训练数据容器
train_data = []
win1 = 0
total = 0

# 修改后的自训练函数
def self_play(board_size, elo_ratings1, elo_ratings2, num_games=100, batch_size=1000, exploration_rate_start=1.0, exploration_rate_end=0.1, exploration_decay_steps=1000):
    global train_data
    global win1, total

    exploration_rate = exploration_rate_start
    exploration_decay_rate = (exploration_rate_start - exploration_rate_end) / exploration_decay_steps

    for _ in range(num_games):
        game = GoGame(board_size)
        current_player = 1
        history = []
        
        while not game.is_game_over():
            board_state = np.expand_dims(np.expand_dims(game.board, axis=0), axis=3)
            predictions = getmodel(current_player).predict(board_state, verbose=0)
            policy_output = predictions[0]
            value_output = predictions[1]

            print(predictions)
            # 使用policy_output中的pass概率
            probabilities = value_output[:-1]
            pass_probability = policy_output[-1]
            
            # 组合包含pass的合法走法概率向量
            legal_probabilities = np.concatenate([probabilities, [pass_probability]])
       
            legal_moves = game.get_legal_moves(current_player) + [(-1, -1)]
            
            # 引入ε-贪心策略，根据探索率随机选择动作或按概率分布选择
            if random.uniform(0, 1) < exploration_rate:
                move_index = random.randint(0, len(legal_moves))
            else:
                move_index = np.argmax(legal_probabilities)
            move = legal_moves[move_index]
            print(legal_moves)
            print(legal_probabilities)
            
            # 更新探索率
            exploration_rate -= exploration_decay_rate if exploration_rate > exploration_rate_end else 0
            
            # 记录当前棋步及其概率分布
            history.append({
                'board': game.board,
                'move': move,
                'probabilities': probabilities,
                'legal_probabilities': legal_probabilities,
                'current_player': current_player
            })
            
            game.make_move(move, current_player)
            current_player = 3 - current_player  # 切换到下一个玩家
        
        winner = game.get_winner()
        if winner == 0:
            result = 0.5  # 平局
        else:
            result = 2 - winner # 胜者索引
        win1 += result
        total += 1
        
        elo_ratings1, elo_ratings2 = calculate_new_elo(elo_ratings1, elo_ratings2, result)
        if total % 200 == 0:
            print("current model1 winrate:", win1 / total, "totalgames: ", total)
        
        # 将本次对局的历史数据添加到全局训练数据容器
        train_data.extend(history)
        
        # 当积累的对局数据达到batch_size时，进行一次模型训练
        if len(train_data) >= batch_size:
            train_batch = train_data[:batch_size]
            train_data = train_data[batch_size:]

            X_train, y_policy_train, y_value_train = prepare_training_data_with_heads(train_batch)
            X_train = np.reshape(X_train, (-1, board_size, board_size, 1))

            # 分别训练policy_head和value_head
            policy_loss, policy_accuracy = getmodel(1).train_on_batch(X_train, {'policy_head': y_policy_train})
            value_loss, _ = getmodel(1).train_on_batch(X_train, {'value_head': y_value_train})

            print(f"Policy loss: {policy_loss}, Policy accuracy: {policy_accuracy}")
            print(f"Value loss: {value_loss}")

    return elo_ratings1, elo_ratings2
def needprint():
    return False
    # return True
def prepare_training_data(history):
    """
    将历史数据转化为模型训练所需的输入（棋盘状态）和标签（走法概率分布）。
    """
    X = []
    y = []

    for record in history:
        X.append(np.expand_dims(np.expand_dims(record['board'], axis=0), axis=3))
        y.append(record['probabilities'])

    X = np.array(X)
    y = np.array(y)

    return X, y
def calculate_new_elo(A_elo, B_elo, result_A, K=20):
    def expected_probability(player_elo, opponent_elo):
        return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

    E_A = expected_probability(A_elo, B_elo)
    E_B = expected_probability(B_elo, A_elo)

    if result_A == 1:  # A胜
        result_B = 0
    elif result_A == 0.5:  # 平局
        result_B = 0.5
    else:  # B胜
        result_B = 1

    NewElo_A = A_elo + K * (result_A - E_A)
    NewElo_B = B_elo + K * (result_B - E_B)

    return NewElo_A, NewElo_B

def update_model(model, optimizer, moves_history, winner, discount_factor=0.9):
    # 将历史数据转化为监督学习格式
    X, y_policy, y_value = preprocess_data(moves_history, winner, discount_factor)

    # 训练模型
    model.train_on_batch(X, [y_policy, y_value])
    
def train(model, optimizer, board_size, max_moves, num_self_plays, temp_anneal_schedule=None):
    for i in range(num_self_plays):
        if temp_anneal_schedule:
            temperature = temp_anneal_schedule(i)  # 可选：根据进度调整探索温度
        else:
            temperature = 1.0

        moves_history, winner = self_play(model, board_size, max_moves, temperature)

        update_model(model, optimizer, moves_history, winner)

def getmodel(player):
    if player == 1:
        return model1
    else:
        return model2
    
def prepare_training_data_with_heads(history):
    """
    返回经过处理的训练数据，以及对应的policy和value标签。
    这里假设y_policy_train是one-hot编码的合法走法，y_value_train是游戏结果（如胜/负/平局的标量值）
    """
    X, y_policy, y_value = [], [], []

    for data_point in history:
        X.append(data_point['board'])
        y_policy.append(to_one_hot(data_point['move'], board_size * board_size + 1))  # 假设to_one_hot函数将落子位置转换为one-hot编码，包括pass
        y_value.append(data_point['result'])  # 假设data_point['result']为游戏结果（如胜/负/平局的标量值）

    return np.array(X), np.array(y_policy), np.array(y_value)

# 测试
board_size = 3
# model1 = load_model("model31n.h5")
model1 = create_model(board_size)
model2 = create_model(board_size)
elo_ratings1, elo_ratings2 = self_play(board_size, elo_ratings1, elo_ratings2, 2000)
print("Final Elo rating:", elo_ratings1, elo_ratings2)
model1.save("model31nn.h5")
model2.save("model32nn.h5")
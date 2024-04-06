import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

import numpy as np
from Utils import needprint
GO_BOARD_SIZE = 3

class GoBoard:
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    def __init__(self, board_size):
        self.size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.ko_move = None
        self.history = []
        self.pass_count = 0
        self.current_player = 1

    def get_legal_moves(self, player):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.is_legal_move((i, j), player)] + [(-1, -1)]

    def is_legal_move(self, move, player):
        if move[0] < 0 or move[0] >= self.size or move[1] < 0 or move[1] >= self.size:
            return False  # 超出棋盘范围

        if self.board[move[0]][move[1]] != 0:
            return False  # 位置已有棋子

        # 新增：检查是否为自杀禁手
        new_board = self.board.copy()
        new_board[move[0]][move[1]] = player
        own_group, own_liberties = self.find_group_with_liberties(move, player=new_board[move[0]][move[1]])
        print("move: ", move, "own_liberties: ", own_liberties)
        if own_liberties == 0:
            return False  # 自杀禁手，不允许下子

        return True  # 合法的移动
    
    def pass_move(self, player):
        """
        玩家主动 pass，即使当前存在合法走法。
        """
        if self.is_game_over():
            raise ValueError("Game is already over.")

        self.pass_count += 1  # 增加 pass 计数器

        if self.is_game_over():
            winner = self.get_winner()
            if needprint():
                if winner == 0:
                    print("Game over. It's a draw.")
                else:
                    print("Game over. Player", winner, "wins!")

    def make_move(self, move, player):
        if move == (-1, -1):  # 使用特殊坐标 (-1, -1) 表示 pass
            self.pass_move(player)
            print("Player", player, "pass")
            return
        
        if not self.is_legal_move(move, player):
            raise ValueError("Illegal move")

        self.board[move[0]][move[1]] = player

        # 检测提子条件
        affected_groups = self.find_affected_enemy_groups(move, player)
        captured_stones = []

        # 执行提子及处理连环提子
        while affected_groups:
            group = affected_groups.pop()
            captured_stones.extend(group)
            self.remove_group_from_board(group)

            new_affected_groups = self.find_newly_unliberated_enemy_groups(move, player)
            affected_groups.extend(new_affected_groups)

        self.ko_move = self.check_ko(move, player)
        if needprint():
            print("Player", player, "moves at", move)
            self.print_board()
        if self.is_game_over():
            winner = self.get_winner()
            if needprint():
                if winner == 0:
                    print("Game over. It's a draw.")
                else:
                    print("Game over. Player", winner, "wins!")
            return

    def find_affected_enemy_groups(self, move, player):
        """
        返回因当前落子而失去全部气的敌方棋子组。
        """
        enemy_player = 3 - player
        affected_groups = []

        # 检查当前位置周围的敌方棋子组
        for neighbor in self.get_adjacent_neighbors(move):
            if self.board[neighbor[0]][neighbor[1]] == enemy_player:
                neighbor_group, liberties = self.find_group_with_liberties(neighbor)
                if liberties == 0:
                    affected_groups.append(neighbor_group)

        return affected_groups

    def find_newly_unliberated_enemy_groups(self, move, player):
        """
        返回因提子而新产生的无气敌方棋子组。
        """
        enemy_player = 3 - player
        newly_unliberated_groups = []

        # 检查提子后受影响区域的敌方棋子组
        for i in range(max(0, move[0] - 1), min(self.size, move[0] + 2)):
            for j in range(max(0, move[1] - 1), min(self.size, move[1] + 2)):
                if self.board[i][j] == enemy_player:
                    group, liberties = self.find_group_with_liberties((i, j))
                    if liberties == 0 and group not in newly_unliberated_groups:
                        newly_unliberated_groups.append(group)

        return newly_unliberated_groups

    def remove_group_from_board(self, group):
        """
        从棋盘上移除指定棋子组。
        """
        for stone in group:
            self.board[stone[0]][stone[1]] = 0

    def get_adjacent_neighbors(self, move):
        neighbors = [(move[0] + 1, move[1]), (move[0] - 1, move[1]), (move[0], move[1] + 1), (move[0], move[1] - 1)]
        return [(i, j) for i, j in neighbors if 0 <= i < self.size and 0 <= j < self.size]

    def find_group_with_liberties(self, start, player=None):
        if player is None:
            player = self.board[start[0]][start[1]]

        group = [start]
        liberties = []

        visited = set()

        stack = [start]

        while stack:
            stone = stack.pop()
            visited.add(stone)

            # Check adjacent neighbors for liberties and same-color stones
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = stone[0] + dx, stone[1] + dy

                if 0 <= ni < self.size and 0 <= nj < self.size and self.board[ni][nj] == 0:
                    liberties.append((ni, nj))  # Found a liberty
                elif 0 <= ni < self.size and 0 <= nj < self.size and self.board[ni][nj] == player and (ni, nj) not in visited:
                    group.append((ni, nj))  # Add same-color stone to the group
                    stack.append((ni, nj))  # Explore this stone in the next iteration

        if not self.is_valid_group(group, player):
            raise ValueError("Invalid group found. Possible logic error.")

        return group, len(liberties)

    def is_valid_group(self, group, player):
        # 检查组内是否有对方颜色的棋子，暂时实现为简单检查，可能需要更复杂的逻辑
        # for stone in group:
        #     if self.board[stone[0]][stone[1]] != player:
        #         return False
        return True

    def check_ko(self, move, player):
        # 检查劫争
        if len(self.get_adjacent_neighbors(move)) == 4 and self.board[move[0]][move[1]] == player:
            return move
        return None

    def is_game_over(self):
        if self.pass_count >= 2:  # 双方连续 pass 两次，游戏结束
            return True
        if np.count_nonzero(self.board) == self.size * self.size:
            return True  # 棋盘被填满
        else:
            return False

    def get_winner(self):
        # 计算领地
        territory = count_territory(self.board)
        player1_territory, player2_territory = territory[1], territory[2]
        if player1_territory > player2_territory:
            return 1  # 玩家1胜利
        elif player1_territory < player2_territory:
            return 2  # 玩家2胜利
        else:
            return 0  # 平局

    def print_board(self):
        for row in self.board:
            print(' '.join(map(str, row)))

    def clone(self):
        """
        Creates a deep copy of the current GoBoard instance.
        """
        cloned_board = GoBoard(self.size)  # Assuming there's a size parameter in the constructor
        cloned_board.board = self.board.copy()  # Assuming `board` is a 2D list representing the game state
        # Copy any other relevant attributes here, ensuring they are deep-copied if necessary

        return cloned_board
    
    def get_reward(self, player):
        """
        返回当前局面下给定玩家的奖励。

        参数：
        player (int): 玩家编号（1 或 2）

        返回值：
        float: 玩家的奖励分数（通常为正数表示胜，负数表示负，0表示平局）
        """
        winner = self.get_winner()

        if winner == 0:  # 平局
            return 0.0

        if winner == player:  # 获胜
            return 1.0  # 或者使用其他您认为合适的正数值表示胜利

        # 输棋
        return -1.0  # 或者使用其他您认为合适的负数值表示失败

def encode_board(board: GoBoard) -> np.ndarray:
    # Define one-hot encoding labels for each possible state on the board
    one_hot_labels = {
        GoBoard.EMPTY: np.array([1, 0, 0]),
        GoBoard.BLACK: np.array([0, 1, 0]),
        GoBoard.WHITE: np.array([0, 0, 1])
    }

    # Encode the board using one-hot encoding
    encoded_board = np.zeros((board.size, board.size, 3), dtype=np.float32)
    for row in range(board.size):
        for col in range(board.size):
            encoded_board[row, col] = one_hot_labels[board.board[row][col]]

    # Flatten the encoded board into a one-dimensional vector
    encoded_board_flat = encoded_board.reshape(-1)

    return encoded_board_flat

def decode_move(move: int, board_size: int) -> tuple:
    if move == board_size ** 2:
        return (-1, -1)
    row = move // board_size
    col = move % board_size
    return (row, col)

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
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


class GoStateDataset(Dataset):
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        return state, action, reward


class MonteCarloAgent:
    def __init__(self, exploration_rate=1.0, decay_factor=0.99, temperature=1.0,
                 model=None, learning_rate=1e-3, device='cpu'):
        self.exploration_rate = exploration_rate
        self.decay_factor = decay_factor
        self.temperature = temperature
        self.device = device

        if model is None:
            self.policy_net = PolicyNet(input_size=GO_BOARD_SIZE**2 * 3, hidden_size=128, output_size=GO_BOARD_SIZE**2 + 1).to(device)
        else:
            self.policy_net = model.to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def select_action(self, legal_moves, go_board, explore=True):
        while True:
            if explore and random.uniform(0, 1) < self.exploration_rate:
                if (-1,-1) in legal_moves:
                    move = random.choices(legal_moves, weights=[0.99, 0.01])  # 假设非pass动作有99%的概率被选中，pass有1%的概率被选中
                else:
                    move = random.choice(legal_moves)
            else:
                # 使用神经网络预测动作概率
                encoded_state = encode_board(go_board)
                state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                action_probs = self.policy_net(state_tensor).squeeze(0)                
                action_dist = torch.distributions.Categorical(probs=action_probs)
                move = decode_move(action_dist.sample().item(), GO_BOARD_SIZE)
            if move in legal_moves:
                return move
    def update_exploration_rate(self):
        self.exploration_rate *= self.decay_factor


    def run_simulation(self, go_board, current_player, max_depth=100):
        simulation_board = go_board.clone()
        steps_taken = 0

        while not simulation_board.is_game_over() and steps_taken < max_depth:
            legal_moves = simulation_board.get_legal_moves(current_player)
            print("legal:",legal_moves, "player:", current_player)
            move = self.select_action(legal_moves, simulation_board, explore=False)
            simulation_board.make_move(move, current_player)
            current_player = 3 - current_player  # Switch to opponent

            steps_taken += 1

        return simulation_board.get_winner()


    def train_episode(self, go_board, max_steps=500, num_simulations=1, save_after_training=True):
        total_reward = 0
        current_player = go_board.current_player
        game_over = False

        for step in range(max_steps):
            legal_moves = go_board.get_legal_moves(current_player)

            if not legal_moves: 
                go_board.pass_move(current_player) 
                current_player = 3 - current_player 
                continue
            # Perform multiple simulations for each move
            move_rewards = {move: 0 for move in legal_moves}
            for _ in range(num_simulations):
                for move in legal_moves:
                    simulation_board = go_board.clone()
                    simulation_board.make_move(move, current_player)
                    winner = self.run_simulation(simulation_board, 3 - current_player)
                    if winner == current_player:
                        move_rewards[move] += 1
                    elif winner == 0:  # Draw
                        move_rewards[move] += 0.5

            # Update exploration rate
            self.update_exploration_rate()

            # Choose best move based on average rewards from simulations
            best_move = max(move_rewards, key=move_rewards.get)
            go_board.make_move(best_move, current_player)

            # Update total reward (assuming binary win/loss/draw rewards)
            total_reward += move_rewards[best_move]
            current_player = 3 - current_player

            if go_board.is_game_over():
                game_over = True
                break
        if save_after_training == True:
            self.save_model('model.pth')
        return total_reward, game_over
    
    def play_games(self, num_games, num_games_per_epoch=1, batch_size=32, epochs=10):
        self.dataset = GoStateDataset()  # 初始化空数据集
        
        for i in range(num_games):
            self.play_game()

            if (i + 1) % num_games_per_epoch == 0:  # 每隔一定数量的游戏进行一次训练
                print(f"Epoch {i // num_games_per_epoch + 1}: Training model on collected data...")
                self.train_on_dataset(self.dataset, batch_size, epochs)  # 使用累积的经验数据训练模型

    def train_on_dataset(self, dataset, batch_size, epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            with tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch", leave=False) as t:
                for state, action, reward in t:
                    state = state.to(self.device)
                    action = action.to(self.device)
                    reward = reward.unsqueeze(-1).to(self.device)

                    # 训练一步
                    self.optimizer.zero_grad()
                    pred_action_probs = self.policy_net(state)
                    loss = -torch.sum(reward * pred_action_probs.log()[range(len(action)), action])
                    loss.backward()
                    self.optimizer.step()

                    t.set_postfix(loss=loss.item())
                
        self.save_model('model.pth')

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))

# 示例：创建一个MonteCarloAgent实例并训练一局，保存模型
agent = MonteCarloAgent()
go_board_instance = GoBoard(GO_BOARD_SIZE)
reward, game_over = agent.play_games(10)
print(f"Total reward: {reward}, Game over: {game_over}")
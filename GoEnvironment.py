import numpy as np
import copy

class GoGame:
    def __init__(self, board_size):
        self.size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.ko_move = None
        self.history = []
        self.pass_count = 0

    def get_legal_moves(self, player):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.is_legal_move((i, j), player)]

    def is_legal_move(self, move, player):
        if move[0] < 0 or move[0] >= self.size or move[1] < 0 or move[1] >= self.size:
            return False  # 超出棋盘范围

        if self.board[move[0]][move[1]] != 0:
            return False  # 位置已有棋子

        # 新增：检查是否为自杀禁手
        new_board = self.board.copy()
        new_board[move[0]][move[1]] = player
        own_group, own_liberties = self.find_group_with_liberties(move, player=new_board[move[0]][move[1]])
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
            # print("Player", player, "pass")
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
        queue = [start]
        while queue:
            stone = queue.pop()
            visited.add(stone)
            for neighbor in self.get_adjacent_neighbors(stone):
                if self.board[neighbor[0]][neighbor[1]] == 0:
                    liberties.append(neighbor)
                elif self.board[neighbor[0]][neighbor[1]] == player and neighbor not in visited:
                    group.append(neighbor)
                    queue.append(neighbor)
        
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

    def __init__(self, board_size=19, komi=7.5):
        self.board_size = board_size
        self.komi = komi
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for black, -1 for white
        self.ko_position = None
        self.game_over = False
        self.passes = 0

    def get_state(self):
        return np.copy(self.board)

    def take_action(self, action):
        if self.game_over or not self.is_valid_action(action):
            return self.get_state(), 0

        if action == 'pass':
            self.passes += 1
        else:
            row, col = action
            self.passes = 0  # Reset passes count

            # Check for Ko rule
            if action == self.ko_position:
                return self.get_state(), 0

            # Copy the current board for future Ko check
            prev_board = copy.deepcopy(self.board)

            self.board[row, col] = self.current_player
            self.current_player *= -1  # Switch player

            self.resolve_captures(action)

            # Check for Ko after resolving captures
            if np.array_equal(self.board, prev_board):
                self.ko_position = action
            else:
                self.ko_position = None

        self.check_game_over()

        return self.get_state(), 0

    def is_valid_action(self, action):
        if action == 'pass':
            return True

        row, col = action
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row, col] == 0

    def check_game_over(self):
        # Game over when both players pass consecutively or the board is full
        if self.passes >= 2 or np.all(self.board != 0):
            self.game_over = True

    def resolve_captures(self, action):
        opponent_color = -self.current_player
        captured_stones = self.get_captured_stones(action, opponent_color)

        for stone in captured_stones:
            self.board[stone] = 0

    def get_captured_stones(self, action, opponent_color):
        row, col = action
        captured_stones = []

        for neighbor in self.get_neighbors(action):
            group, liberties = self.get_group_and_liberties(neighbor, opponent_color)
            if liberties == 0:
                captured_stones.extend(group)

        return captured_stones

    def get_group_and_liberties(self, start, color):
        visited = set()
        group = []
        liberties = 0

        def dfs(point):
            if point in visited or self.board[point] != color:
                return
            visited.add(point)
            group.append(point)

            for neighbor in self.get_neighbors(point):
                if self.board[neighbor] == 0:
                    liberties += 1
                elif self.board[neighbor] == color:
                    dfs(neighbor)

        dfs(start)
        return group, liberties

    def get_neighbors(self, point):
        row, col = point
        neighbors = []

        if row > 0:
            neighbors.append((row - 1, col))
        if row < self.board_size - 1:
            neighbors.append((row + 1, col))
        if col > 0:
            neighbors.append((row, col - 1))
        if col < self.board_size - 1:
            neighbors.append((row, col + 1))

        return neighbors

    def get_legal_actions(self):
        legal_actions = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_valid_action((row, col)):
                    legal_actions.append((row, col))
        legal_actions.append('pass')  # Pass is always a legal action
        return legal_actions

    def calculate_score(self):
        black_territory, white_territory = self.calculate_territory()
        black_captured = np.sum(self.board == -1)
        white_captured = np.sum(self.board == 1)

        black_score = black_territory + black_captured
        white_score = white_territory + white_captured + self.komi

        return black_score, white_score

    def calculate_territory(self):
        # Implement territory calculation
        # This is a placeholder, and you may need a more sophisticated algorithm
        # to accurately calculate territory
        return 0, 0

    def get_winner(self):
        if not self.game_over:
            return None

        black_score, white_score = self.calculate_score()

        if black_score > white_score:
            return 1
        elif black_score < white_score:
            return -1
        else:
            return 0
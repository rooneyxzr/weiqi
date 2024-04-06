class MCTSNode:
    # ... 实现MCTS节点的数据结构和方法 ...
    def __init__(self) -> None:
        pass
        
class MCTS:
    def __init__(self, model, c_puct=1.0, n_playout=1000):
        self.model = model
        self.c_puct = c_puct
        self.n_playout = n_playout

    def search(self, board: GoBoard, player):
        root = MCTSNode(board, player)
        
        for _ in range(self.n_playout):
            node = root
            while not node.is_leaf():
                node = node.select()

            leaf_value = self.model.predict(node.state)
            node.expand(leaf_value)

            backpropagate = node
            while backpropagate is not None:
                backpropagate.update(leaf_value)
                backpropagate = backpropagate.parent

        return root.get_best_action()
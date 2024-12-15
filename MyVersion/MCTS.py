from functools import update_wrapper

import numpy as np


class NodeSelectFunc():
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)

def UCB():
    pass

class TreeNode(object):
    def __init__(self, parent):
        self._parent = parent  # node
        self._children = {}  # index->node
        self._n_visits = 0
        self._value = 0
        self.is_terminal = False

    def expand(self, actions):
        # 根据mask和棋局格式，单纯扩展结点而不做任何选择
        if len(self._children) != 0:
            assert 'node is already expanded, it is illegal to expand it again'
        for id in self.actions:
            if id not in self._children:
                self._children[id] = TreeNode(self)

    def is_leaf(self):
        return len(self._children) == 0

    def is_visited(self):
        return self._n_visits > 0

    def backward(self, reward):
        self._n_visits += 1
        self._value += reward
        if self._parent is not None:
            self._parent.backward(reward)
        return

    def get_value(self, fn):
        # 如UCB、UCT
        return fn(self._value, self._n_visits)

    def select(self, fn, available_actions):
        best_action = available_actions[0]
        best_score = -float('inf')
        for action, in available_actions:
            assert action in self._children.keys()
            if self._children[action].get_value(fn) > best_score:
                best_score = self._children[action].get_value(fn)
                best_action = action
        return best_action, self._children[best_action]

    def go_to_node(self, action):
        assert action not in self._children
        return self._children[action]

class MCTS():
    # state：棋盘格局
    # 蒙特卡洛树搜索算法，主框架描述
    # 通过UCB或UCT搜索蒙特卡洛树到叶子state节点
    # 寻找合法下棋点（new_state棋盘格局），计算rollout.reward&is_terminate，backward路线所有节点的value&n
    # 如果is_terminate==TRUE，则该节点不再有合法下棋点，在模拟搜索过程中会略过。

    # rollout方法打算用遗传算法，不在本类，需要import
    def __init__(self, shape, rollout_fn, epoch):
        self._rootnode = TreeNode(None)
        self._rollout_policy = rollout_fn  # 这个fn应该被封装成类,输入棋局数据
        self._select_policy = NodeSelectFunc(UCB)
        self.epoch = epoch
        self.gomoku_shape = shape

    def play(self,actions_trajectory, unavailable_actions):
        gomoku = np.zeros(self.gomoku_shape)
        gomoku[np.unravel_index(unavailable_actions, gomoku.shape)] = -1.0

        node = self._rootnode
        for action in actions_trajectory:  # 寻找叶子节点
            node = node.go_to_node(action)

        gomoku[np.unravel_index(actions_trajectory, gomoku.shape)] = 1.0
        available_actions = np.where(gomoku==0.)
        available_actions = np.ravel_multi_index(available_actions, gomoku.shape)

        if node.is_leaf():  # 叶子节点
            node.expand(range(self.gomoku_shape[0] * self.gomoku_shape[1]))

        action, node = node.select(self._select_policy, available_actions)
        actions_trajectory.append(action)

        reward, is_over =self._rollout_policy(actions_trajectory)
        node.backward(reward)
        return actions_trajectory, is_over

    def update(self):
        is_win = False
        while True:
            self.update_step()
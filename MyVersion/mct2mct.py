

# 使用两个蒙特卡洛树进行对抗生成state
# 进度：rollout policy还没写，拟定遗传算法
# 进度：select policy还没写，拟定UCB
# 可视化plot还没写，最后再弄！
import numpy as np
from MCTS import MCTS

gomoku_shape = (6,6)
epoch = 1000


player1 = MCTS(gomoku_shape, None, epoch)
player2 = MCTS(gomoku_shape, None, epoch)
actions1 = []
actions2 = []

for i in range(epoch):
    while True:
        actions1, is_over = player1.play(actions1, actions2)
        if is_over:
            print(f'player 1 wins!game over') if is_over == 1 else \
                print(f'gomoku board is full! game over!')
            break
        actions2, is_over= player2.play(actions2, actions1)
        if is_over:
            print(f'player 2 wins!game over') if is_over == 1 else \
                print(f'gomoku board is full! game over!')
            break
import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
dir = [[0, 1],
       [0, -1],
       [1, 0],
       [-1, 0],
       [1, 1],
       [1, -1],
       [-1, 1],
       [-1, -1]
       ]
Vmap = np.array([[500, -25, 10, 5, 5, 10, -25, 500],
                 [-25, -200, 1, 1, 1, 1, -200, -25],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [5, 1, 2, 1, 1, 2, 1, 5],
                 [5, 1, 2, 1, 1, 2, 1, 5],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [-25, -200, 1, 1, 1, 1, -200, -25],
                 [500, -25, 10, 5, 5, 10, -25, 500]])
corner = [[0, 0], [0, 7], [7, 0], [7, 7]]
dic = {
    0: [1, 1],
    49: [1, -1],
    7: [-1, 1],
    56: [-1, -1]
}
star = [(1, 6), (1, 1), (6, 1), (6, 6)]


# 更改： 下子后更新棋盘状态！ --finish

class AI(object):

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    # 计分标准：
    # 可行点个数
    # 稳定点个数
    # 位置权值

    #done
    def go(self, chessboard):
        #t1 = time.time()
        self.candidate_list.clear()

        temp = self.usable(chessboard, self.color)
        # eva = []
        level = 3
        max_val = -100000000
        stage = sum(sum(abs(chessboard)))
        if stage>=45:
            level = 5
        elif stage<=8:
            level = 4

        for i in temp:
            self.candidate_list.append(i)
        temp,boardlist = self.cutting(chessboard,temp,self.color)
        for i in range(len(temp)):
            val = self.alphaBeta(level,-self.color,boardlist[i],max_val,100000000)
            if temp[i] in star and level < 5:
                val -= 200
            if val > max_val:
                max_val = val
                self.candidate_list.append(temp[i])

    #done
    def cutting(self,chessboard,moves,color):
        boardlist = []
        values = []
        for i in moves:
            newboard = chessboard.copy()
            newboard[i[0]][i[1]] = self.color
            self.update(newboard, i, self.color)
            val = self.alphaBeta(0, 0 - self.color, newboard, -10000, 10000000)
            boardlist.append(newboard)
            values.append(val)
        if color==self.color:
            ind = np.argsort(values)[::-1]
        else:
            ind = np.argsort(values)
        maxN = 6
        moves = [moves[i] for i in ind[0:maxN]]
        boardlist = [boardlist[i] for i in ind[0:maxN]]
        return moves,boardlist
    ###########done
    def update(self, chessboard, dot, color):
        re_dir = []
        for i in dir:
            row = i[0]
            col = i[1]
            x = row + dot[0]
            y = col + dot[1]
            if (x not in range(0, 8)) or (y not in range(0, 8)):
                continue
            elif chessboard[x][y] == 0 or chessboard[x][y] == color:
                continue
            else:
                while x in range(0, 8) and y in range(0, 8):
                    if chessboard[x][y] == color:
                        re_dir.append(i)
                    elif chessboard[x][y] == 0:
                        break
                    x += row
                    y += col

        for i in re_dir:
            row = i[0]
            col = i[1]
            x = row + dot[0]
            y = col + dot[1]
            while True:
                if chessboard[x][y] == color:
                    break
                else:
                    chessboard[x][y] = 0 - chessboard[x][y]
                x += row
                y += col

    ###########done
    def alphaBeta(self, level, color, chessboard, alpha, beta):
        if level == 0:
            return self.evaluate(chessboard)

        if color == self.color:
            # AI turn
            child = self.usable(chessboard, color)
            if len(child) == 0:
                val = self.alphaBeta(level - 1, 0 - color, chessboard, alpha, beta)
                if val > alpha:
                    alpha = val
            for i in child:
                newchess = chessboard.copy()
                newchess[i[0]][i[1]] = color
                self.update(newchess, i, color)
                val = self.alphaBeta(level - 1, 0 - color, newchess, alpha, beta)
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    break
            return alpha
        else:
            # player's turn
            child = self.usable(chessboard, color)
            if len(child) == 0:
                val = self.alphaBeta(level - 1, 0 - color, chessboard, alpha, beta)
                if val < beta:
                    beta = val
            for i in child:
                newchess = chessboard.copy()
                newchess[i[0]][i[1]] = color
                self.update(newchess, i, color)
                val = self.alphaBeta(level - 1, 0 - color, newchess, alpha, beta)
                if val < beta:
                    beta = val
                if alpha >= beta:
                    break
            return beta


    def evaluate(self, chessboard):
        steps = len(np.where(chessboard == COLOR_NONE)[0])
        arg_location = 1
        arg_actionable = 0
        arg_stable = 1
        arg_num = 0
        sum = 0

        if steps > 50:
            # 初期
            arg_actionable = 10
            arg_stable = 50
        elif steps <= 15 and steps>8:
            #中后期
            arg_num = 10
            arg_stable = 30
            arg_actionable = 20
        elif steps<=8:
            #后期
            arg_num = 15
            arg_stable = 20
        else:
            # 中期
            arg_actionable = 20
            arg_stable = 50

        sum += arg_location * self.ev_location(chessboard)
        sum += arg_stable * (self.stableCorner(chessboard, self.color) - self.stableCorner(chessboard, 0 - self.color))
        sum += arg_actionable * (self.ev_actionable(chessboard,self.color)-self.ev_actionable(chessboard,-self.color))
        sum += arg_num * self.ev_num(chessboard)
        return sum
    #done
    def ev_num(self, chessboard):
        mine = len(np.where(chessboard == self.color)[0])
        other = len(np.where(chessboard == (0 - self.color))[0])
        return mine - other
    #done
    def stableCorner(self, chessboard, color):
        sta = self.checkCorner(chessboard, color)
        sta_list = []
        result = []
        if len(sta) != 0:
            for i in sta:
                self.countStable(chessboard, i, sta_list)
        for i in sta_list:
            if i not in result:
                result.append(i)
        return len(result)

    def countStable(self, chessboard, cor, sta_list):
        cn_dir = dic[cor[0] + cor[1] * cor[1]]
        row = cn_dir[0]
        col = cn_dir[1]
        x = cor[0]
        y = cor[1]
        while x in range(8):
            y = cor[1]
            if chessboard[x][cor[1]] != chessboard[cor[0]][cor[1]]:
                break
            while y in range(8):
                if chessboard[x][y] == chessboard[cor[0]][cor[1]]:
                    if (x - row) not in range(8) or (y + col) not in range(8) or (x - row, y + col) in sta_list:
                        sta_list.append((x, y))
                else:
                    break
                y += col
            x += row

        while y in range(8):
            x = cor[0]
            if chessboard[cor[0]][y] != chessboard[cor[0]][cor[1]]:
                break
            while x in range(8):
                if chessboard[x][y] == chessboard[cor[0]][cor[1]]:
                    if (x + row) not in range(8) or (y - col) not in range(8) or (x + row, y - col) in sta_list:
                        sta_list.append((x, y))
                else:
                    break
                x += row
            y += col

    def checkCorner(self, chessboard, color):
        cor = []
        for i in corner:
            if chessboard[i[0]][i[1]] == color:
                cor.append(i)
        return cor

    #done
    def ev_actionable(self, chessboard,color):
        cnt = 0
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        for i in range(0, len(idx)):
            x = idx[i][0]
            y = idx[i][1]
            if self.checking(chessboard, x, y, color):
                cnt += 1
        return cnt

    def ev_location(self, chessboard):
        return sum(sum(chessboard * Vmap)) * self.color

    #done
    def usable(self, chessboard, color):
        temp = []
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        for i in range(0, len(idx)):
            x = idx[i][0]
            y = idx[i][1]
            if self.checking(chessboard, x, y, color):
                temp.append((x, y, Vmap[x][y]))
        temp.sort(key=lambda temp: temp[2], reverse=True)
        result = []
        for i in temp:
            result.append((i[0], i[1]))
        return result
    #done
    def checking(self, chessboard, ini_x, ini_y, color):
        for i in range(8):
            row = dir[i][0]
            col = dir[i][1]
            x = row + ini_x
            y = col + ini_y
            if (x not in range(0, 8)) or (y not in range(0, 8)):
                continue
            elif chessboard[x][y] == 0 or chessboard[x][y] == color:
                continue
            else:
                while x in range(0, 8) and y in range(0, 8):
                    if chessboard[x][y] == color:
                        return True
                    elif chessboard[x][y] == 0:
                        break
                    x += row
                    y += col
            if i == 7:
                return False


if __name__ == '__main__':
    board = AI(chessboard_size=8, color=1, time_out=50)
    arr = np.array(
[[ 0, 0, 1, 0,-1,-1,-1, 0],
 [ 1, 0, 1, 1, 1, 1, 1, 0],
 [ 1,-1, 1,-1, 1, 1,-1,-1],
 [ 1,-1,-1,-1, 1,-1,-1, 1],
 [ 1, 1,-1, 1, 1,-1,-1, 1],
 [ 0,-1, 1,-1,-1,-1,-1,-1],
 [-1, 0,-1, 1, 1, 0, 0, 0],
 [ 0,-1, 1, 1, 1, 0, 0, 0]])
    t1 = time.time()
    board.go(arr)
    t2 = time.time()
    print(board.candidate_list)
    print(t2 - t1)
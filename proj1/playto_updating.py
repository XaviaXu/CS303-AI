import Tlayer_copyChessboard_withTiming
import upload
import numpy as np
import time

chessboard = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, -1, 0, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])
board_new = Tlayer_copyChessboard_withTiming.AI(chessboard_size=8, color=1, time_out=50)
board_old = upload.AI(chessboard_size=8, color=-1, time_out=50)
board = [board_old,board_new]
counting = 0
while True:
    cnt = 0
    for j in range(2):
        t1 = time.time()
        temp= board[j].go(chessboard)
        t2 = time.time()
        li1 = board[j].candidate_list
        if len(li1)!=0:
            i = li1[len(li1)-1]
            chessboard[i[0]][i[1]] = board[j].color
            board[j].update(chessboard, i, board_new.color)
            with open("E://log_changearg(2)_upload_rev.txt", "a+", encoding='utf-8') as f:
                f.writelines("*********************************\n")
                f.writelines(str(temp))
                tplp = "\n'{}', '{}', {}\n"
                f.writelines(tplp.format(i[0], i[1], board[j].color))
                t2 = t2-t1
                f.writelines(str(t2))
                f.write('\n')
                f.writelines(str(chessboard))
                f.write('\n')
                if(t2>5):
                    f.writelines("!!!!!!!!!!!\n")

        else:
            cnt+=1

        counting+=1
        print(counting)

    if cnt==2:
        print(len(np.where(chessboard==-1)[0]))
        print(len(np.where(chessboard == 1)[0]))
        break
import Tlayer_copyChessboard_withTiming
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
board = Tlayer_copyChessboard_withTiming.AI(chessboard_size=8, color=-1, time_out=50)
counting = 0
while True:
    cnt = 0
    for i in range(2):
        t1 = time.time()
        temp ,level = board.go(chessboard)
        t2 = time.time()
        li1 = board.candidate_list
        if len(li1)!=0:
            i = li1[len(li1)-1]
            chessboard[i[0]][i[1]] = board.color
            board.update(chessboard,i,board.color)
            with open("E://log_Tl_changearg(2)_timing.txt", "a+", encoding='utf-8') as f:
                f.writelines("*********************************\n")
                f.writelines(str(temp))
                tplp = "\n'{}', '{}', {}, {}\n"
                f.writelines(tplp.format(i[0],i[1],board.color,level))
                t2 = t2-t1
                f.writelines(str(t2))
                f.write('\n')
                f.writelines(str(chessboard))
                f.write('\n')
                if(t2>5):
                    f.writelines("!!!!!!!!!!!\n")

        else:
            cnt+=1
        board.color = -board.color
        counting+=1
        print(counting)

    if cnt==2:
        print(len(np.where(chessboard==-1)[0]))
        print(len(np.where(chessboard == 1)[0]))
        break
from typing import List


class AverageMeter:
    """
        用一个指定大小的数组来存放积累的数据，
        并且可以计算数据的均值 
    """
    def __init__(self, history_length) -> None:
        self.sum = 0.0
        self.count: int = 0
        self.history: List[float] = [None] * history_length
        self.current_pointer = -1
    
    @property
    def val(self) -> float:
        # 返回当前值
        return self.history[self.current_pointer]

    @property
    def avg(self) -> float:
        # 返回当前数组中所有的值的均值
        return self.sum / self.count
    
    def update(self, val):
        # 更新数组
        self.sum += val
        self.current_pointer = (self.current_pointer + 1) % \
            len(self.history)

        old_val = self.history[self.current_pointer]
        if old_val == None:
            # history还没被填满
            self.count += 1
        else:
            self.sum -= old_val


import time
import numpy as np

class tester:
    def __init__(self, sample_num, iter_num, hist = True):
        self.sample_num = sample_num
        self.iter_num = iter_num
        self.it_list = []
        self.F1_list = []
        self.run_time = 0.0
        self.hist = hist
        
    def run_once(self, solver):
        solver.clear()
        it = []
        F1 = []
        for i in range(self.sample_num):
            solver.select_point()
            solver.update()
            if i > 1 and self.hist:
                it.append(i)
                F1.append(solver.get_score(1.96)[1])
        if self.hist:
            return it, F1
    
    def run(self, solver):
        start = time.clock()
        for i in range(self.iter_num):
            if self.hist:
                it, F1 = self.run_once(solver)
                self.it_list.append(it)
                self.F1_list.append(F1)
            else:
                self.run_once(solver)
                self.F1_list.append(solver.get_score(1.96)[1])
        self.run_time = time.clock() - start
            
            
    def clear(self):
        self.it_list = []
        self.F1_list = []
        self.run_time = 0.0
        
    def get_error_bar_param(self):
        F1_array = np.array(self.F1_list)
        std = np.std(F1_array, axis=0)
        mean = np.mean(F1_array, axis=0)
        return self.it_list[0], mean, std
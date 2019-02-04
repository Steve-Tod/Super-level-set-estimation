import numpy as np
import random, copy
from scipy.stats import norm

class plain_solver(object):
    
    def __init__(self, sample_space, threshold, sigma, function,
                 kernel, mean):
        self.kernel = kernel
        self.mean = mean
        self.x_hist = []
        self.y_hist = []
        self.sample_space = sample_space
        self.function = function
        self.threshold = threshold
        self.sigma = sigma
        self.Kt = None
        
    def select_point(self):
        point = random.choice(self.sample_space)
        y = self.function(point) + np.random.normal(scale=self.sigma)
        self.x_hist.append(point)
        self.y_hist.append(y)

    def update(self):
        assert len(self.x_hist) == len(self.y_hist) and len(self.x_hist) > 0
        sample_num = len(self.x_hist)
        if sample_num == 1:
            x0 = self.x_hist[0]
            self.Kt = np.zeros((1, 1))
            self.Kt[0, 0] = self.kernel(x0, x0)
        else:
            self.Kt = np.pad(self.Kt, ((0, 1), (0, 1)), mode='constant')
            x_new = self.x_hist[-1]
            for i, xi in enumerate(self.x_hist[:-1]):
                self.Kt[-1, i] = self.kernel(x_new, xi)
                self.Kt[i, -1] = self.kernel(xi, x_new)
            self.Kt[-1, -1] = self.kernel(x_new, x_new)
        
        assert self.Kt.shape == (sample_num, sample_num)

    def get_super_level_set(self, beta):
        assert len(self.x_hist) == len(self.y_hist)
        super_level_set = []
        left_set = []
        sample_num = len(self.x_hist)
        yt = np.reshape(np.array(self.y_hist), (-1, 1))
        mut = np.reshape(np.array([self.mean(x) for x in self.x_hist]), (-1, 1))
        Kt_inv = np.linalg.inv(self.Kt +
                               np.eye(sample_num) * np.square(self.sigma))
        for x in self.sample_space:
            if sample_num > 0:
                kt = [self.kernel(x_sample, x) for x_sample in self.x_hist]
                kt = np.reshape(np.array(kt), (-1, 1))
                
                mean_t = self.mean(x) + np.transpose(kt) @ Kt_inv @ (yt - mut)
                variance = self.kernel(x, x) - np.transpose(kt) @ Kt_inv @ kt
                if variance > 0:
                    std = np.sqrt(variance)
                else:
                    std = 0.0
                if mean_t - std * beta > self.threshold:
                    super_level_set.append(x)
                else:
                    left_set.append(x)
        return super_level_set, left_set
    
    def get_gt_set(self):
        gt_set = []
        for x in self.sample_space:
            if self.function(x) > self.threshold:
                gt_set.append(x)
        return gt_set

    def get_score(self, beta):
        super_level_set, left_set = self.get_super_level_set(beta)
        TP, FP, TN, FN = .0, .0, .0, .0
        for point in super_level_set:
            if self.function(point) > self.threshold:
                TP += 1
            else:
                FP += 1

        for point in left_set:
            if self.function(point) < self.threshold:
                TN += 1
            else:
                FN += 1

        Acc = (TP + TN) / (TP + TN + FN + FP)
        F1 = 2 * TP / (2 * TP + FP + FN)
        return Acc, F1
    
    def clear(self):
        self.x_hist = []
        self.y_hist = []
        self.Kt = None
    
class straddle_solver(plain_solver):
    def __init__(self, sample_space, threshold, sigma, function, kernel, mean):
        super().__init__(sample_space, threshold, sigma, function, kernel,
                         mean)

    def select_point(self):
        sample_num = len(self.x_hist)
        max_score = -100
        if sample_num > 0:
            yt = np.reshape(np.array(self.y_hist), (-1, 1))
            Kt_inv = np.linalg.inv(self.Kt +
                                   np.eye(sample_num) * np.square(self.sigma))
            mut = np.reshape(np.array([self.mean(x) for x in self.x_hist]), (-1, 1))
        for x in self.sample_space:
            if sample_num > 0:
                kt = [self.kernel(x_sample, x) for x_sample in self.x_hist]
                kt = np.reshape(np.array(kt), (-1, 1))

                mean_t = self.mean(x) + np.transpose(kt) @ Kt_inv @ (yt - mut)
                variance = self.kernel(x, x) - np.transpose(kt) @ Kt_inv @ kt
                if variance > 0:
                    std = np.sqrt(variance)
                else:
                    std = 0.0
            else:  # at first, choose randomly
                mean_t = self.mean(x)
                std = np.sqrt(self.kernel(x, x))
            straddle_score = 1.96 * std - np.abs(mean_t - self.threshold)
            if straddle_score > max_score:
                max_point = x
                max_score = straddle_score

        y = self.function(max_point) + np.random.normal(scale=self.sigma)
        self.x_hist.append(max_point)
        self.y_hist.append(y)
        
        
class p_straddle_solver(plain_solver):
    def __init__(self, sample_space, threshold, sigma, function, kernel, mean, dropout = 0.5):
        super().__init__(sample_space, threshold, sigma, function, kernel,
                         mean)
        self.dropout = dropout

    def select_point(self):
        sample_num = len(self.x_hist)
        max_score = -100
        if sample_num > 0:
            yt = np.reshape(np.array(self.y_hist), (-1, 1))
            Kt_inv = np.linalg.inv(self.Kt +
                                   np.eye(sample_num) * np.square(self.sigma))
            mut = np.reshape(np.array([self.mean(x) for x in self.x_hist]), (-1, 1))
        for x in self.sample_space:
            if sample_num > 0:
                if random.uniform(0, 1) < self.dropout:
                    continue
                kt = [self.kernel(x_sample, x) for x_sample in self.x_hist]
                kt = np.reshape(np.array(kt), (-1, 1))

                mean_t = self.mean(x) + np.transpose(kt) @ Kt_inv @ (yt - mut)
                variance = self.kernel(x, x) - np.transpose(kt) @ Kt_inv @ kt
                if variance > 0:
                    std = np.sqrt(variance)
                else:
                    std = 0.0
            else:  # at first, choose randomly
                mean_t = self.mean(x)
                std = np.sqrt(self.kernel(x, x))
            straddle_score = 1.96 * std - np.abs(mean_t - self.threshold)
            if straddle_score > max_score:
                max_point = x
                max_score = straddle_score

        y = self.function(max_point) + np.random.normal(scale=self.sigma)
        self.x_hist.append(max_point)
        self.y_hist.append(y)
      
    
class point_region_pair:
    def __init__(self, point):
        assert isinstance(point, np.ndarray)
        assert point.shape == (2,)
        self.point = copy.deepcopy(point)
        self.region = [-np.inf, np.inf]
    
    def intersect(self, lower, upper):
        if upper < self.region[0] or lower > self.region[1]:
            raise ValueError("No intersection")
        else:
            self.region[0] = max(lower, self.region[0])
            self.region[1] = min(upper, self.region[1])
            
            
class lse_solver(plain_solver):
    def __init__(self, sample_space, threshold, sigma, function, kernel, mean, epsilon):
        super().__init__(sample_space, threshold, sigma, function, kernel,
                         mean)
        self.epsilon = epsilon
        self.H = []
        self.U = [point_region_pair(p) for p in self.sample_space]
        self.done = False
        
    def clear(self):
        self.x_hist = []
        self.y_hist = []
        self.H = []
        self.U = [point_region_pair(p) for p in self.sample_space]
        self.Kt = None

    def update(self):
        assert len(self.x_hist) == len(self.y_hist) and len(self.x_hist) > 0
        sample_num = len(self.x_hist)
        if sample_num == 1:
            x0 = self.x_hist[0]
            self.Kt = np.zeros((1, 1))
            self.Kt[0, 0] = self.kernel(x0, x0)
        else:
            self.Kt = np.pad(self.Kt, ((0, 1), (0, 1)), mode='constant')
            x_new = self.x_hist[-1]
            for i, xi in enumerate(self.x_hist[:-1]):
                self.Kt[-1, i] = self.kernel(x_new, xi)
                self.Kt[i, -1] = self.kernel(xi, x_new)
            self.Kt[-1, -1] = self.kernel(x_new, x_new)
        
        assert self.Kt.shape == (sample_num, sample_num)
            
    def select_point(self):
        sample_num = len(self.x_hist)
        max_score = -100
        del_list = []
        if sample_num > 0:
            yt = np.reshape(np.array(self.y_hist), (-1, 1))
            Kt_inv = np.linalg.inv(self.Kt +
                                   np.eye(sample_num) * np.square(self.sigma))
            mut = np.reshape(np.array([self.mean(x) for x in self.x_hist]), (-1, 1))
        for pair in self.U:
            x = pair.point
            if sample_num > 0:
                kt = [self.kernel(x_sample, x) for x_sample in self.x_hist]
                kt = np.reshape(np.array(kt), (-1, 1))

                mean_t = self.mean(x) + np.transpose(kt) @ Kt_inv @ (yt - mut)
                variance = self.kernel(x, x) - np.transpose(kt) @ Kt_inv @ kt
                if variance > 0:
                    std = np.sqrt(variance)
                else:
                    std = 0.0
            else:  # at first, choose randomly
                mean_t = self.mean(x)
                std = np.sqrt(self.kernel(x, x))
                
            pair.intersect(mean_t - 1.96 * std, mean_t + 1.96 * std)
            if pair.region[0] + self.epsilon > self.threshold:
                self.H.append(pair)
                del_list.append(pair)
                continue

            if pair.region[1] - self.epsilon <= self.threshold:
                del_list.append(pair)
                continue
                
            score = min(np.abs(pair.region[0] - self.threshold), np.abs(pair.region[1] - self.threshold))
            if score > max_score:
                max_point = x
                max_score = score

        # clear U
        for pair in del_list:
            remove_index = -1
            for i, pair_u in enumerate(self.U):
                if np.array_equal(pair.point, pair_u.point):
                    remove_index = i
                    break
            if remove_index > -1:
                self.U.pop(remove_index)
            else:
                raise ValueError('array not found in list.')
                
        if len(self.U) > 0:
            y = self.function(max_point) + np.random.normal(scale=self.sigma)
            self.x_hist.append(max_point)
            self.y_hist.append(y)
        else:
            self.done = True
            
      
class p_lse_solver(plain_solver):
    def __init__(self, sample_space, threshold, sigma, function, kernel, mean,
                 epsilon, dropout):
        super().__init__(sample_space, threshold, sigma, function, kernel,
                         mean)
        self.epsilon = epsilon
        self.H = []
        self.U = [point_region_pair(p) for p in sample_space]
        self.dropout = dropout
        self.done = False

    def clear(self):
        self.x_hist = []
        self.y_hist = []
        self.H = []
        self.U = [point_region_pair(p) for p in sample_space]
        self.Kt = None

    def update(self):
        assert len(self.x_hist) == len(self.y_hist) and len(self.x_hist) > 0
        sample_num = len(self.x_hist)
        if sample_num == 1:
            x0 = self.x_hist[0]
            self.Kt = np.zeros((1, 1))
            self.Kt[0, 0] = self.kernel(x0, x0)
        else:
            self.Kt = np.pad(self.Kt, ((0, 1), (0, 1)), mode='constant')
            x_new = self.x_hist[-1]
            for i, xi in enumerate(self.x_hist[:-1]):
                self.Kt[-1, i] = self.kernel(x_new, xi)
                self.Kt[i, -1] = self.kernel(xi, x_new)
            self.Kt[-1, -1] = self.kernel(x_new, x_new)

        assert self.Kt.shape == (sample_num, sample_num)

    def select_point(self):
        sample_num = len(self.x_hist)
        max_score = -100
        del_list = []
        if sample_num > 0:
            yt = np.reshape(np.array(self.y_hist), (-1, 1))
            Kt_inv = np.linalg.inv(self.Kt +
                                   np.eye(sample_num) * np.square(self.sigma))
            mut = np.reshape(
                np.array([self.mean(x) for x in self.x_hist]), (-1, 1))
        for pair in self.U:
            if random.uniform(0, 1) < self.dropout:
                continue
            x = pair.point
            if sample_num > 0:
                kt = [self.kernel(x_sample, x) for x_sample in self.x_hist]
                kt = np.reshape(np.array(kt), (-1, 1))

                mean_t = self.mean(x) + np.transpose(kt) @ Kt_inv @ (yt - mut)
                variance = self.kernel(x, x) - np.transpose(kt) @ Kt_inv @ kt
                if variance > 0:
                    std = np.sqrt(variance)
                else:
                    std = 0.0
            else:  # at first, choose randomly
                mean_t = self.mean(x)
                std = np.sqrt(self.kernel(x, x))

            pair.intersect(mean_t - 1.96 * std, mean_t + 1.96 * std)
            if pair.region[0] + self.epsilon > self.threshold:
                self.H.append(pair)
                del_list.append(pair)
                continue

            if pair.region[1] - self.epsilon <= self.threshold:
                del_list.append(pair)
                continue

            score = min(
                np.abs(pair.region[0] - self.threshold),
                np.abs(pair.region[1] - self.threshold))
            if score > max_score:
                max_point = x
                max_score = score

        # clear U
        for pair in del_list:
            remove_index = -1
            for i, pair_u in enumerate(self.U):
                if np.array_equal(pair.point, pair_u.point):
                    remove_index = i
                    break
            if remove_index > -1:
                self.U.pop(remove_index)
            else:
                raise ValueError('array not found in list.')

        if len(self.U) > 0:
            y = self.function(max_point) + np.random.normal(scale=self.sigma)
            self.x_hist.append(max_point)
            self.y_hist.append(y)
        else:
            self.done = True
            
           
class p_prob_var_solver(plain_solver):
    def __init__(self, sample_space, threshold, sigma, function, kernel, mean, epsilon, beta, dropout=0.5):
        super().__init__(sample_space, threshold, sigma, function, kernel,
                         mean)
        self.epsilon = epsilon
        self.H = []
        self.U = copy.deepcopy(sample_space)
        self.dropout = dropout
        self.beta = beta
        self.integrater = norm()
        self.done = False

    def clear(self):
        self.x_hist = []
        self.y_hist = []
        self.H = []
        self.U = copy.deepcopy(self.sample_space)
        self.Kt = None
        
    def update(self):
        assert len(self.x_hist) == len(self.y_hist) and len(self.x_hist) > 0
        sample_num = len(self.x_hist)
        if sample_num == 1:
            x0 = self.x_hist[0]
            self.Kt = np.zeros((1, 1))
            self.Kt[0, 0] = self.kernel(x0, x0)
        else:
            self.Kt = np.pad(self.Kt, ((0, 1), (0, 1)), mode='constant')
            x_new = self.x_hist[-1]
            for i, xi in enumerate(self.x_hist[:-1]):
                self.Kt[-1, i] = self.kernel(x_new, xi)
                self.Kt[i, -1] = self.kernel(xi, x_new)
            self.Kt[-1, -1] = self.kernel(x_new, x_new)
        
        assert self.Kt.shape == (sample_num, sample_num)
            
    def select_point(self):
        sample_num = len(self.x_hist)
        max_score = -100
        del_list = []
        if sample_num > 0:
            yt = np.reshape(np.array(self.y_hist), (-1, 1))
            Kt_inv = np.linalg.inv(self.Kt +
                                   np.eye(sample_num) * np.square(self.sigma))
            mut = np.reshape(np.array([self.mean(x) for x in self.x_hist]), (-1, 1))
        for x in self.U:
            if random.uniform(0, 1) < self.dropout:
                continue
            if sample_num > 0:
                kt = [self.kernel(x_sample, x) for x_sample in self.x_hist]
                kt = np.reshape(np.array(kt), (-1, 1))

                mean_t = self.mean(x) + np.transpose(kt) @ Kt_inv @ (yt - mut)
                variance = self.kernel(x, x) - np.transpose(kt) @ Kt_inv @ kt
                if variance > 0:
                    std = np.sqrt(variance)
                else:
                    std = 0.0
            else:  # at first, choose randomly
                mean_t = self.mean(x)
                std = np.sqrt(self.kernel(x, x))
            if mean_t - 1.96 * std > self.threshold - self.epsilon:
                self.H.append(x)
                del_list.append(x)
                continue
            
            if mean_t + 1.96 * std < self.threshold + self.epsilon:
                del_list.append(x)
                continue
            
            score = self.beta * std + 1 - self.integrater.cdf(np.abs(mean_t - self.threshold)/std)
            if score > max_score:
                max_point = x
                max_score = score

        # clear U
        for x in del_list:
            remove_index = -1
            for i, x_u in enumerate(self.U):
                if np.array_equal(x, x_u):
                    remove_index = i
                    break
            if remove_index > -1:
                self.U.pop(remove_index)
            else:
                raise ValueError('array not found in list.')
                
        if len(self.U) > 0:
            y = self.function(max_point) + np.random.normal(scale=self.sigma)
            self.x_hist.append(max_point)
            self.y_hist.append(y)
        else:
            self.done = True
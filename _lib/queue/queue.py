# _*_coding:utf-8_*_

import time
from abc import abstractmethod
from collections import defaultdict

import numpy as np
from random import randint
# import tensorflow as tf
import datetime
import random
import os

from utils import config


class FuzzQueue(object):
    """Class that holds inputs and associated coverage."""

    def __init__(self,
                 outdir,
                 is_random,
                 sample_type,
                 criteria,
                 check_point,
                 DUMPS,
                 nb_class=10):
        """Init the class.
        """

        # self.plot_file = open(os.path.join(outdir, 'plot.log'), 'a+')
        self.out_dir = outdir
        self.mutations_processed = 0
        self.queue = {}
        self.sample_type = sample_type
        self.start_time = time.time()
        # whether it is random testing
        self.random = is_random
        self.criteria = criteria
        self.check_point = check_point
        self.DUMPS = DUMPS

        self.log_time = time.time()
        # Like AFL, it records the coverage of the seeds in the queue
        # self.virgin_bits = np.full(cov_num, 0xFF, dtype=np.uint8)

        self.diverse_num_bins = np.zeros((nb_class, nb_class), dtype="int32")
        self.diverse_iter_bins = np.zeros((nb_class, nb_class), dtype="int32")

        # self.adv_bits = np.full(cov_num, 0xFF, dtype=np.uint8)

        self.uniq_crashes = 0
        self.total_queue = 0
        # self.total_cov = cov_num

        # Some log information
        self.last_crash_time = self.start_time
        self.last_reg_time = self.start_time
        self.current_id = 0
        self.seed_attacked = set()  # 只记录根种子
        self.seed_bugs = defaultdict(int)  # 记录(轮次,种子id)
        self.seed_attacked_first_time = dict()

        self.dry_run_cov = None

        # REG_MIN and REG_GAMMA are the p_min and gamma in Equation 3
        self.REG_GAMMA = 5
        self.REG_MIN = 0.3
        self.REG_INIT_PROB = 0.8

    # def has_new_bits(self, seed):
    #     if self.criteria == "space":
    #         return self.has_new_bits_space(seed)
    #     else:
    #         return self.has_new_bits_cov(seed)
    #
    # def has_new_bits_space(self, seed):
    #     temp = np.invert(seed.coverage, dtype=np.uint8)
    #     cur = np.bitwise_and(self.virgin_bits, temp)
    #     has_new = not np.array_equal(cur, self.virgin_bits)
    #     if has_new:
    #         # If the coverage is increased, we will update the coverage
    #         self.virgin_bits = cur
    #     return has_new or self.random

    # 统计覆盖对
    # def set_deverse_bins(self, i, j, iter):  # i ground truth  j meta-data
    #     if self.diverse_iter_bins[i][j] != 0:  # 如果是空的,代表第一次发现
    #         self.diverse_iter_bins[i][j] = iter
    #     self.diverse_num_bins[i][j] += 1

    # def save_deverse_bins(self, ):
    #     np.save(os.path.join(self.out_dir, "diverse_iter.npy"), self.diverse_iter_bins)
    #     np.save(os.path.join(self.out_dir, "diverse_num.npy"), self.diverse_num_bins)

    # def save_bugs_seed_dcit(self):
    #     np.save(os.path.join(self.out_dir, "bugs_seed.npy"), self.seed_bugs)

    # def has_new_bits(self, seed):
    #     if self.criteria == "fake":
    #         return False
    #     else:
    #         temp = np.invert(seed.coverage, dtype=np.uint8)
    #         cur = np.bitwise_and(self.virgin_bits, temp)
    #         has_new = not np.array_equal(cur, self.virgin_bits)
    #         if has_new:
    #             # If the coverage is increased, we will update the coverage
    #             self.virgin_bits = cur
    #     return has_new or self.random

    # def plot_log(self, id):
    #     # Plot the data during fuzzing, include: the current time, current iteration, length of queue, initial coverage,
    #     # total coverage, number of crashes, number of seeds that are attacked, number of mutations, mutation speed
    #     queue_len = len(self.queue)
    #     coverage = self.compute_cov()
    #     current_time = time.time()
    #     self.plot_file.write(
    #         "%d,%d,%d,%s,%s,%d,%d,%s,%s\n" %
    #         (time.time(),
    #          id,
    #          queue_len,
    #          self.dry_run_cov,
    #          coverage,
    #          self.uniq_crashes,
    #          len(self.seed_attacked),
    #          self.mutations_processed,
    #          round(float(self.mutations_processed) / (current_time - self.start_time), 2)
    #          ))
    #     self.plot_file.flush()

    # def write_logs(self):
    #     log_file = open(os.path.join(self.out_dir, 'fuzz.log'), 'w+')
    #     for k in self.seed_attacked_first_time:
    #         log_file.write("%s:%s\n" % (k, self.seed_attacked_first_time[k]))
    #     log_file.close()
    #     self.plot_file.close()

    # def log(self):
    #     queue_len = len(self.queue)
    #     coverage = self.compute_cov()
    #     current_time = time.time()
    #     # print(
    #     #     "Metrics %s | corpus_size %s | crashes_size %s | mutations_per_second: %s | total_exces %s | last new reg: %s | last new adv %s | coverage: %s -> %s%%"%(
    #     #     self.criteria,
    #     #     queue_len,
    #     #     self.uniq_crashes,
    #     #     round(float(self.mutations_processed) / (current_time - self.start_time), 2),
    #     #     self.mutations_processed,
    #     #     datetime.timedelta(seconds=(time.time() - self.last_reg_time)),
    #     #     datetime.timedelta(seconds=(time.time() - self.last_crash_time)),
    #     #     self.dry_run_cov,
    #     #     coverage)
    #     # )

    # def compute_cov(self):
    #     # Compute the current coverage in the queue
    #     cov = float(self.total_cov - np.count_nonzero(self.virgin_bits == 0xFF)) * 100 / self.total_cov

    #     if self.criteria == "space":  # DeepHunter中Pr有一半的区域是不会被覆盖
    #         coverage = round(cov * 2, 2)
    #     else:
    #         coverage = round(cov, 2)
    #     return str(coverage)

    # def tensorfuzz(self):
    #     """Grabs new input from corpus according to sample_function."""
    #     # choice = self.sample_function(self)
    #     corpus = self.queue
    #     reservoir = corpus[-5:] + [random.choice(corpus)]
    #     choice = random.choice(reservoir)
    #     return choice
    #     # return random.choice(self.queue)

    def select_next(self, num=4):
        # Different seed selection strategies (See details in Section 4)
        # if self.sample_type == 'uniform':
        #     return self._select()
        # elif self.sample_type == 'deep':
        #     return self.deep_select()
        # elif self.sample_type == 'random':
        #     return self.random_select()
        # elif self.sample_type == 'c1':
        #     return self._select()
        # elif self.sample_type == 'c2':
        if self.sample_type == 'c2':
            return self._select_c2()
        else:
            return self.___select(num)

    def ___select(self, num):
        ret = []
        for _i in range(num):
            ret.append(self._select())
        return ret

    def _select(self):
        scene_list = np.array(list(self.DUMPS['prob'].items()))

        prob = scene_list[:, 1].astype(np.float32)
        scene_list = scene_list[:, 0]

        prob /= np.sum(prob)

        scene = np.random.choice(scene_list, p=prob)

        prob = self.queue[scene][:, 1].astype(np.float32)
        prob /= np.sum(prob)
        return np.random.choice(self.queue[scene][:, 0], p=prob)

    def _select_c2(self):
        # sg_type_list = np.array(list(self.DUMPS['prob'].items()))

        # prob = sg_type_list[:, 1].astype(np.float32)
        # sg_type_list = sg_type_list[:, 0]

        # prob /= np.sum(prob)

        # sg_type = np.random.choice(sg_type_list, p=prob)

        # scene = np.random.choice(self.DUMPS['cluster'][sg_type]['scene_list'])

        # prob = self.queue[scene][:, 1].astype(np.float32)
        # prob /= np.sum(prob)
        # return np.random.choice(self.queue[scene][:, 0], p=prob)
        pass

    # def deep_select(self):
    #     scene_list = np.array(list(self.DUMPS['prob'].items()))

    #     prob = scene_list[:, 1].astype(np.float32)
    #     scene_list = scene_list[:, 0]

    #     prob /= np.sum(prob)

    #     scene = np.random.choice(scene_list, p=prob)

    #     deep_prob = self.queue[scene][:, 1].astype(np.float32)
    #     deep_prob /= np.sum(deep_prob)
    #     return np.random.choice(self.queue[scene][:, 0], p=deep_prob)

    # def random_select(self):
    #     scene_list = np.array(list(self.DUMPS['prob'].items()))[:, 0]

    #     prob = np.array([self.queue[scene].shape[0]
    #                      for scene in scene_list]).astype(np.float32)
    #     prob /= np.sum(prob)

    #     scene = np.random.choice(scene_list, p=prob)
    #     return np.random.choice(self.queue[scene][:, 0])

    def c1_select(self):
        return

    def c2_select(self):
        return

    # def deeptest_next(self):
    #     choice = self.queue[-1]
    #     return choice

    def fuzzer_handler(self, iteration, cur_seed, bug_found, coverage_inc):
        # The handler after each iteration
        if self.sample_type == 'deeptest' and not coverage_inc:
            # If deeptest cannot increase the coverage, it will pop the last seed from the queue
            self.queue.pop()

        elif self.sample_type == 'prob':
            # Update the probability based on the Equation 3 in the paper
            if cur_seed.probability > self.REG_MIN and cur_seed.fuzzed_time < self.REG_GAMMA * (
                    1 - self.REG_MIN):
                cur_seed.probability = self.REG_INIT_PROB - float(
                    cur_seed.fuzzed_time) / self.REG_GAMMA

        if bug_found:
            # Log the initial seed from which we found the adversarial. It is for the statics of Table 6
            self.seed_attacked.add(cur_seed.root_seed)
            self.seed_bugs[iteration] = cur_seed.id
            if not (cur_seed.parent in self.seed_attacked_first_time):
                # Log the information about when (which iteration) the initial seed is attacked successfully.
                self.seed_attacked_first_time[cur_seed.root_seed] = iteration

    # def prob_next(self):
    #     """Grabs new input from corpus according to sample_function."""
    #     while True:
    #         if self.current_id == len(self.queue):
    #             self.current_id = 0

    #         cur_seed = self.queue[self.current_id]
    #         if randint(0, 100) < cur_seed.probability * 100:
    #             # Based on the probability, we decide whether to select the current seed.
    #             cur_seed.fuzzed_time += 1
    #             self.current_id += 1
    #             return cur_seed
    #         else:
    #             self.current_id += 1

    @abstractmethod
    def save_if_interesting(self,
                            seed,
                            data,
                            crash,
                            dry_run=False,
                            suffix=None):
        pass

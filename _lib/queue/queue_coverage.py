import time
import numpy as np
import pickle
import os
import copy

from _lib.queue.queue import FuzzQueue
from utils import config


class ImageInputCorpus(FuzzQueue):

    def __init__(self,
                 outdir,
                 israndom,
                 sample_function,
                 criteria,
                 check_point,
                 DUMPS,
                 nb_class=10):
        """Init the class.

        Args:
          outdir:  the output directory
          israndom: whether this is random testing

          sample_function: a function that looks at the whole current corpus and
            samples the next element to mutate in the fuzzing loop.

          cov_num: the total number of items to keep the coverage. For example, in NC, the number of neurons is the cov_num
          see details in last paragraph in Setion 3.3
        Returns:
          Initialized object.
        """
        FuzzQueue.__init__(self,
                           outdir,
                           israndom,
                           sample_function,
                           criteria,
                           check_point,
                           DUMPS,
                           nb_class=nb_class)

    def _update_prob(self, scene, data):
        ret = None
        if self.sample_type == 'new':
            self.DUMPS['prob'][scene] = 1 / (
                self.DUMPS['scene'][scene]['scene_len']**2)
            ret = 1
        elif self.sample_type == 'deep':
            self.DUMPS['prob'][scene] = np.sqrt(
                self.DUMPS['scene'][scene]['scene_len'])
            ret = data['level']
        elif self.sample_type == 'random':
            self.DUMPS['prob'][scene] = self.DUMPS['scene'][scene]['scene_len']
            ret = 1
        elif self.sample_type == 'c1':
            eps = np.finfo(np.float32).eps
            self.DUMPS['prob'][scene] = 1 / (
                self.DUMPS['scene'][scene]['coverage1']**2 + eps)
            ret = 1
        elif self.sample_type == 'c2':
            # from _lib.func import cal_single_c2
            # eps = np.finfo(np.float32).eps
            # sg_type = data['scene_graph_type']
            # self.DUMPS['prob'][sg_type] = 1 / (cal_single_c2(
            #     self.DUMPS['cluster'][sg_type]['cluster'], sg_type)**2 + eps)
            # ret = 1
            pass
        return ret

    def save_if_interesting(self,
                            seed,
                            data,
                            crash,
                            dry_run=False,
                            suffix=None):
        """Save the seed if it is a bug or increases the coverage."""

        self.mutations_processed += 1
        current_time = time.time()

        # compute the dry_run coverage,
        # i.e., the initial coverage. See the result in row Init. in Table 4
        # if dry_run:
        # coverage = self.compute_cov()
        # self.dry_run_cov = coverage
        # print some information
        if current_time - self.log_time > 2:
            self.log_time = current_time
            # if not dry_run:
            #     self.log()

        # similar to AFL, generate the seed name
        if seed.parent is None:
            describe_op = "src_%s" % (suffix)
        else:
            describe_op = "src_%06d" % (seed.parent.id)
        # if this is the crash seed, just put it into the crashes dir
        scene = seed.root_seed.split('.')[0]
        if crash:
            # fn = "%s/crashes/id_%06d_%s.npy" % (self.out_dir, self.uniq_crashes, describe_op)
            if not os.path.exists(f'{self.out_dir}/crashes/{scene}/'):
                os.makedirs(f'{self.out_dir}/crashes/{scene}/')
            fn = f'{self.out_dir}/crashes/{scene}/id_{self.total_queue:06d}_{describe_op}.pickle'
            self.total_queue += 1
            self.last_crash_time = current_time
        else:
            # fn = "%s/queue/id_%06d_%s.npy" % (self.out_dir, self.total_queue, describe_op)
            if not os.path.exists(f'{self.out_dir}/queue/{scene}/'):
                os.makedirs(f'{self.out_dir}/queue/{scene}/')
            fn = f'{self.out_dir}/queue/{scene}/id_{self.total_queue:06d}_{describe_op}.pickle'
            # has_new_bits : implementation for Line-9 in Algorithm1, i.e., has increased the coverage
            # During dry_run process, we will keep all initial seeds.
            self.last_reg_time = current_time
            seed.queue_time = current_time
            seed.id = self.total_queue
            # the seed path
            seed.fname = fn

            self.DUMPS['scene'][scene]['scene_len'] += 1
            self.DUMPS['fa'][seed.id] = data['level']
            prob = self._update_prob(scene, data)

            if scene in self.queue:
                self.queue[scene] = np.concatenate(
                    [self.queue[scene], [[seed, prob]]], axis=0)
            else:
                self.queue[scene] = np.array([[seed, prob]])

            self.total_queue += 1

        if not dry_run:
            print("seed_name : %s" % fn.split('/')[-1])
        with open(fn, 'wb') as f:
            pickle.dump(data, f)
        return True

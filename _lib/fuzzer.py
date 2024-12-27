# _*_coding:utf-8_*_

# import tensorflow as tf
import gc
import time
import numpy as np
from utils import config, DUMPS_utlis
import pickle
import os
from _others.fid.lidargen_fid import get_fid


class Fuzzer(object):
    """Class representing the fuzzer itself."""

    def __init__(self,
                 corpus,
                 metadata_function,
                 objective_function,
                 mutation_function,
                 fetch_function,
                 iterate_function,
                 frd_function,
                 plot=True):
        """Init the class.

    Args:
      corpus: An InputCorpus object.
      coverage_function: a function that does CorpusElement -> Coverage.
      metadata_function: a function that does CorpusElement -> Metadata.
      objective_function: a function that checks if a CorpusElement satisifies
        the fuzzing objective (e.g. find a NaN, find a misclassification, etc).
      mutation_function: a function that does CorpusElement -> Metadata.
      fetch_function: grabs numpy arrays from the TF runtime using the relevant
        tensors.
    Returns:
      Initialized object.
    """
        self.plot = plot
        self.queue = corpus
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function
        self.fetch_function = fetch_function
        self.iterate_function = iterate_function
        self.frd_function = frd_function

    def save(self, iteration):
        cov1 = self.queue.DUMPS['coverage1']
        cov2 = self.queue.DUMPS['coverage2']
        err_cov1 = self.queue.DUMPS['error_coverage1']
        err_cov2 = self.queue.DUMPS['error_coverage2']

        print(
            f'Coverage1 : {cov1:.02%}\nCoverage2 : {cov2:.02%}\nError Coverage1 : {err_cov1:.02%}\nError Coverage2 : {err_cov2:.02%}'
        )
        print('\n')

        fn = f'{self.queue.out_dir}/result/result_{iteration}.pickle'
        with open(fn, 'wb') as f:
            pickle.dump(self.queue.DUMPS, f)

    def loop(self, iterations):
        """Fuzzes a machine learning model in a loop, making *iterations* steps."""
        iteration = 0
        while True:

            if len(self.queue.queue) < 1 or iteration >= iterations:
                break
            # if iteration % 100 == 0:
            if True:
                # tf.logging.info("fuzzing iteration: %s", iteration)
                # print("fuzzing iteration: %s"%(iteration))
                gc.collect()

            selected_parent_list = []
            mutated_data_batches = []

            for __i in range(4):
                T_p, T_db = self.mutation_function()
                selected_parent_list.append(T_p)
                mutated_data_batches.append(T_db)

                scene = T_p.root_seed.split('.')[0]

                # calculate frd
                frd_score = get_fid(
                    self.frd_function(
                        self.queue.DUMPS['scene'][scene]['points']),
                    self.frd_function(mutated_data_batches[-1]['points']))
                frd_limit = self.queue.DUMPS['frd_limit']
                if frd_score > frd_limit:
                    selected_parent_list.pop()
                    mutated_data_batches.pop()
                    print(
                        f'## Exceeded FRD Score Limit ## Current Score : {frd_score}, Limit : {frd_limit}'
                    )

            for mutated_data_batch in mutated_data_batches:
                coverage_batches, metadata_batches = self.fetch_function(
                    [mutated_data_batch])

                mutated_data_batch['pred_scores'] = coverage_batches[0]
                mutated_data_batch['pred_boxes'] = metadata_batches[0]

            if len(mutated_data_batches) > 0:
                self.iterate_function(self.queue, selected_parent_list,
                                      mutated_data_batches,
                                      self.objective_function)
            else:
                print(f'## All Exceeded FRD Score Limit ##')

                DUMPS_utlis.updateIter_DUMPS_errorMetric(self.queue.DUMPS)

            # del mutated_data_batches
            # del coverage_batches
            # del metadata_batches

            iteration += 1

            print('\n')

            DUMPS_utlis.updateCoverage_DUMPS(self.queue.DUMPS)

            if self.queue.check_point != 0:
                if (iteration) % self.queue.check_point == 0:
                    self.save(iteration)
                elif iteration == iterations:
                    self.save(iteration)
            else:
                self.save(iteration)

        return None

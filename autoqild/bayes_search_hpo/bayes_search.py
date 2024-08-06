import logging

import dill
import numpy as np
from sklearn.utils import check_random_state
from skopt import BayesSearchCV as BayesSearchCVSK
from skopt.utils import eval_callbacks, point_asdict

from autoqild import log_exception_error

__all__ = ["BayesSearchCV"]
class BayesSearchCV(BayesSearchCVSK):
    def __init__(
            self,
            estimator,
            search_spaces,
            optimizer_kwargs=None,
            n_iter=50,
            scoring=None,
            fit_params=None,
            n_jobs=1,
            n_points=1,
            iid=True,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch="2*n_jobs",
            random_state=None,
            error_score="raise",
            return_train_score=False,
            optimizers_file_path="results.pkl"
    ):
        super().__init__(
            estimator,
            search_spaces,
            optimizer_kwargs,
            n_iter,
            scoring,
            fit_params,
            n_jobs,
            n_points,
            iid,
            refit,
            cv,
            verbose,
            pre_dispatch,
            random_state,
            error_score,
            return_train_score,
        )
        self.optimizers_file_path = optimizers_file_path
        self.logger = logging.getLogger(BayesSearchCV.__name__)

    def _step(self, search_space, optimizer, evaluate_candidates, n_points=1):
        """Generate n_jobs parameters and evaluate them in parallel.
        """
        # get parameter values to evaluate
        params = optimizer.ask(n_points=n_points)

        # convert parameters to python native types
        params = [[np.array(v).item() for v in p] for p in params]

        # make lists into dictionaries
        params_dict = [point_asdict(search_space, p) for p in params]
        self.logger.info(f"Parameters values to be tested {params}")
        try:
            all_results = evaluate_candidates(params_dict)
            local_results = all_results["mean_test_score"][-len(params):]
        except Exception as e:
            local_results = list(np.zeros(len(params)))
            self.logger.info(params_dict)
            log_exception_error(self.logger, e)
        # Feed the point and objective value back into optimizer
        # Optimizer minimizes objective, hence provide negative score

        return optimizer.tell(params, [-score for score in local_results])

    def _run_search(self, evaluate_candidates):
        # check if space is a single dict, convert to list if so
        search_spaces = self.search_spaces
        if isinstance(search_spaces, dict):
            search_spaces = [search_spaces]

        callbacks = self._callbacks

        random_state = check_random_state(self.random_state)
        self.optimizer_kwargs_['random_state'] = random_state

        # Instantiate optimizers for all the search spaces.
        try:
            optimizers, optim_results = dill.load(open(self.optimizers_file_path, "rb"))
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error(f"No such file or directory: {self.optimizers_file_path}")
            optimizers = None
            optim_results = []
        if optimizers is None:
            optimizers = []
            for search_space in search_spaces:
                if isinstance(search_space, tuple):
                    search_space = search_space[0]
                optimizers.append(self._make_optimizer(search_space))
            self.optimizers_ = optimizers  # will save the states of the optimizers
            self._optim_results = [0 for o in optimizers]
        else:
            self._optim_results = optim_results
            self.optimizers_ = optimizers

        n_points = self.n_points

        for i, (search_space, optimizer) in enumerate(zip(search_spaces, optimizers)):
            # if not provided with search subspace, n_iter is taken as
            # self.n_iter
            if isinstance(search_space, tuple):
                search_space, n_iter = search_space
            else:
                n_iter = self.n_iter
            n_finished = len(optimizer.yi)
            n_iter = n_iter - n_finished
            self.logger.info(f"Iterations already done: {n_finished} and running iterations {n_iter}")
            # do the optimization for particular search space
            optim_result = None
            iter_idx = 0
            while n_iter > 0:
                # when n_iter < n_points points left for evaluation
                n_points_adjusted = min(n_iter, n_points)
                iter_idx += n_points
                self.logger.info(f"The {iter_idx + n_finished}th parameter values are being tested")
                try:
                    optim_result = self._step(search_space, optimizer, evaluate_candidates, n_points=n_points_adjusted)
                except Exception as error:
                    log_exception_error(self.logger, error)
                    self.logger.info(f"Cannot evaluate the points {n_points_adjusted}")
                n_iter -= n_points
                if eval_callbacks(callbacks, optim_result):
                    break
                self._optim_results[i] = optim_result
                dill.dump((self.optimizers_, self._optim_results), open(self.optimizers_file_path, "wb"))
            if optim_result is not None:
                self._optim_results[i] = optim_result
            dill.dump((self.optimizers_, self._optim_results), open(self.optimizers_file_path, "wb"))

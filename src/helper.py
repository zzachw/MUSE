import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from itertools import chain

import neptune
import numpy as np
import torch

import src.credentials as credentials
from src.utils import project_path, remote_project_path, create_directory, set_seed


class Helper:

    def __init__(self, parse_arguments_fn):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        # set_args
        parser = argparse.ArgumentParser()
        args = parse_arguments_fn(parser)
        args.exp_name = self.get_exp_name(*[getattr(args, attr) for attr in args.exp_name_attr])
        args.checkpoint = os.path.join(remote_project_path, f"output/{args.checkpoint}") if args.checkpoint else None
        args.device = self.get_device(args.no_cuda)
        self.args = args

        # set_output_folder
        output_saved_path = os.path.join(remote_project_path, f"output/{args.exp_name}")
        model_saved_path = os.path.join(output_saved_path, "models")
        prediction_saved_filename = os.path.join(output_saved_path, "predictions.txt")
        log_saved_filename = os.path.join(output_saved_path, "log.txt")
        create_directory(output_saved_path)
        create_directory(model_saved_path)
        self.output_saved_path = output_saved_path
        self.model_saved_path = model_saved_path
        self.prediction_saved_filename = prediction_saved_filename
        self.log_saved_filename = log_saved_filename

        # set_logging
        file_handler = logging.FileHandler(log_saved_filename)
        logging.getLogger().addHandler(file_handler)
        neptune_logger = None
        if args.official_run:
            neptune_logger = neptune.init_run(api_token=credentials.NEPTUNE_API_KEY,
                                              project=credentials.NEPTUNE_PROJECT,
                                              name=args.exp_name,
                                              description=args.note,
                                              source_files=self.get_source_file_list())
        self.neptune_logger = neptune_logger

        # set_seed
        set_seed(args.seed)

        # log_properties
        args_dict = vars(args)
        for k, v in args_dict.items():
            self.log(k, str(v))

        self.best_score = -1 * float("inf") if args.monitor_criterion == "max" else float("inf")

        return

    def log(self, k: str, v: str):
        logging.info(f"{k}: {v}")
        if self.neptune_logger:
            self.neptune_logger[k].log(v)

    def save_checkpoint(self, model, name):
        state_dict = model.state_dict()
        path = os.path.join(self.model_saved_path, name)
        torch.save(state_dict, path)
        logging.info(f"Save checkpoint to: {path}")
        return

    def load_checkpoint(self, model, path):
        model.load_state_dict(torch.load(path, map_location=self.args.device))
        logging.info(f"Load checkpoint from: {path}")
        return

    def save_checkpoint_if_best(self, model, name, scores):
        save = False
        score = scores[self.args.monitor]
        if self.args.monitor_criterion == "max":
            if score > self.best_score:
                save = True
        elif self.args.monitor_criterion == "min":
            if score < self.best_score:
                save = True
        else:
            raise NotImplementedError
        if save:
            logging.info(f"New best {self.args.monitor} score ({score:.4f})")
            self.best_score = score
            self.save_checkpoint(model, name)
        return

    def save_predictions(self, predictions):
        logging.info(f"Save predictions to: {self.prediction_saved_filename}")
        np.savetxt(self.prediction_saved_filename, predictions, delimiter=",", fmt="%s")
        return

    @staticmethod
    def get_device(no_cuda=False):
        cuda = torch.cuda.is_available() and (not no_cuda)
        device = torch.device("cuda" if cuda else "cpu")
        logging.info(f"Device: {device}")
        return device

    @staticmethod
    def get_exp_name(*args, **kwargs):
        """ exp name will be: {date}-{time}-{*args} """
        args = [arg for arg in args if arg.strip()]  # remove empty arg
        kwargs = [f"{k}:{v}" for k, v in kwargs.items()]
        return "-".join([datetime.now().strftime("%y%m%d-%H%M%S")] + [*args] + [*kwargs])

    @staticmethod
    def get_source_file_list(suffix=None):

        def nested_list_reduce(nested_list):
            return list(chain(*nested_list))

        if suffix is None:
            suffix = ["py", "ipynb"]
        return nested_list_reduce([glob.glob(f"{project_path}/src/**/*.{suffix}", recursive=True) for suffix in suffix])

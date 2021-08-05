import json
import logging
import os
import signal
from pathlib import Path

import paddle


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def latest_checkpoint(model_dir, model_name):
    """return path of latest checkpoint in a model_dir
    Args:
        model_dir: string, indicate your model dir(save pds, summarys,
            logs, etc).
        model_name: name of your model. we find pds by name
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """
    pd_info_path = Path(model_dir) / "checkpoints.json"
    if not pd_info_path.is_file():
        return None
    with open(pd_info_path, 'r') as f:
        pd_dict = json.loads(f.read())
    if model_name not in pd_dict['latest_pd']:
        return None
    latest_pd = pd_dict['latest_pd'][model_name]
    pd_file_name = Path(model_dir) / latest_pd
    if not pd_file_name.is_file():
        return None
    
    return str(pd_file_name)

def _ordered_unique(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def save(model_dir,
         model,
         model_name,
         global_step,
         max_to_keep=8,
         keep_latest=True):
    """save a model into model_dir.
    Args:
        model_dir: string, indicate your model dir(save pds, summarys,
            logs, etc).
        model: paddle.nn.Layer instance.
        model_name: name of your model. we find pds by name
        global_step: int, indicate current global step.
        max_to_keep: int, maximum checkpoints to keep.
        keep_latest: bool, if True and there are too much pds,
            will delete oldest pd. else will delete pd which has
            smallest global step.
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """

    # prevent save incomplete checkpoint due to key interrupt
    with DelayedKeyboardInterrupt():
        pd_info_path = Path(model_dir) / "checkpoints.json"
        pd_filename = "{}-{}.pdparams".format(model_name, global_step)
        pd_path = Path(model_dir) / pd_filename
        if not pd_info_path.is_file():
            pd_info_dict = {'latest_pd': {}, 'all_pds': {}}
        else:
            with open(pd_info_path, 'r') as f:
                pd_info_dict = json.loads(f.read())
        pd_info_dict['latest_pd'][model_name] = pd_filename
        if model_name in pd_info_dict['all_pds']:
            pd_info_dict['all_pds'][model_name].append(pd_filename)
        else:
            pd_info_dict['all_pds'][model_name] = [pd_filename]
        all_pds = pd_info_dict['all_pds'][model_name]

        paddle.save(model.state_dict(), pd_path)
        # check pd in all_pds is exist, if not, delete it from all_pds
        all_pds_checked = []
        for pd in all_pds:
            pd_path_uncheck = Path(model_dir) / pd
            if pd_path_uncheck.is_file():
                all_pds_checked.append(str(pd_path_uncheck))
        all_pds = all_pds_checked
        if len(all_pds) > max_to_keep:
            if keep_latest:
                pd_to_delete = all_pds.pop(0)
            else:
                # delete smallest step
                get_step = lambda name: int(name.split('.')[0].split('-')[1])
                min_step = min([get_step(name) for name in all_pds])
                pd_to_delete = "{}-{}.pdparams".format(model_name, min_step)
                all_pds.remove(pd_to_delete)

            os.remove( pd_to_delete)
        all_pds_filename = _ordered_unique([Path(f).name for f in all_pds])
        pd_info_dict['all_pds'][model_name] = all_pds_filename
        with open(pd_info_path, 'w') as f:
            f.write(json.dumps(pd_info_dict, indent=2))


def restore(pd_path, model):
    if not Path(pd_path).is_file():
        raise ValueError("checkpoint {} not exist.".format(pd_path))
    model.load_state_dict(paddle.load(pd_path))
    print("Restoring parameters from {}".format(pd_path))


def _check_model_names(models):
    model_names = []
    for model in models:
        if not hasattr(model, "name"):
            raise ValueError("models must have name attr")
        model_names.append(model.name)
    if len(model_names) != len(set(model_names)):
        raise ValueError("models must have unique name: {}".format(
            ", ".join(model_names)))


def _get_name_to_model_map(models):
    if isinstance(models, dict):
        name_to_model = {name: m for name, m in models.items()}
    else:
        _check_model_names(models)
        name_to_model = {m.name: m for m in models}
    return name_to_model


def try_restore_latest_checkpoints(model_dir, models):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        latest_pd = latest_checkpoint(model_dir, name)
        if latest_pd is not None:
            restore(latest_pd, model)

def restore_latest_checkpoints(model_dir, models):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        latest_pd = latest_checkpoint(model_dir, name)
        if latest_pd is not None:
            restore(latest_pd, model)
        else:
            raise ValueError("model {}\'s pd isn't exist".format(name))

def restore_models(model_dir, models, global_step):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        pd_filename = "{}-{}.pdparams".format(name, global_step)
        pd_path = model_dir + "/" + pd_filename
        restore(pd_path, model)


def save_models(model_dir,
                models,
                global_step,
                max_to_keep=15,
                keep_latest=True):
    with DelayedKeyboardInterrupt():
        name_to_model = _get_name_to_model_map(models)
        for name, model in name_to_model.items():
            save(model_dir, model, name, global_step, max_to_keep, keep_latest)

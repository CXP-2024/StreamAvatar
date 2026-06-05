from importlib import import_module
from omegaconf import OmegaConf
import os
from pathlib import Path
import shutil
from omegaconf import DictConfig
from lightning.pytorch.utilities import rank_zero_info

def instantiate(config: DictConfig, instantiate_module=True):
    """Get arguments from config."""
    module = import_module(config.module_name)
    class_ = getattr(module, config.class_name)
    if instantiate_module:
        init_args = {k: v for k, v in config.items() if k not in ["module_name", "class_name"]}
        return class_(**init_args)
    else:
        return class_

def instantiate_motion_gen(module_name, class_name, cfg, hfstyle=False, **init_args):
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    if hfstyle:
        config_class = class_.config_class
        cfg = config_class(config_obj=cfg)
    return class_(cfg, **init_args)

def save_config_and_codes(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sanity_check_dir = os.path.join(save_dir, 'sanity_check')
    os.makedirs(sanity_check_dir, exist_ok=True)
    with open(os.path.join(sanity_check_dir, f'{config.exp_name}.yaml'), 'w') as f:
        OmegaConf.save(config, f)
    current_dir = Path.cwd()
    for py_file in current_dir.rglob('*.py'):
        dest_path = Path(sanity_check_dir) / py_file.relative_to(current_dir)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(py_file, dest_path)

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_zero_info(f"Total parameters: {total_params:,}")
    rank_zero_info(f"Trainable parameters: {trainable_params:,}")
    rank_zero_info(f"Non-trainable parameters: {(total_params - trainable_params):,}")

def load_metrics(file_path):
    metrics = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            try:
                metrics[key] = float(value)  # Convert to float if possible
            except ValueError:
                metrics[key] = value  # Keep as string if conversion fails
    return metrics

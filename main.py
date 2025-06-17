import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset.data_module import DataModule
from src.misc.step_tracker import StepTracker
from src.config import load_typed_root_config

@hydra.main(version_base=None, config_path="conf", config_name="base")
def my_app(cfg : DictConfig) -> None:

    '''get cfg'''
    print(OmegaConf.to_yaml(cfg))
    cfg_typed = load_typed_root_config(cfg) # make sure each config is typed
    '''Dataset intialization'''
    step_tracker = StepTracker(cfg.train.step_offset)
    data_module = DataModule(cfg_typed.dataset, cfg_typed.data_loader, step_tracker)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

if __name__ == "__main__":
    my_app()
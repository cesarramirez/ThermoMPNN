import sys

# Importing weights and biases module
import wandb

# Importing torch, the neural network module (nn) and the stateless functional module (functional)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing torch dataloader
from torch.utils.data import DataLoader

# Importing pytorch lightning for enabling checkpointing and logs
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Importing metrics that will be computed afterwards
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef

# Importing OmegaConf to handle configurations from multiple sources
from omegaconf import OmegaConf

# Importing the transfer model module, as ThermoMPNN does transfer learning
from transfer_model import TransferModel

# Importing the datasets of FireProt, Megascale and Combo, indicated in datasets.py
from datasets import FireProtDataset, MegaScaleDataset, S4038Dataset, ComboDataset

# Defining the metrics to be measured
def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }

# Defining the transfer model hyperparameters, which are passed in cfg via the config.yaml file
class TransferModelPL(pl.LightningModule):
    """Class managing training loop with pytorch lightning"""
# Taking the cfg configuration
    def __init__(self, cfg):
# Calling the parent class pl.LightningModule
        super().__init__()
# Passing the cfg configuration
        self.model = TransferModel(cfg)
# There are two separate learning rates, one for the model and another for MPNN,
# the latter can be set in the config.yaml file
# There is also a learning rate scheduler setup that can be set to true in the config.yaml file
        self.learn_rate = cfg.training.learn_rate
        self.mpnn_learn_rate = cfg.training.mpnn_learn_rate if 'mpnn_learn_rate' in cfg.training else None
        self.lr_schedule = cfg.training.lr_schedule if 'lr_schedule' in cfg.training else False

# Setting up an empty dictionary to hold metrics
        self.metrics = nn.ModuleDict()
# Looping over training and validation metrics splits
        for split in ("train_metrics", "val_metrics"):
# Storing the metrics of each split in the dictionary
            self.metrics[split] = nn.ModuleDict()
# Metrics will be tracked based on the ddG output
            out = "ddG"
# Storing the output for each split in the dictionary
            self.metrics[split][out] = nn.ModuleDict()
# Obtaining the error metrics for each split
            for name, metric in get_metrics().items():
                self.metrics[split][out][name] = metric

# Defining a single pass of the data through the Transfer model
    def forward(self, *args):
        return self.model(*args)

# This section sets the batches and their ids, whereas the prefix indicates if its tran, val or test
    def shared_eval(self, batch, batch_idx, prefix):

# Passing a single PDB and its mutations
        assert len(batch) == 1
        mut_pdb, mutations = batch[0]
# Here is the prediction after passing through the model
        pred, _ = self(mut_pdb, mutations)

# Here we are defining the loss function based on the predicted and known ddG for each mutation
        ddg_mses = []
        for mut, out in zip(mutations, pred):
# Here we are setting that this is only evaluated for mutations with known ddG in the dataset
            if mut.ddG is not None:
# A simple mean square error function for the loss
                ddg_mses.append(F.mse_loss(out["ddG"], mut.ddG))
                for metric in self.metrics[f"{prefix}_metrics"]["ddG"].values():
# Update metrics
                    metric.update(out["ddG"], mut.ddG)

# If no ddG data, then loss is 0, else calculate the mean
        loss = 0.0 if len(ddg_mses) == 0 else torch.stack(ddg_mses).mean()
# Metrics are logged on every epoch
        on_step = False
        on_epoch = not on_step

        output = "ddG"
# Here we are computing the metrics
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                        batch_size=len(batch))
# Returning the loss for backpropagation
        if loss == 0.0:
            return None
        return loss

# Here we predict the values, losses and metrics for each batch in the training set
# From these losses, the gradient computations and weights are updated
    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

# This is for monitoring the performance to make adjustments to the model
    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

# Assessing the performance of the model on unseen data
    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

# This is the learning scheduler, we are dropping the learning rate hyperparameter by 10
    def configure_optimizers(self):
        if self.stage == 2: # for second stage, drop LR by factor of 10
            self.learn_rate /= 10.
            print('New second-stage learning rate: ', self.learn_rate)

# This is what is needed to update the weights of ProteinMPNN if needed
        if not cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": self.mpnn_learn_rate}]
        else: # fully frozen MPNN
            param_list = []

# Here is the light attention module parameters
        if self.model.lightattn:  # adding light attention parameters
# Here we are freezing the parameters in the second stage
            if self.stage == 2:
                param_list.append({"params": self.model.light_attention.parameters(), "lr": 0.})
            else:
                param_list.append({"params": self.model.light_attention.parameters()})

# Setting the parameters of the MLP that will be optimized during the training
        mlp_params = [
            {"params": self.model.both_out.parameters()},
            {"params": self.model.ddg_out.parameters()}
            ]

        param_list = param_list + mlp_params

# Our optimizer is the SGD method AdamW (weight and gradient decays are decoupled)
        opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

# If using learning rate scheduler, reduce the learning rate by 0.5 when val_ddG_mse is plateauing
        if self.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, verbose=True, mode='min', factor=0.5)
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': 'val_ddG_mse'
            }
        else:
            return opt

# Here we are defining the training datasets and configurations in the config.yaml file
def train(cfg):
    print('Configuration:\n', cfg)

# If there is a project in the yaml file, then we will active the tracking of weights and biases
    if 'project' in cfg:
        wandb.init(project=cfg.project, name=cfg.name)
    else:
        cfg.name = 'test'

# Checks whether we are loading a single or multiple data sets in combination
# Depends on the lenght of the list in the yaml file, with all dirs indicated in the local.yaml file
    if len(cfg.datasets) == 1: # one dataset training
        dataset = cfg.datasets[0]
        if dataset == 'fireprot':
            train_dataset = FireProtDataset(cfg, "train")
            val_dataset = FireProtDataset(cfg, "val")
        elif dataset == 'S4038':
            train_dataset = S4038Dataset(cfg, "train")
            val_dataset = S4038Dataset(cfg, "val")
        elif dataset == 'megascale_s669':
            train_dataset = MegaScaleDataset(cfg, "train_s669")
            val_dataset = MegaScaleDataset(cfg, "val")
        elif dataset.startswith('megascale_cv'):
                cv = dataset[-1]
                train_dataset = MegaScaleDataset(cfg, f"cv_train_{cv}")
                val_dataset = MegaScaleDataset(cfg, f"cv_val_{cv}")
        elif dataset == 'megascale':
                train_dataset = MegaScaleDataset(cfg, "train")
                val_dataset = MegaScaleDataset(cfg, "val")
        else:
            raise ValueError("Invalid dataset specified!")
    else:
        train_dataset = ComboDataset(cfg, "train")
        val_dataset = ComboDataset(cfg, "val")

# Setting the number of workers for Dataset Loading
    if 'num_workers' in cfg.training:
        train_workers, val_workers = int(cfg.training.num_workers * 0.75), int(cfg.training.num_workers * 0.25)
    else:
        train_workers, val_workers = 0, 0

# Setting up the Data Loaders for training and validation
# The lambda passes the information as is, without collation
    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
    val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)

# Here we are finally calling the Transfer Model
    model_pl = TransferModelPL(cfg)
    model_pl.stage = 1

# We are monitoring the spearman against the validation to save the model
    filename = cfg.name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    monitor = 'val_ddG_spearman'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath='checkpoints', filename=filename)
    logger = WandbLogger(project=cfg.project, name="test", log_model="all") if 'project' in cfg else None
# Training for the number of epochs indicated in the yaml file, otherwise 100 epochs
    max_ep = cfg.training.epochs if 'epochs' in cfg.training else 100
#Training on either CPU or GPU based on the local.yaml file
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        max_epochs=max_ep,
        accelerator=cfg.platform.accel,
        devices=1,
        strategy='auto',
        use_distributed_sampler=False,
        plugins=[])
    trainer.fit(model_pl, train_loader, val_loader)

# If we want to train in two stages, we can do different sets of training and validation
    if 'two_stage' in cfg.training:  # sequential combo training
        if cfg.training.two_stage:
            print('Two-stage Training Enabled')
            del trainer, train_dataset, val_dataset, train_loader, val_loader
            # load new datasets for further training
            train_dataset = FireProtDataset(cfg, "train")
            val_dataset = MegaScaleDataset(cfg, "val")
            train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
            val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)

            model_pl.stage = 2
            # re-start training with a new trainer
            trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep * 2,
                                accelerator=cfg.platform.accel, devices=1)
            trainer.fit(model_pl, train_loader, val_loader, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments
    if len(sys.argv) == 1:
        yaml = "config.yaml"
    else:
        yaml = sys.argv[1]

    cfg = OmegaConf.load(yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.load("local.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    train(cfg)

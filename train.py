import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

from src.model import Model
from src.datamodules import get_datamodule
from src.utils import hparams_from_config

parser = ArgumentParser()
parser.add_argument('-c', '--config_path', type=Path, help='Path to the config.', default='.') # TODO: define defaults in model, use LitModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

# Load hparams from config file
hparams = hparams_from_config(args.config_path)
args.max_epochs = hparams.epochs

# Define callbacks
tb_logger = TensorBoardLogger(
    save_dir=hparams.output_path,
    name=hparams.experiment_name
)
csv_logger = CSVLogger(
    save_dir=hparams.output_path,
    name=hparams.experiment_name
)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(tb_logger.root_dir, 'best-{epoch}-{val_acc:.4f}'),
    save_top_k=1,
    verbose=True,
    monitor='val_acc',
    mode='max',
    save_last=True,
)

# Load datamodule
dm = get_datamodule(hparams)
hparams.n_classes = dm.n_classes
hparams.dims = dm.dims

# Load model
model = Model(hparams)

# Run trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    #resume_from_checkpoint=hparams.checkpoint if hparams.checkpoint else None,
    checkpoint_callback=checkpoint_callback,
    logger=[tb_logger, csv_logger],
)

trainer.fit(model, dm)

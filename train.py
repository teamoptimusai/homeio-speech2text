import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import argparse
from model.model import SpeechRecognition
from utils.dataset import Dataset, collate_fn_padd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, QuantizationAwareTraining, ModelSummary, RichProgressBar, DeviceStatsMonitor
import yaml
from datetime import datetime
import os


class Speech2Text(LightningModule):
    def __init__(self, model, opt, data_params):
        super(Speech2Text, self).__init__()
        self.model = model
        self.opt = opt
        self.data_params = data_params
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.opt.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': self.optimizer, 'scheduler': self.scheduler, 'monitor': 'val_loss'}

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss.item()}
        print("Validation Loss:", avg_loss.item())
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        train_dataset = Dataset(
            json_path=self.opt.train_file, **self.data_params)
        return DataLoader(dataset=train_dataset,
                          batch_size=self.opt.batch_size,
                          num_workers=self.opt.data_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_padd)

    def val_dataloader(self):
        test_dataset = Dataset(json_path=self.opt.valid_file,
                               **self.data_params, valid=True)
        return DataLoader(dataset=test_dataset,
                          batch_size=self.opt.batch_size,
                          num_workers=self.opt.data_workers,
                          collate_fn=collate_fn_padd,
                          pin_memory=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # distributed learning (DDP)
    parser.add_argument('-n', '--nodes', default=1, type=int,
                        help='Number of DataLoader Workers')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='Number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=8, type=int,
                        help='n data loading workers, default 0 = main process only')
    # Dataset
    parser.add_argument('--train_file', default=None, required=True, type=str,
                        help='JSON file to load Training data')
    parser.add_argument('--valid_file', default=None, required=True, type=str,
                        help='JSON file to load Validation data')

    # Logging and Saving Model
    parser.add_argument('--weights', default=None, required=False, type=str,
                        help='Path to the pretrained weights to continue training')
    parser.add_argument('--logdir', default='runs', required=False, type=str,
                        help='Path to save TensorBoard logs')
    parser.add_argument('--name', default=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), required=True, type=str,
                        help='Name of the run')

    # General
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of Epochs')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='Size of batch')
    parser.add_argument('--pct_start', default=0.3, type=float,
                        help='Percentage of growth phase in one cycle')
    parser.add_argument('--div_factor', default=100,
                        type=int, help='div factor for one cycle')
    parser.add_argument('--params', default=None, required=True,
                        type=str, help='YAML file to load parameters')
    parser.add_argument('--lr', default=1e-4,
                        type=float, help='Learning Rate')

    opt = parser.parse_args()
    print("arguments:", opt)

    with open(opt.params, 'r') as stream:
        config = yaml.safe_load(stream)
    dataset_params, hyp_params, lr = config.values()
    print("Dataset Parameters", dataset_params)
    print("Hyperparameters", hyp_params)

    model = SpeechRecognition(**hyp_params)

    if opt.weights:
        speech2text = Speech2Text.load_from_checkpoint(
            opt.weights, model=model, opt=opt, data_params=dataset_params)
    else:
        speech2text = Speech2Text(model, opt, dataset_params)

    # increment a number if the folder already exists
    if os.path.exists(os.path.join(opt.logdir, opt.name)):
        i = 1
        while os.path.exists(os.path.join(opt.logdir, opt.name + str(i))):
            i += 1
        opt.name = opt.name + str(i)
    os.makedirs(os.path.join(opt.logdir, opt.name))
    print("Saving Checkpoints at", os.path.join(opt.logdir, opt.name))

    logger = TensorBoardLogger(save_dir=opt.logdir, name=opt.name)

    best_callback = ModelCheckpoint(
        dirpath=f'{opt.logdir}/{opt.name}',
        filename='speech2text_best',
        monitor='val_loss',
        mode='min'
    )

    last_callback = ModelCheckpoint(
        dirpath=f'{opt.logdir}/{opt.name}',
        save_last=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{opt.logdir}/{opt.name}',
        filename='speech2text_{epoch:02d}',
        every_n_epochs=opt.epochs // 4
    )

    trainer = Trainer(
        max_epochs=opt.epochs, gpus=-1,
        auto_select_gpus=True,
        num_nodes=opt.nodes,
        logger=logger,
        check_val_every_n_epoch=2,
        auto_scale_batch_size=True,
        auto_lr_find=True,
        callbacks=[best_callback, last_callback, checkpoint_callback,
                   ModelSummary(), RichProgressBar(), DeviceStatsMonitor()],
        strategy='ddp'
    )
    trainer.fit(speech2text)

import copy
import datetime
import random
import traceback

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from dnnutils.tensorboard import SummaryWriterHelper


def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    elif isinstance(data, (dict, tuple)):
        return {key: copy_data_to_device(elem, device) for key, elem in data.items()}
    raise ValueError('Unsupported data type {}'.format(type(data)))


def print_grad_stats(model):
    mean = 0
    std = 0
    norm = 1e-5
    for param in model.parameters():
        grad = getattr(param, 'grad', None)
        if grad is not None:
            mean += grad.data.abs().mean()
            std += grad.data.std()
            norm += 1
    mean /= norm
    std /= norm
    print(f'Mean grad {mean}, std {std}, n {norm}')


def train_eval_loop(model, train_dataset, val_dataset, criterion,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=100000,
                    max_batches_per_epoch_val=10000,
                    data_loader_ctor=DataLoader,
                    train_sampler=None,
                    val_sampler=None,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0,
                    tensorboard: SummaryWriterHelper=None,
                    on_epoch_cb=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n, sampler=train_sampler)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n, sampler=val_sampler)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)
    
    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print('Epoch {}'.format(epoch_i))


            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(tqdm.tqdm(train_dataloader, total=len(train_dataloader))):
                if batch_i > max_batches_per_epoch_train:
                    break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += float(loss)
                train_batches_n += 1
                if tensorboard:
                    tensorboard.add_scalar('train/loss', loss)

            mean_train_loss /= train_batches_n
            print('Epoch: {} iteration, {:0.2f} sec'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Mean value of loss function on learning', mean_train_loss)

            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(tqdm.tqdm(val_dataloader, total=len(val_dataloader))):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)

                    mean_val_loss += float(loss)
                    val_batches_n += 1
                    if tensorboard:
                        tensorboard.add_scalar('val/loss', loss)

            mean_val_loss /= val_batches_n
            print('Mean value of loss function on validation', mean_val_loss)
            
            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('New best model!')
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Model didnt improve for last {} epoch, interrupt learing'.format(
                    early_stopping_patience))
                break
            
            if on_epoch_cb is not None:
                on_epoch_cb(epoch_i, model, tensorboard)

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)
                
        except KeyboardInterrupt:
            print('Interrupted by user')
            break
        except Exception as ex:
            print('Exception while learning: {}\n{}'.format(ex, traceback.format_exc()))
            break

    return best_val_loss, best_model


def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0, return_labels=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        import tqdm
        for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataset)/batch_size):
            batch_x = copy_data_to_device(batch_x, device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)

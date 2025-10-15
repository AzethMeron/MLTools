import torch
from torch import nn
from collections import deque
from typing import List, Optional, Callable
import os
import math

# ---------- Loss counters (small fixes) ----------

class RecentLossCounter:
    def __init__(self, memory: int = 10):
        self.memory = memory
        self.running_sum = 0.0
        self.que = deque(maxlen=memory)

    def add(self, val: float):
        if len(self.que) == self.memory:
            dropped = self.que[0]
            self.running_sum -= dropped
        self.que.append(float(val))
        self.running_sum += float(val)

    def value(self) -> float:
        return self.running_sum / len(self.que) if self.que else 0.0

    def __len__(self): 
        return len(self.que)


class AverageLossCounter:
    def __init__(self):
        self.total_sum = 0.0
        self.count = 0

    def add(self, val: float):
        self.total_sum += float(val)
        self.count += 1

    def value(self) -> float:
        return self.total_sum / self.count if self.count > 0 else 0.0

    def __len__(self): 
        return self.count


# ---------- Storage that batches on GPU and moves once per flush ----------

class TensorBatchStorage:
    """
    Accumulates detached GPU (or CPU) tensors and, on value(), returns
    ONE concatenated CPU tensor. Works for scalars and higher-dim tensors.

    Assumptions:
      - All added tensors are compatible for torch.cat along `cat_dim`
        (same shape except along cat_dim).
      - Scalars (0-D) are auto-unsqueezed to 1-D for concatenation when needed.
    """
    def __init__(self, cat_dim: int = 0, auto_unsqueeze_scalars: bool = True):
        self._store: List[torch.Tensor] = []
        self.cat_dim = cat_dim
        self.auto_unsqueeze_scalars = auto_unsqueeze_scalars

    def add(self, t: torch.Tensor):
        t = t.detach()
        if t.ndim == 0 and self.auto_unsqueeze_scalars:
            t = t.unsqueeze(0)  # scalars -> [1]
        self._store.append(t)

    def value(self) -> torch.Tensor:
        """Concatenate on the tensors' current device, then move once to CPU."""
        if not self._store: return torch.empty(0, dtype=torch.float32)  # empty 1-D
        cat_gpu = torch.cat(self._store, dim=self.cat_dim)
        cat_cpu = cat_gpu.cpu() 
        return cat_cpu

    def clear(self):
        self._store.clear()

    def __len__(self): 
        return len(self._store)


class DelayedStorage:
    """
    Buffers K tensors and, when K is reached, emits ONE concatenated CPU tensor.
    This minimizes CPU transfers (1 per K batches).
    """
    def __init__(self, itemize_every_k: int = 10, cat_dim: int = 0, auto_unsqueeze_scalars: bool = True):
        self._k = int(itemize_every_k)
        self._storage = TensorBatchStorage(cat_dim=cat_dim, auto_unsqueeze_scalars=auto_unsqueeze_scalars)
        self._last_value: Optional[torch.Tensor] = None

    def add(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        self._storage.add(t)
        if len(self._storage) >= self._k:
            self._last_value = self._storage.value()  # CPU tensor
            self._storage.clear()
            return self._last_value
        return None

    def last_value(self) -> Optional[torch.Tensor]:
        return self._last_value

    def __len__(self): 
        return len(self._storage)

# Versatile training loop class, which can be easily expanded to accomodate pretty much any usecase.
# Functions to be overloaded:
# train_batch, test_batch, compute_metrics, print_epoch_results, update_pbar, quantify
# Although basic implementation is sane, so if your dataloader returns pairs (inputs, targets) and model processes outputs = model(inputs), it should work as-is
# train_loop_constructor, test_loop_constructor must be callable and return tqdm.tqdm. You should pass something like lambda: tqdm.tqdm(train_loader, mininterval=1.0)
# I strongly recommand using mininterval=1.0 or more.
#   log_every_k - outputs, targets, losses are kept at original device (GPU) until every K, then it's moved to CPU. Saves some time.
#                 This causes last K batches to be dropped, skipped - for big tasks negligible.
#                 If outputs and targets are big, I recommand setting K to lower value (K=4).
#   keep_outputs - stores all of the outputs and targets in tensors on CPU, for later computation of metrics.
#                 Set to False if memory is an issue
# Scheduler is called once per epoch.
# Also I recommand setting pin_memory=True, persistent_workers=True in dataloader. However if you don't set pin_memory, remember to set non_blocking = False. By default it's set to True
class TrainingLoop:
    def __init__(self, model: nn.Module, optimizer, criterion, device, epochs: int, train_loop_constructor: Callable, test_loop_constructor: Callable, log_every_k: int = 10, recent_memory: int = 10, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, checkpoint_path: Optional[str] = None, best_path: Optional[str] = None, non_blocking: bool = True, keep_outputs = True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler or torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        self.criterion = criterion 
        self.train_loop_constructor = train_loop_constructor
        self.test_loop_constructor = test_loop_constructor
        self.recent_memory = recent_memory
        self.log_every_k = log_every_k
        self.checkpoint_path = checkpoint_path
        self.best_path = best_path
        self.loaded = False
        self.best_val = math.inf
        self.epoch = 0
        self.num_epochs = epochs
        self.device = device
        self.non_blocking = non_blocking
        self.history = [] # history[epoch_id] = {'epoch', 'train_loss', 'test_loss', 'train_metrics', 'test_metrics'}
        self.keep_outputs = keep_outputs
        
    def load_checkpoint(self, path): # Sometimes it's worth to overload this function
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optim"])
            self.scheduler.load_state_dict(ckpt["sched"])
            self.epoch = ckpt["epoch"] + 1
            self.best_val = ckpt["best_val"]
            self.history = ckpt["history"]
            self.loaded = True
            
    def save_checkpoint(self, path): # Sometimes it's worth to overload this function
        state = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "sched": self.scheduler.state_dict(),
            "best_val": self.best_val,
            "history": self.history
        }
        torch.save(state, path)
        
    def train_batch(self, data): # Sometimes it's worth to overload this function. Must return loss, outputs, targets
        inputs, targets = data
        inputs = inputs.to(self.device, non_blocking = self.non_blocking)
        targets = targets.to(self.device, non_blocking = self.non_blocking)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss, outputs, targets
        
    def test_batch(self, data): # Sometimes it's worth to overload this function. Must return loss, outputs, targets
        with torch.inference_mode():
            inputs, targets = data
            inputs = inputs.to(self.device, non_blocking = self.non_blocking)
            targets = targets.to(self.device, non_blocking = self.non_blocking)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            return loss, outputs, targets
            
    def compute_metrics(self, outputs, targets): # To be implemented on-demand. Must return one packed, pickle-able result. Both outputs and targets are None if self.keep_outputs=False
        if self.keep_outputs:
            # outputs, targets have data
            pass
        return None
    
    def post_epoch(self, epoch): # Called at the very end of each epoch - if you want something to happen then, it's good idea to put it there
        pass
        
    def print_epoch_results(self, epoch, train_loss, train_metrics, test_loss, test_metrics):
        print(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, test_loss={test_loss:.3f}")
        
    def update_pbar(self, loop, mode, epoch, avg_loss, recent_loss):
        loop.set_postfix(mode=mode, epoch=epoch, running_loss=recent_loss, loss=avg_loss)
        
    def quantify(self, test_loss, test_metrics): # Must return single number. The lower = the better (quality)
        return test_loss
        
    def _train_step(self, epoch): # Don't change
        self.model.train()
        loop = self.train_loop_constructor()
        avg_loss, recent_loss = AverageLossCounter(), RecentLossCounter(memory = self.recent_memory)
        loss_buffer = DelayedStorage(itemize_every_k = self.log_every_k)
        output_buffer = DelayedStorage(itemize_every_k = self.log_every_k)
        target_buffer = DelayedStorage(itemize_every_k = self.log_every_k)
        all_outputs = []
        all_targets = []
        for data in loop:
            loss, outputs, targets = self.train_batch(data)
            # Since all are configured to use itemize_every_k = self.log_every_k, all of them will be logged in the same batch
            lb = loss_buffer.add(loss)
            if self.keep_outputs: 
                ob = output_buffer.add(outputs)
                tb = target_buffer.add(targets)
            if lb is not None:
                for v in lb.tolist():
                    avg_loss.add(v)
                    recent_loss.add(v)
                if self.keep_outputs:
                    all_outputs.append(ob)
                    all_targets.append(tb)
                self.update_pbar(loop, "train", epoch, avg_loss.value(), recent_loss.value())
        all_outputs = torch.cat(all_outputs) if self.keep_outputs else None
        all_targets = torch.cat(all_targets) if self.keep_outputs else None
        return avg_loss.value(), all_outputs, all_targets     
    def _test_step(self, epoch): # Don't change
        self.model.eval()
        loop = self.test_loop_constructor()
        avg_loss, recent_loss = AverageLossCounter(), RecentLossCounter(memory = self.recent_memory)
        loss_buffer = DelayedStorage(itemize_every_k = self.log_every_k)
        output_buffer = DelayedStorage(itemize_every_k = self.log_every_k)
        target_buffer = DelayedStorage(itemize_every_k = self.log_every_k)
        all_outputs = []
        all_targets = []
        for data in loop:
            loss, outputs, targets = self.test_batch(data)
            # Since all are configured to use itemize_every_k = self.log_every_k, all of them will be logged in the same batch
            lb = loss_buffer.add(loss)
            if self.keep_outputs:
                ob = output_buffer.add(outputs)
                tb = target_buffer.add(targets)
            if lb is not None:
                for v in lb.tolist():
                    avg_loss.add(v)
                    recent_loss.add(v)
                if self.keep_outputs:
                    all_outputs.append(ob)
                    all_targets.append(tb)
                self.update_pbar(loop, "test", epoch, avg_loss.value(), recent_loss.value())
        all_outputs = torch.cat(all_outputs) if self.keep_outputs else None
        all_targets = torch.cat(all_targets) if self.keep_outputs else None
        return avg_loss.value(), all_outputs, all_targets   
    def run(self, resume=True): # Don't change
        if resume and self.checkpoint_path and not self.loaded: self.load_checkpoint(self.checkpoint_path)
        for epoch in range(self.epoch, self.num_epochs):
            train_loss, train_outputs, train_targets = self._train_step(epoch)
            train_metrics = self.compute_metrics(train_outputs, train_targets)
            test_loss, test_outputs, test_targets = self._test_step(epoch)
            test_metrics = self.compute_metrics(test_outputs, test_targets)
            self.scheduler.step()
            
            self.history.append({'epoch':epoch, 'train_loss':train_loss, 'test_loss':test_loss, 'train_metrics':train_metrics, 'test_metrics':test_metrics})
            self.epoch = epoch
            val = self.quantify(test_loss, test_metrics)
            if self.best_val > val:
                self.best_val = val
                if self.best_path: self.save_checkpoint(self.best_path)
            if self.checkpoint_path: self.save_checkpoint(self.checkpoint_path)
            self.print_epoch_results(epoch, train_loss, train_metrics, test_loss, test_metrics)
            
            self.post_epoch(epoch)
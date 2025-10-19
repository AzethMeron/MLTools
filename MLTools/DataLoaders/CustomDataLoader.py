import multiprocessing as mp
import random
import math
import torch
from torch.utils.data._utils.collate import default_collate

def WorkerFunc(dataset, input_queue, output_queue):
  while True:
    idx = input_queue.get()
    if idx is None: break
    try:
      data = dataset[idx]
      output_queue.put((idx, data))
    except Exception as e:
      output_queue.put((idx, e))

class CustomDataLoader:
  def __init__(self, dataset, batch_size = 1, shuffle=False, num_workers=0, collate_fn=None, min_foresight = 4):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.num_workers = num_workers
    self.collate_fn = collate_fn if collate_fn else default_collate
    self.foresight = max(min_foresight, num_workers)

    self.indices = [idx for idx in range(len(dataset))]
    self.input_queue = None
    self.output_queue = None
    self.workers = []

    if num_workers > 0:
      self.__start_workers(num_workers)

  def __start_workers(self, num_workers):
    self.workers = []
    self.input_queue = mp.Queue()
    self.output_queue = mp.Queue()

    for _ in range(num_workers):
      p = mp.Process(target=WorkerFunc, args=(self.dataset, self.input_queue, self.output_queue))
      p.start()
      self.workers.append(p)

  def __iter__(self):
      if self.shuffle: random.shuffle(self.indices)
      self.batch_idx = 0 # BATCH INDEX, Not element index
      self.data_buffer = dict() # buffer[idx] = data
      self.requested_indices = set()
      self.received_indices = set()
      if self.workers:
        self.__schedule_foresight(self.batch_idx)
      return self

  def __get_batch_indices(self, batch_index):
      start_idx = batch_index * self.batch_size
      if start_idx >= len(self.indices): return []
      end_idx = min((batch_index+1) * self.batch_size, len(self.indices))
      return self.indices[start_idx:end_idx]

  def __schedule_foresight(self, batch_index):
      indices = []
      for i in range(self.foresight):
        indices.extend(self.__get_batch_indices(batch_index+i))
      indices = [ idx for idx in indices if idx not in self.requested_indices ]
      self.requested_indices.update(indices)
      for idx in indices: self.input_queue.put(idx)

  def __next__(self):
        if self.batch_idx >= len(self): raise StopIteration # End iteration
        batch_indices = self.__get_batch_indices(self.batch_idx)
        self.batch_idx += 1

        if self.num_workers == 0:
          batch_items = [self.dataset[idx] for idx in batch_indices]
        else:
          self.__schedule_foresight(self.batch_idx)
          batch_items = []
          for desired_index in batch_indices:
            while True:
              if desired_index in self.data_buffer:
                batch_items.append(self.data_buffer[desired_index])
                del self.data_buffer[desired_index]
                self.received_indices.add(desired_index)
                break
              elif desired_index in self.received_indices:
                e = RuntimeError(f"Repeating index: {desired_index}")
                self.__reset(e)
                raise e
              else:
                out_idx, data = self.output_queue.get()
                if isinstance(data, Exception):
                  self.__reset(data)
                  raise data
                self.data_buffer[out_idx] = data

        return self.collate_fn(batch_items)

  def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)
  def __del__(self):
    try:
      self.shutdown()
    except Exception as e:
      print(f"Exception during automatic shutdown of CustomDataLoader {type(e)}: {e}\nConsider calling shutdown manually.")

  def __reset(self, exception):
    if self.num_workers > 0:
      print(f"Resetting CustomDataLoader due to exception {type(exception)}: {exception}")
      for _ in self.workers: self.input_queue.put(None)
      for p in self.workers: p.join()
      self.input_queue.close()
      self.output_queue.close()
      self.input_queue.cancel_join_thread()
      self.output_queue.cancel_join_thread()
      self.__start_workers(self.num_workers)

  def shutdown(self):
        if self.num_workers > 0:
            for _ in self.workers:
                self.input_queue.put(None)
            for p in self.workers:
                p.join()
            self.input_queue.close()
            self.output_queue.close()
            self.workers = []
            self.num_workers = 0
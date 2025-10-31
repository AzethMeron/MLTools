
import multiprocessing as mp
import atexit

class TransformProcess(mp.Process):
    def __init__(self, transform, in_queue, out_queue):
        super().__init__(daemon=True)
        self.transform = transform
        self.in_queue = in_queue
        self.out_queue = out_queue
    def run(self):
        while True:
            item = self.in_queue.get()
            if item is None: break
            idx, pil_image = item
            x = self.transform(pil_image)
            self.out_queue.put( (idx, x) )

class ParallelizedTransformer:
    def __init__(self, transform, num_workers):
        ctx = mp.get_context("spawn")
        self.__transform = transform
        self.__in_queue = ctx.Queue()
        self.__out_queue = ctx.Queue()
        self.__workers = []
        for _ in range(num_workers):
            p = TransformProcess(self.__transform, self.__in_queue, self.__out_queue)
            p.start()
            self.__workers.append(p)
        atexit.register(self._shutdown_workers)
        
    def _shutdown_workers(self):
        for p in self.__workers:
            self.__in_queue.put(None)
        for p in self.__workers:
            p.join(timeout=1)
            
    def run(self, pil_images):
        # Send data for processing
        for idx, img in enumerate(pil_images):
            self.__in_queue.put( (idx, img) )
        # Receive processed data
        transformed_images = [ None for _ in pil_images ]
        for _ in range(len(pil_images)):
            idx, img = self.__out_queue.get()
            transformed_images[idx] = img
        # Return the images 
        return transformed_images
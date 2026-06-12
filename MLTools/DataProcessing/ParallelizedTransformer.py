
import multiprocessing as mp
import atexit


class _WorkerError:
    """Marker wrapping an exception raised inside a worker process."""
    def __init__(self, exception, formatted_traceback):
        self.exception = exception
        self.formatted_traceback = formatted_traceback


class TransformProcess(mp.Process):
    def __init__(self, transform, in_queue, out_queue):
        super().__init__(daemon=True)
        self.transform = transform
        self.in_queue = in_queue
        self.out_queue = out_queue
    def run(self):
        import traceback
        while True:
            item = self.in_queue.get()
            if item is None: break
            idx, pil_image = item
            # A raising transform must not kill the worker: report the error
            # back to the parent instead, so run() can re-raise rather than
            # block forever waiting for a result that never comes.
            try:
                x = self.transform(pil_image)
            except Exception as e:
                self.out_queue.put( (idx, _WorkerError(e, traceback.format_exc())) )
                continue
            self.out_queue.put( (idx, x) )

# Allows parallelized transformation of pillow images (or any data, actually)
# Should be used ONLY by "platform" (wrapper for model)
# Using during training will actually lower performance. Use DataLoaders.
class ParallelizedTransformer:
    def __init__(self, transform, num_workers):
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
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
        first_error = None
        for _ in range(len(pil_images)):
            idx, img = self.__out_queue.get()
            if isinstance(img, _WorkerError):
                if first_error is None:
                    first_error = img
                continue
            transformed_images[idx] = img
        # Re-raise after draining the queue so workers stay in a clean state
        if first_error is not None:
            raise RuntimeError(
                f"Transform failed in worker process:\n{first_error.formatted_traceback}"
            ) from first_error.exception
        # Return the images
        return transformed_images

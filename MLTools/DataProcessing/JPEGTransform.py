from io import BytesIO
import random
from PIL import Image

class RandomJPEGCompression:
    """
    Apply random JPEG recompression using an in-memory buffer.
    """

    def __init__(self, min_quality=25, max_quality=95, subsampling=2, seed=None, p = 1.0):
        """
        Args:
            min_quality (int): Minimum JPEG quality.
            max_quality (int): Maximum JPEG quality.
            subsampling (int): JPEG subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0) OR list of subsampling values.
            seed (int): Random seed
            p (float): Probability to apply JPEG compression, given as float 0.0 to 1.0.  
        """
        assert 1 <= min_quality <= max_quality <= 95
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.subsampling = subsampling
        self.random = random.Random(seed)
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL.Image")
        
        if self.p < 1.0 and self.random.random() > self.p: return img

        quality = self.random.randint(self.min_quality, self.max_quality)
        subsampling = self.subsampling if isinstance(self.subsampling, int) else random.choice(self.subsampling)

        buffer = BytesIO()
        img.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=subsampling,
            progressive=False,
            optimize=False,
        )
        buffer.seek(0)

        out = Image.open(buffer)
        if out.mode != img.mode:
            out = out.convert(img.mode)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(min_quality={self.min_quality}, "
            f"max_quality={self.max_quality}, "
            f"subsampling={self.subsampling}), "
            f"seed={self.random.seed}"
        )

import io

from model import TransGAN
from config import ModelParams

model = TransGAN(ModelParams())


def call(_):
    """
    Calling the GAN and returning a JPEG image. We ignore the input and return a JPEG-encoded image as bytes.
    """
    image = model()
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='JPEG')
    return bytes_io.getvalue()

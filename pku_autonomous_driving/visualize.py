import matplotlib.pyplot as plt
from .io import load_image, DataRecord
from .util import get_img_coords

def plot_data_record(plt: plt, record: DataRecord) -> None:
    plt.imshow(load_image(record.image_id)[:,:,::-1])
    plt.scatter(*get_img_coords(record.data), color='red', s=100);

# import os
# import keras
# import pickle
# import argparse
import tensorflow as tf
from pathlib import Path
# from keras.models import load_model
# from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# from models import ImageCaptionModel
# from utils import add_end_start_tokens, clean_bad_text_data, create_vocab, data_generator, get_max_length, get_train_image_captions_mapping, train_test_split, word_index_mapping

import warnings
warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

print(f'{ROOT} / weights')
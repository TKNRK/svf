import numpy as np
import logging
from functools import partial
from scipy import optimize as opt
from pathlib import PurePath, Path
from OpenGL.raw.GL._types import GL_UNSIGNED_INT, GL_UNSIGNED_SHORT

from sn.io.graph.load import Loader
from .profile import Profile


def load_centrality(cent):
  path = PurePath(Profile.dataset_dir).joinpath(Profile.name)
  c_dir = path.joinpath('centrality/v')
  c_path = c_dir.joinpath(cent + '.npy')
  CN = None
  score_min = 0
  score_max = 1
  if Path(c_path).exists():
    CN = np.load(str(c_path))
    logging.info("loaded: {0} ...".format(CN[0:5]))
    score_min = np.min(CN)
    score_max = np.max(CN)
  else:
    logging.warning("There's no centrality file")
  return CN, score_min, score_max


def load_normalized(cent):
  path = PurePath(Profile.dataset_dir).joinpath(Profile.name)
  c_dir = path.joinpath('centrality/v')
  c_path = c_dir.joinpath(cent + '.npy')
  CN = None
  if Path(c_path).exists():
    logging.warning("There's no centrality file")
    CN = np.load(str(c_path))
    val_min = np.min(CN)
    val_max = np.max(CN)
    CN = (CN - val_min) / (val_max - val_min)
  return CN



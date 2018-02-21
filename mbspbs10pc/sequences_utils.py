"""This module keeps the functions for sequences extraction from MBS files."""

import calendar
import datetime
import multiprocessing as mp
import os
from multiprocessing import Manager

import numpy as np
import pandas as pd
from tqdm import tqdm

"""Pinned Toolbox functions, without importing its unrelated trainer dependencies.

Only the Haar XML path is relocated. Upstream files remain intact for auditing.
"""
import ast
import importlib.util
from pathlib import Path
import math
import glob
import os
import re
import json
import cv2
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace
from scipy.signal import welch, butter, filtfilt
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve

REFERENCE = Path(__file__).resolve().parents[1] / 'docs/protocol_audit_sources'
COMMIT = 'b7500b848f84ad7f86e277b4612563b69f4f88f9'


def extract_class(relative, name, methods, bases=()):
    path = REFERENCE / relative
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = [ast.parse(b,mode='eval').body for b in bases]
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    source = ast.unparse(ast.Module(body=[node], type_ignores=[]))
    source = source.replace('./dataset/haarcascade_frontalface_default.xml',
                            (REFERENCE/'dataset/haarcascade_frontalface_default.xml').as_posix())
    exec(compile(source, str(path), 'exec'), globals())
    return globals()[name]


BaseLoader = extract_class('dataset/data_loader/BaseLoader.py', 'BaseLoader', {
    'preprocess', 'face_detection', 'crop_face_resize', 'chunk', 'diff_normalize_data',
    'diff_normalize_label', 'standardized_data', 'standardized_label', 'resample_ppg'})
PURELoader = extract_class('dataset/data_loader/PURELoader.py', 'PURELoader', {
    'get_raw_data', 'split_raw_data', 'read_video', 'read_wave'}, ('BaseLoader',))
UBFCrPPGLoader = extract_class('dataset/data_loader/UBFCrPPGLoader.py', 'UBFCrPPGLoader', {
    'get_raw_data', 'split_raw_data', 'read_video', 'read_wave'}, ('BaseLoader',))
PhysFormerTrainer = extract_class('neural_methods/trainer/PhysFormerTrainer.py',
                                  'PhysFormerTrainer', {'get_hr'})


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, REFERENCE/relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


post = module('evaluation/post_process.py', 'reference_post')
loss_module = module('neural_methods/loss/PhysFormerLossComputer.py', 'reference_loss')
pearson_module = SimpleNamespace(Neg_Pearson=extract_class('neural_methods/loss/PhysNetNegPearsonLoss.py',
                                'Neg_Pearson',{'__init__','forward'},('nn.Module',)))


def validation_hr(y):
    return float(PhysFormerTrainer.get_hr(None, np.asarray(y), sr=30))


def detrend_fast(y):
    """Same smoothness-prior linear system as upstream; sparse rather than dense inverse."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    d = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n))
    return y - spsolve(eye(n, format='csc') + 10000*(d.T@d).tocsc(), y)


def test_hr(y):
    # Pinned current code defaults, NOT its optional NeurIPS-2023 45–150 recommendation.
    y = detrend_fast(np.cumsum(np.asarray(y, dtype=np.float64)))
    b, a = butter(1, [0.6/30*2, 3.3/30*2], btype='bandpass')
    return float(post._calculate_fft_hr(filtfilt(b, a, y), fs=30))

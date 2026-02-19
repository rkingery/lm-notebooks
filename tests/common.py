import pickle
from pathlib import Path

import numpy as np
from torch import Tensor

THIS_DIR = Path(__file__).resolve().parent
FIXTURES_PATH = THIS_DIR / 'fixtures'
SNAPSHOTS_DIR = THIS_DIR / 'snapshots'

def canonicalize_array(arr):
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr

class NumpySnapshot:
    def __init__(self, test_name):
        self.test_name = test_name
    def assert_match(self, actual, rtol=1e-4, atol=1e-2):
        snapshot_path = SNAPSHOTS_DIR / f'{self.test_name}.npz'
        arrays_dict = actual if isinstance(actual, dict) else {'array': actual}
        arrays_dict = {k: canonicalize_array(v) for k, v in arrays_dict.items()}
        expected_arrays = dict(np.load(snapshot_path))
        missing = set(arrays_dict.keys()) - set(expected_arrays.keys())
        if missing:
            raise AssertionError(f'Keys {missing} not found in snapshot for {self.test_name}')
        extra = set(expected_arrays.keys()) - set(arrays_dict.keys())
        if extra:
            raise AssertionError(f'Snapshot contains extra keys {extra} for {self.test_name}')
        for key in arrays_dict:
            np.testing.assert_allclose(
                canonicalize_array(arrays_dict[key]), expected_arrays[key], rtol=rtol, atol=atol,
                err_msg=f'Array `{key}` does not match snapshot for {self.test_name}'
            )

class Snapshot:
    def __init__(self, test_name):
        self.test_name = test_name
    def assert_match(self, actual):
        snapshot_path = SNAPSHOTS_DIR / f'{self.test_name}.pkl'
        with open(snapshot_path, 'rb') as f:
            expected = pickle.load(f)
        if isinstance(actual, dict):
            for key in actual:
                if key not in expected:
                    raise AssertionError(f'Key `{key}` not found in snapshot for {self.test_name}')
                assert actual[key] == expected[key], f'Data for key `{key}` does not match snapshot for {self.test_name}'
        else:
            assert actual == expected, f'Data does not match snapshot for {self.test_name}'

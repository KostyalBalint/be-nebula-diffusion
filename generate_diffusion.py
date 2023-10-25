import json
import time

import numpy as np


def generate_with_diffusion():
    while True:
        yield f'data: {json.dumps((np.random.rand(10_000, 3) * 5).tolist())} \n\n'
        time.sleep(0.02)
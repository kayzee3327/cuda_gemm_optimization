import numpy as np
import os
import random

rows = 2048
cols = 4096
matname = 'matrixB'
bits = 32
t = np.float32

matpath = os.path.join(os.getcwd(), 'data', f'{matname}.bin')

randmat = 3 + 5 * np.random.randn(rows, cols)
randmat = randmat.astype(t)
# randmat = np.random.randint(1, 10, [rows, cols], dtype=np.int32)
# print(f"Verifying dtype before save: {randmat.dtype}")

with open(matpath, 'wb') as f:
    f.write(np.array([rows, cols, bits], dtype=np.uint32).tobytes())
    f.write(randmat.tobytes())

print(f'{matpath} generated.')
# print(randmat[0,0])
# print(randmat.tobytes())



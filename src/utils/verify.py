import struct
import numpy as np
import time
import pandas as pd

def read_special_binary(filepath):
    """
    Reads a binary file with a header containing dimensions
    and a data payload.
    
    Args:
        filepath (str): The path to the binary file.
        
    Returns:
        A tuple containing (rows, cols, bits, data_array).
    """
    print(f"Reading from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            # 1. Read and unpack the header (12 bytes)
            # '<' for little-endian, 'i' for 4-byte signed integer
            header_bytes = f.read(12)
            if len(header_bytes) < 12:
                raise IOError("File is too small to contain a valid header.")
                
            rows, cols, bits = struct.unpack('<iii', header_bytes)
            print(f"  - Header found: Rows={rows}, Cols={cols}, Bits={bits}")

            # 2. Determine the numpy data type from the bits
            if bits == 32:
                dtype = np.float32
            elif bits == 16:
                dtype = np.float16
            elif bits == 0:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported bit depth: {bits}")

            # 3. Read the rest of the file (the data payload)
            data_bytes = f.read()

            # 4. Create a NumPy array from the buffer
            # This is very efficient as it avoids copying data
            data_array_1d = np.frombuffer(data_bytes, dtype=dtype)

            # 5. Reshape the 1D array into the final 2D matrix
            if data_array_1d.size != rows * cols:
                raise ValueError("Mismatch between header dimensions and data size.")
            
            data_matrix = data_array_1d.reshape((rows, cols))
            
            print(f"  - Successfully read data into a {data_matrix.shape} matrix.")
            
            return rows, cols, bits, data_matrix

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # --- Read the file using our function ---
    resultA = read_special_binary("data/matrixA.bin")
    resultB = read_special_binary("data/matrixB.bin")
    resAB = read_special_binary("data/resAB.bin")
    resAB_cublas = read_special_binary("data/resAB_cuBLAS.bin")

    if resultA and resultB and resAB:
        rA, cA, bA, matrixA = resultA
        print("\n--- Function Output A ---")
        print(f"Rows: {rA}, Cols: {cA}, Bits: {bA}")
        print("Matrix data:")
        print(matrixA)

        rB, cB, bB, matrixB = resultB
        print("\n--- Function Output B ---")
        print(f"Rows: {rB}, Cols: {cB}, Bits: {bB}")
        print("Matrix data:")
        print(matrixB)

        rAB, cAB, bAB, matrixAB = resAB
        print("\n--- Function Output AB ---")
        print(f"Rows: {rAB}, Cols: {cAB}, Bits: {bAB}")
        print("Matrix data:")
        print(matrixAB)

        rAB_cublas, cAB_cublas, bAB_cublas, matrixAB_cublas = resAB_cublas
        print("\n--- Function Output AB_cublas ---")
        print(f"Rows: {rAB_cublas}, Cols: {cAB_cublas}, Bits: {bAB_cublas}")
        print("Matrix data:")
        print(matrixAB_cublas)

        t1 = time.time()
        r = matrixA @ matrixB
        t2 = time.time()
        print(f"numpy matmul uses {t2-t1} s")
        print(np.allclose(r, matrixAB))
        print(np.allclose(matrixAB_cublas, matrixAB))

# --- How to use the function ---
if __name__ == "__main__":
    main()
    

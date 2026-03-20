import numpy as np
import pytest
from wasm_shim.model import WasmMemoryBuffer

def test_init_default_size():
    buf = WasmMemoryBuffer()
    assert buf.size == 64 * 1024 * 1024

def test_write_and_read():
    buf = WasmMemoryBuffer(1024)
    buf.write(0, b"hello")
    assert buf.read(0, 5) == b"hello"

def test_write_at_offset():
    buf = WasmMemoryBuffer(1024)
    buf.write(100, b"world")
    assert buf.read(100, 5) == b"world"

def test_write_exceeds_raises():
    buf = WasmMemoryBuffer(10)
    with pytest.raises(MemoryError):
        buf.write(8, b"too long data")

def test_f32_roundtrip():
    buf = WasmMemoryBuffer(1024)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    buf.write_f32_array(0, arr)
    result = buf.read_f32_array(0, 3)
    np.testing.assert_allclose(result, arr)

def test_grow_memory():
    buf = WasmMemoryBuffer(1024)
    original_size = buf.size
    buf.grow(1)
    assert buf.size == original_size + 65536

def test_write_read_zeros():
    buf = WasmMemoryBuffer(64)
    buf.write(0, bytes(32))
    data = buf.read(0, 32)
    assert data == bytes(32)

import numpy as np
import serial
import matplotlib
matplotlib.use('TkAgg')  # set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import zoom, gaussian_filter
import threading
import time

# ---------------- Serial setup ----------------
# Adjust port/baud as needed
ser_seat = serial.Serial('COM8', baudrate=115200, timeout=0.2)

# ---------------- Seat map parameters ----------------
ROWS_SEAT, COLS_SEAT = 20, 20
NEW_ROWS_SEAT, NEW_COLS_SEAT = 60, 60
Values_seat = np.zeros((ROWS_SEAT, COLS_SEAT), dtype=np.float32)

seat_lock = threading.Lock()

# ---------------- Utilities ----------------
def read_exact(ser, n):
    """Read exactly n bytes or raise TimeoutError."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError("Serial read timeout")
        buf += chunk
    return bytes(buf)

# ---------------- Protocol I/O ----------------
def RequestPressureMapSeat():
    ser_seat.write(b"S")  # binary, not text

def ReceiveRowSeat(col_idx: int):
    # Read one column of 20 samples (each sample: HighByte then LowByte)
    # Protocol as per original code: val = 4096 - ((low << 8) + high)
    col = np.empty(ROWS_SEAT, dtype=np.float32)
    for x in range(ROWS_SEAT):
        high = read_exact(ser_seat, 1)[0]
        low  = read_exact(ser_seat, 1)[0]
        val = 4096 - ((low << 8) + high)
        if val < 0:
            val = 0
        if val > 300:
            val = 300
        col[x] = val
    # Consume one delimiter byte if present (was previously decoded; keep binary)
    _ = ser_seat.read(1)

    # Commit into shared array atomically
    with seat_lock:
        Values_seat[:, col_idx] = col

def ReceiveMapSeat():
    # Expect COLS_SEAT columns, each preceded by an 'M' marker and an index
    for _ in range(COLS_SEAT):
        # Wait for marker 'M'
        marker = read_exact(ser_seat, 1)
        if marker != b"M":
            # If garbage, keep consuming until 'M' is found
            # Guard against infinite loops by limited scans
            attempts = 0
            while marker != b"M" and attempts < 64:
                marker = ser_seat.read(1) or b""
                attempts += 1
            if marker != b"M":
                raise ValueError("Seat stream desync: 'M' marker not found")

        # Original code discarded one byte after 'M' before the index; preserve that behavior
        _ = read_exact(ser_seat, 1)          # header/padding byte
        row_index = read_exact(ser_seat, 1)[0]  # actually the column index per original code
        if not (0 <= row_index < COLS_SEAT):
            # Clamp or skip invalid index to avoid IndexError
            continue
        ReceiveRowSeat(row_index)

# ---------------- Thread worker ----------------
def seat_thread():
    while True:
        try:
            RequestPressureMapSeat()
            ReceiveMapSeat()
            with seat_lock:
                np.clip(Values_seat, 0, None, out=Values_seat)
        except (TimeoutError, ValueError) as e:
            # Brief backoff on sync issues/timeouts, then retry
            time.sleep(0.02)
            # Optionally flush input buffer to resync
            try:
                ser_seat.reset_input_buffer()
            except Exception:
                pass
        except Exception:
            # Do not crash the thread on unexpected errors; brief backoff
            time.sleep(0.05)

        time.sleep(0.01)

# ---------------- Visualization ----------------
fig, ax_seat = plt.subplots(1, figsize=(8, 4.5))

heatmap_seat = ax_seat.imshow(
    np.zeros((NEW_ROWS_SEAT, NEW_COLS_SEAT), dtype=np.float32),
    cmap='viridis',
    interpolation='nearest',
    vmin=0, vmax=300
)
ax_seat.set_title("Seat Pressure Map")
plt.colorbar(heatmap_seat, ax=ax_seat)

def update_seat(_frame):
    with seat_lock:
        upscaled = zoom(Values_seat, (NEW_ROWS_SEAT / ROWS_SEAT, NEW_COLS_SEAT / COLS_SEAT), order=1)
    blurred = gaussian_filter(upscaled, sigma=2)
    heatmap_seat.set_data(blurred)
    return (heatmap_seat,)

# ---------------- Run ----------------
threading.Thread(target=seat_thread, daemon=True).start()

ani_seat = animation.FuncAnimation(
    fig,
    update_seat,
    blit=False,
    interval=100,
    cache_frame_data=False  # suppress unbounded cache warning
)

plt.tight_layout()
plt.show()

"""
Chan3ProcAll_cssl.py

Combined conversion of Chan3ProcAllSiteA.m and PBandChan3Procall_temp.m
Channel 3 Not-switched data processing at ~1 s Integration Time.

Input filename format: UHFLHCP_CH3_YYYYMMDDHHmmss_ch0 / _ch1
  e.g., UHFLHCP_CH3_20251220060042_ch0, UHFLHCP_CH3_20251220060042_ch1

Features:
  - Filename matching for UHFLHCP_CH3_*_ch0/ch1 format
  - Accumulates all waveform segments with coherent averaging
  - Windowing (Hanning, Blackman-Harris, etc.) and zero-padding
  - SNR and peak detection on cross-correlation
  - Multithreaded FFTs via scipy.fft
  - Input validation and structured logging
  - YAML config file support
  - Summary CSV output with timestamps
  - Batch time-series plotting

Run with --help or see README for full usage.
"""

import os
import re
import sys
import csv
import gc
import time
import glob
import logging
import argparse
import datetime
import numpy as np

# Use scipy.fft for multithreaded FFTs (fallback to numpy)
try:
    from scipy.fft import fft, ifft, fftshift, set_workers
    HAS_SCIPY_FFT = True
except ImportError:
    from numpy.fft import fft, ifft, fftshift
    HAS_SCIPY_FFT = False

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Optional YAML config
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ============================================================
# Logging Setup
# ============================================================
def setup_logging(log_file=None, verbose=False):
    """Configure logging to console and optionally to a file."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a'))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

# ============================================================
# Timestamp Extraction
# ============================================================
def extract_timestamp(filename):
    """
    Extract a datetime from a filename containing a numeric timestamp.

    Supports formats:
        YYYYMMDDHHmmss  (14 digits) -> e.g., UHF_20251223021044_ch0.dat
        YYYYMMDDHHmm    (12 digits)
        YYYYMMDD        (8 digits)

    Returns datetime object or None if no timestamp found.
    """
    # Look for 14-digit timestamp first, then 12, then 8
    patterns = [
        (r'(\d{14})', '%Y%m%d%H%M%S'),
        (r'(\d{12})', '%Y%m%d%H%M'),
        (r'(\d{8})',  '%Y%m%d'),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return datetime.datetime.strptime(match.group(1), fmt)
            except ValueError:
                continue
    return None


# ============================================================
# Window Functions
# ============================================================
WINDOW_FUNCTIONS = {
    'none':            lambda n: np.ones(n),
    'hanning':         np.hanning,
    'hamming':         np.hamming,
    'blackman':        np.blackman,
    'blackmanharris':  lambda n: _blackman_harris(n),
    'kaiser4':         lambda n: np.kaiser(n, 4.0 * np.pi),
}

def _blackman_harris(n):
    """4-term Blackman-Harris window."""
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    k = np.arange(n)
    return a0 - a1 * np.cos(2*np.pi*k/(n-1)) + a2 * np.cos(4*np.pi*k/(n-1)) - a3 * np.cos(6*np.pi*k/(n-1))

def get_window(name, n):
    """Get a window function by name."""
    name = name.lower().replace('-', '').replace('_', '')
    if name not in WINDOW_FUNCTIONS:
        raise ValueError(f"Unknown window '{name}'. Options: {list(WINDOW_FUNCTIONS.keys())}")
    return WINDOW_FUNCTIONS[name](n)


# ============================================================
# SNR / Peak Detection
# ============================================================
def _extract_at_fractional_lag(correlation, lag, target_lag_value):
    """
    Extract the complex correlation value at an exact fractional lag
    using sinc interpolation (mathematically exact for bandlimited signals).

    Parameters
    ----------
    correlation : ndarray - Complex correlation waveform.
    lag : ndarray - Integer lag indices.
    target_lag_value : float - Exact fractional lag to extract at.

    Returns
    -------
    complex - Interpolated complex value at the target lag.
    """
    # Use a small window of ±8 samples around the target for sinc interpolation
    center_idx = np.argmin(np.abs(lag - target_lag_value))
    half_win = 8
    lo = max(0, center_idx - half_win)
    hi = min(len(correlation), center_idx + half_win + 1)

    local_lag = lag[lo:hi].astype(np.float64)
    local_corr = correlation[lo:hi]

    # Sinc interpolation: sum of samples weighted by sinc(target - lag_i)
    offsets = target_lag_value - local_lag
    weights = np.sinc(offsets)  # np.sinc(x) = sin(pi*x) / (pi*x)
    return np.sum(local_corr * weights)


def detect_peak_and_snr(correlation, lag, noise_fraction=0.2, peak_lag_value=None):
    """
    Extract correlation amplitude and phase at a specified lag, and estimate SNR.

    Parameters
    ----------
    correlation : ndarray - Complex correlation waveform.
    lag : ndarray - Corresponding lag indices.
    noise_fraction : float - Fraction of outer samples used for noise floor estimate.
    peak_lag_value : float or None - Exact lag value to extract amplitude/phase at.
                     If None, auto-detects from maximum of |correlation|.

    Returns
    -------
    dict with keys:
        'peak_lag'       : float - The lag value used
        'peak_amplitude' : float - Absolute amplitude at that lag
        'peak_phase_deg' : float - Phase at that lag in degrees
        'snr_db'         : float - SNR in dB (peak power / mean noise power)
        'noise_floor'    : float - Estimated mean noise amplitude
    """
    mag = np.abs(correlation)

    # --- Noise floor estimate (from outer edges) ---
    n = len(mag)
    n_noise = max(int(n * noise_fraction), 10)
    noise_samples = np.concatenate([mag[:n_noise], mag[-n_noise:]])
    noise_floor = np.mean(noise_samples)

    if peak_lag_value is not None:
        # Fixed lag: extract at exactly this value using sinc interpolation
        peak_lag = float(peak_lag_value)
        complex_val = _extract_at_fractional_lag(correlation, lag, peak_lag)
        peak_amp = float(np.abs(complex_val))
        peak_phase_deg = float(np.angle(complex_val, deg=True))
    else:
        # Auto-detect: find maximum magnitude
        idx = np.argmax(mag)
        peak_lag = float(lag[idx])
        peak_amp = float(mag[idx])
        peak_phase_deg = float(np.angle(correlation[idx], deg=True))

    if noise_floor > 0:
        snr_db = 10.0 * np.log10((peak_amp ** 2) / (noise_floor ** 2))
    else:
        snr_db = np.inf

    return {
        'peak_lag': peak_lag,
        'peak_amplitude': peak_amp,
        'peak_phase_deg': peak_phase_deg,
        'snr_db': snr_db,
        'noise_floor': float(noise_floor),
    }


# ============================================================
# Processing Function
# ============================================================
def PBandChan3Procall(fname1, fname2, Ti, sample_rate, savefile=None, num=2,
                      llm=None, uum=None, fft_workers=1,
                      window='none', zero_pad_factor=1, target_lag=None):
    """
    Process two-channel IQ data files and compute correlation waveforms.

    Parameters
    ----------
    fname1 : str - Path to channel 0 data file (int16 interleaved I/Q).
    fname2 : str - Path to channel 1 data file (int16 interleaved I/Q).
    Ti : float - Coherent integration time (seconds).
    sample_rate : float - Data sampling rate (Hz).
    savefile : str or None - Path to save output .txt file.
    num : int - Number of waveform segments to process.
    llm : int or None - Lower lag index to save (inclusive).
    uum : int or None - Upper lag index to save (inclusive).
    fft_workers : int - Number of threads for scipy.fft (default: 1).
    window : str - Window function name ('none', 'hanning', 'blackmanharris', etc.).
    zero_pad_factor : int - Zero-pad FFT length to npts * factor (1 = no padding).
    target_lag : float or None - Fixed lag value to extract amplitude/phase at.
                 Supports fractional values (e.g., 0.25). If None, auto-detects peak.

    Returns
    -------
    dict with processing results, peak detection, and metadata.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing pair:")
    logger.info(f"  ch0: {fname1}")
    logger.info(f"  ch1: {fname2}")

    samples_per_segment = int(sample_rate * Ti)
    expected_read = samples_per_segment * 2  # int16 values (I and Q interleaved)

    # --- Validate input files ---
    for fpath in [fname1, fname2]:
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

    fsize1 = os.path.getsize(fname1)
    fsize2 = os.path.getsize(fname2)
    bytes_needed = expected_read * 2 * num  # 2 bytes per int16, times num segments
    for fpath, fsize in [(fname1, fsize1), (fname2, fsize2)]:
        if fsize < bytes_needed:
            logger.warning(
                f"File {os.path.basename(fpath)} has {fsize} bytes, "
                f"need {bytes_needed} for {num} segment(s). "
                f"Will process as many complete segments as available."
            )

    # --- FFT context ---
    if HAS_SCIPY_FFT and fft_workers != 1:
        fft_ctx = set_workers(fft_workers)
        logger.debug(f"Using scipy.fft with {fft_workers} worker(s)")
    else:
        from contextlib import nullcontext
        fft_ctx = nullcontext()
        if not HAS_SCIPY_FFT:
            logger.debug("scipy.fft not available, using numpy.fft")

    # --- Precompute window ---
    win = get_window(window, samples_per_segment)
    win_norm = np.sum(win ** 2)  # Window power normalization
    if window != 'none':
        logger.debug(f"Applying '{window}' window (norm={win_norm:.2f})")

    # --- FFT length with zero-padding ---
    nfft = samples_per_segment * zero_pad_factor
    if zero_pad_factor > 1:
        logger.debug(f"Zero-padding: {samples_per_segment} -> {nfft} points ({zero_pad_factor}x)")

    # --- Memory estimate ---
    # Each complex128 array of nfft points = nfft * 16 bytes
    # Per segment: 2 input + 2 spectra + 3 correlations = ~7 arrays
    mem_per_seg_gb = (7 * nfft * 16) / (1024 ** 3)
    total_mem_gb = mem_per_seg_gb * num
    logger.info(f"  FFT size: {nfft:,} points | "
                f"Est. memory: {mem_per_seg_gb:.2f} GB/seg, {total_mem_gb:.2f} GB total")
    if mem_per_seg_gb > 2.0:
        logger.warning(
            f"  High memory usage estimated ({mem_per_seg_gb:.1f} GB per segment). "
            f"Consider reducing --zero_pad or --Ti if processing is killed."
        )

    # --- Storage ---
    cal_all = []
    cald_all = []
    calr_all = []
    lag = None
    segments_processed = 0

    with fft_ctx, open(fname1, 'rb') as fid1, open(fname2, 'rb') as fid2:
        for seg in range(num):
            raw1 = np.fromfile(fid1, dtype=np.int16, count=expected_read)
            raw2 = np.fromfile(fid2, dtype=np.int16, count=expected_read)

            # --- Validate read ---
            if len(raw1) < expected_read or len(raw2) < expected_read:
                logger.warning(
                    f"Segment {seg + 1}/{num}: incomplete read "
                    f"(ch0: {len(raw1)}/{expected_read}, ch1: {len(raw2)}/{expected_read}). "
                    f"Stopping early."
                )
                break

            # Convert to complex
            data1c = raw1[0::2].astype(np.float64) + 1j * raw1[1::2].astype(np.float64)
            data2c = raw2[0::2].astype(np.float64) + 1j * raw2[1::2].astype(np.float64)

            if len(data1c) != len(data2c):
                raise ValueError(
                    f"Segment {seg + 1}: channel length mismatch "
                    f"(ch0: {len(data1c)}, ch1: {len(data2c)})"
                )

            # --- Apply window ---
            data1c = data1c * win
            data2c = data2c * win

            # --- FFT (with optional zero-padding) ---
            spec1 = fft(data1c, n=nfft)
            spec2 = fft(data2c, n=nfft)

            # Cross-correlation
            cal = fftshift(ifft(spec1 * np.conj(spec2))) / nfft

            # Auto-correlation ch0 (direct)
            cald = fftshift(ifft(np.abs(spec1) ** 2)) / nfft

            # Auto-correlation ch1 (reference)
            calr = fftshift(ifft(np.abs(spec2) ** 2)) / nfft

            # Lag vector
            if lag is None:
                lag = np.arange(-nfft // 2, (nfft - 1) // 2 + 1)

            cal_all.append(cal)
            cald_all.append(cald)
            calr_all.append(calr)
            segments_processed += 1
            logger.debug(f"  Segment {seg + 1}/{num} done ({nfft} FFT points)")

            # --- Free intermediate arrays ---
            del raw1, raw2, data1c, data2c, spec1, spec2, cal, cald, calr

    if segments_processed == 0:
        raise RuntimeError(f"No complete segments could be read from files.")

    logger.info(f"  Processed {segments_processed}/{num} segment(s)")

    # --- Stack arrays ---
    cal_all = np.array(cal_all)
    cald_all = np.array(cald_all)
    calr_all = np.array(calr_all)

    # --- Averaging ---
    cal_mean = np.mean(cal_all, axis=0)           # Coherent (complex) average
    cald_mean = np.mean(cald_all, axis=0)
    calr_mean = np.mean(calr_all, axis=0)

    # --- Trim to lag subset ---
    if llm is not None and uum is not None:
        if llm < 0 or uum >= cal_all.shape[1] or llm > uum:
            logger.warning(
                f"  llm={llm}, uum={uum} out of range [0, {cal_all.shape[1] - 1}]. Saving full arrays."
            )
        else:
            cal_all = cal_all[:, llm:uum + 1]
            cald_all = cald_all[:, llm:uum + 1]
            calr_all = calr_all[:, llm:uum + 1]
            cal_mean = cal_mean[llm:uum + 1]
            cald_mean = cald_mean[llm:uum + 1]
            calr_mean = calr_mean[llm:uum + 1]
            lag = lag[llm:uum + 1]
            logger.debug(f"  Trimmed to lag [{llm}:{uum}] ({len(lag)} points)")

    # --- Peak / SNR detection ---
    peak_info = detect_peak_and_snr(cal_mean, lag, peak_lag_value=target_lag)
    logger.info(f"  Peak lag={peak_info['peak_lag']:.4f}, "
                f"amplitude={peak_info['peak_amplitude']:.4e}, "
                f"phase={peak_info['peak_phase_deg']:.1f}°, "
                f"SNR={peak_info['snr_db']:.1f} dB")

    # --- Extract timestamp ---
    timestamp = extract_timestamp(os.path.basename(fname1))

    # --- Build results ---
    results = {
        'cal': cal_all, 'cald': cald_all, 'calr': calr_all,
        'cal_mean': cal_mean,
        'cald_mean': cald_mean, 'calr_mean': calr_mean,
        'lag': lag,
        'segments_processed': segments_processed,
        'peak_info': peak_info,
        'timestamp': timestamp,
        'window': window,
        'zero_pad_factor': zero_pad_factor,
    }

    # --- Save ---
    if savefile is not None:
        # Save as text file with header metadata
        # Columns: lag, |cal_mean|, |cald_mean|, |calr_mean|, |cal_mag_mean|
        # (absolute values since complex numbers can't go in plain text columns)
        header_lines = [
            f"# Chan3ProcAll_cssl Output",
            f"# fname1: {fname1}",
            f"# fname2: {fname2}",
            f"# Ti: {Ti}",
            f"# sample_rate: {sample_rate}",
            f"# num_segments: {segments_processed}",
            f"# window: {window}",
            f"# zero_pad_factor: {zero_pad_factor}",
            f"# peak_lag: {peak_info['peak_lag']:.4f}",
            f"# peak_amplitude: {peak_info['peak_amplitude']:.6e}",
            f"# peak_phase_deg: {peak_info['peak_phase_deg']:.2f}",
            f"# snr_db: {peak_info['snr_db']:.2f}",
            f"# noise_floor: {peak_info['noise_floor']:.6e}",
            f"#",
            f"# Columns: lag  |cross_corr_mean|  |direct_autocorr_mean|  |ref_autocorr_mean|",
        ]
        header = '\n'.join(header_lines)

        out_data = np.column_stack([
            lag.real,
            np.abs(cal_mean),
            np.abs(cald_mean),
            np.abs(calr_mean),
        ])

        np.savetxt(savefile, out_data, header=header, fmt='%.6e',
                   delimiter='\t', comments='')
        logger.info(f"  Saved: {savefile}")

    return results


# ============================================================
# File Pair Discovery (modified)
# ============================================================
def find_file_pairs(infolder, prefix, suffix_ch0, suffix_ch1, start_limit=None, end_limit=None):
    """
    Find matched pairs of channel 0 and channel 1 files in a directory, in a specific time range.

    Parameters
    ----------
    infolder : str - Directory to search.
    prefix : str - Filename prefix to filter. Use '' for no filter.
    suffix_ch0 : str - Suffix identifying channel 0 files.
    suffix_ch1 : str - Suffix identifying channel 1 files.

    Returns
    -------
    list of tuples: [(ch0_path, ch1_path, base_name), ...]
    """
    logger = logging.getLogger(__name__)
    pattern = os.path.join(infolder, f"{prefix}*{suffix_ch0}")
    ch0_files = sorted(glob.glob(pattern))

    pairs = []
    for ch0_path in ch0_files:
        fname = os.path.basename(ch0_path)

        # Modified block for timestamp filtering
        if start_limit or end_limit:
            file_time = extract_timestamp(fname)
            if file_time:
                if start_limit and file_time < start_limit:
                    continue # File is before start time
                if end_limit and file_time > end_limit:
                    continue # File is after end time
            else:
                logger.warning(f"Could not extract timestamp from {fname}, skipping.")
                continue

        base_name = fname[:-len(suffix_ch0)]
        ch1_path = os.path.join(infolder, base_name + suffix_ch1)
        if os.path.exists(ch1_path):
            pairs.append((ch0_path, ch1_path, base_name))
        else:
            logger.warning(f"No matching ch1 file for {fname}, skipping.")

    logger.info(f"Found {len(pairs)} file pair(s) matching: {prefix}*{suffix_ch0} / {suffix_ch1}")
    return pairs


# ============================================================
# Summary CSV
# ============================================================
def write_summary_csv(csv_path, summary_rows):
    """
    Write a summary CSV with one row per processed file pair.

    Parameters
    ----------
    csv_path : str - Output CSV file path.
    summary_rows : list of dicts - Each dict has keys from processing results.
    """
    if not summary_rows:
        return

    fieldnames = ['timestamp', 'base_name', 'peak_lag', 'peak_amplitude', 'peak_phase_deg',
                  'snr_db', 'noise_floor', 'segments_processed',
                  'window', 'zero_pad_factor', 'savefile']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    logging.getLogger(__name__).info(f"Summary CSV written: {csv_path}")


# ============================================================
# Time-Series Plot
# ============================================================
def plot_time_series(summary_rows, savefolder, run_timestamp=''):
    """
    Plot peak lag, amplitude, phase, and SNR vs. time from summary data.
    Saves figure to savefolder with run timestamp in filename.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Filter rows with valid timestamps
    rows = [r for r in summary_rows if r['timestamp'] is not None]
    if len(rows) < 2:
        logging.getLogger(__name__).warning("Not enough timestamped files for time-series plot.")
        return

    times = [r['timestamp'] for r in rows]
    peak_lags = [r['peak_lag'] for r in rows]
    peak_amps = [r['peak_amplitude'] for r in rows]
    peak_phases = [r['peak_phase_deg'] for r in rows]
    snrs = [r['snr_db'] for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(times, peak_lags, 'o-', markersize=4, linewidth=1)
    axes[0].set_ylabel('Peak Lag (samples)')
    axes[0].set_title('Cross-Correlation Time Series')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, peak_amps, 'o-', markersize=4, linewidth=1, color='tab:orange')
    axes[1].set_ylabel('Peak Amplitude')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, peak_phases, 'o-', markersize=4, linewidth=1, color='tab:purple')
    axes[2].set_ylabel('Peak Phase (°)')
    axes[2].set_ylim([-180, 180])
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, snrs, 'o-', markersize=4, linewidth=1, color='tab:green')
    axes[3].set_ylabel('SNR (dB)')
    axes[3].set_xlabel('Time (UTC)')
    axes[3].grid(True, alpha=0.3)

    # Format x-axis
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    fig.autofmt_xdate()
    plt.tight_layout()

    plot_path = os.path.join(savefolder, f'time_series_summary_{run_timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.getLogger(__name__).info(f"Time-series plot saved: {plot_path}")


# ============================================================
# Per-File Correlation Plot
# ============================================================
def save_correlation_plot(results, base_name, plot_dir):
    """
    Save a correlation waveform plot for a single file pair.

    Parameters
    ----------
    results : dict - Output from PBandChan3Procall.
    base_name : str - Base filename (used for title and output filename).
    plot_dir : str - Directory to save the plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lag = results['lag']
    peak = results['peak_info']
    n_seg = results['segments_processed']
    timestamp = results['timestamp']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Determine zoom range around peak ---
    peak_lag_val = peak['peak_lag']
    zoom_half = 100  # ±100 samples around peak
    zoom_mask = (lag >= peak_lag_val - zoom_half) & (lag <= peak_lag_val + zoom_half)
    if np.any(zoom_mask):
        lag_zoom = lag[zoom_mask]
        cal_mean_zoom = results['cal_mean'][zoom_mask]
        cald_mean_zoom = results['cald_mean'][zoom_mask]
    else:
        # Fallback to full range if peak_lag is outside lag range
        lag_zoom = lag
        cal_mean_zoom = results['cal_mean']
        cald_mean_zoom = results['cald_mean']

    # --- Panel 1: Mean cross-correlation and direct auto-correlation ---
    axes[0].plot(lag_zoom, np.abs(cal_mean_zoom), linewidth=2, label='Cross Correlation')
    axes[0].plot(lag_zoom, np.abs(cald_mean_zoom), linewidth=2, label='Direct Correlation')
    axes[0].axvline(peak_lag_val, color='r', linestyle='--', alpha=0.5,
                    label=f"Peak @ lag {peak_lag_val:.2f}")
    axes[0].set_title(f"Mean Correlation (SNR={peak['snr_db']:.1f} dB)")
    axes[0].set_ylabel('Power (uncalibrated)')
    axes[0].set_xlabel('Lag (samples)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Per-segment cross-correlations ---
    for seg in range(results['cal'].shape[0]):
        label = f'Seg {seg + 1}' if n_seg <= 8 else None
        cal_seg_zoom = results['cal'][seg, :][zoom_mask] if np.any(zoom_mask) else results['cal'][seg, :]
        axes[1].plot(lag_zoom, np.abs(cal_seg_zoom),
                     linewidth=1, alpha=0.6, label=label)
    axes[1].axvline(peak_lag_val, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Cross Correlation ({n_seg} segments)')
    axes[1].set_ylabel('Power (uncalibrated)')
    axes[1].set_xlabel('Lag (samples)')
    if n_seg <= 8:
        axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # --- Title ---
    title = base_name
    if timestamp:
        title += f"  |  {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Save ---
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{base_name}_correlation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')  # Extra safety: close any leaked figures

    logging.getLogger(__name__).debug(f"  Correlation plot saved: {plot_path}")


# ============================================================
# YAML Config
# ============================================================
def load_config(config_path):
    """
    Load processing parameters from a YAML config file.

    Returns a dict that can be merged with argparse defaults.
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.getLogger(__name__).info(f"Loaded config from: {config_path}")
    return config


def generate_example_config(path):
    """Write an example YAML config file."""
    example = """# Chan3ProcAll_cssl Configuration File
# ================================

# Input/output directories
infolder: /data/raw/
savefolder: /data/processed/

# Filename pattern (for UHFLHCP_CH3_YYYYMMDDHHmmss_ch0/_ch1 format)
prefix: UHFLHCP_CH3_
suffix0: _ch0.dat
suffix1: _ch1.dat

# Processing parameters
Ti: 0.9
num: 2
sample_rate: 20e6
# llm: 8997500       # Uncomment to trim lag range
# uum: 9002500

# Signal processing
window: hanning       # Options: none, hanning, hamming, blackman, blackmanharris, kaiser4
zero_pad_factor: 1    # 1 = no padding, 2 = 2x zero-pad, etc.
# peak_lag: 0.575     # Fixed fractional lag to extract amplitude/phase at

# Performance
fft_workers: -1       # -1 = all CPU cores

# Logging
verbose: false
# log: processing.log  # Uncomment to log to file

# Output
summary_csv: true     # Write summary CSV
time_series_plot: true # Generate time-series plot
save_plots: false      # Save correlation plot per file pair (to <savefolder>/plots/)
rerun: false           # Reprocess files even if output already exists
"""
    with open(path, 'w') as f:
        f.write(example)
    print(f"Example config written to: {path}")


# ============================================================
# Main - Modified
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="P-Band two-channel IQ correlation processor (CSSL format).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Basic batch processing:
  python3 Chan3ProcAll_cssl.py -i /data/raw/ -o /data/processed/

  # With windowing and zero-padding:
  python3 Chan3ProcAll_cssl.py -i /data/raw/ -o /data/processed/ --window hanning --zero_pad 2

  # Using a config file:
  python3 Chan3ProcAll_cssl.py --config my_config.yaml

  # Generate example config file:
  python3 Chan3ProcAll_cssl.py --gen_config my_config.yaml

  # Debug mode (single file pair with plot):
  python3 Chan3ProcAll_cssl.py --debug --file0 /data/UHFLHCP_CH3_20251220060042_ch0.dat --file1 /data/UHFLHCP_CH3_20251220060042_ch1.dat

  # Fast processing with verbose logging:
  python3 Chan3ProcAll_cssl.py -i /data/raw/ -o /data/processed/ --fft_workers 8 -v --log run.log
        """
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Load parameters from YAML config file')
    parser.add_argument('--gen_config', type=str, default=None, metavar='PATH',
                        help='Generate an example YAML config file and exit')

    # Mode selection
    parser.add_argument('--debug', action='store_true',
                        help='Single-file debug mode with plotting')

    # Folder arguments
    parser.add_argument('-i', '--infolder', type=str, default=None,
                        help='Input directory containing .dat files')
    parser.add_argument('-o', '--savefolder', type=str, default=None,
                        help='Output directory for .npz files')

    # Filename pattern
    parser.add_argument('--prefix', type=str, default='UHFLHCP_CH3_',
                        help='Filename prefix filter (default: UHFLHCP_CH3_)')
    parser.add_argument('--suffix0', type=str, default='_ch0.dat',
                        help='Suffix for channel 0 files (default: _ch0.dat)')
    parser.add_argument('--suffix1', type=str, default='_ch1.dat',
                        help='Suffix for channel 1 files (default: _ch1.dat)')

    # Modified: Date Range Filters
    parser.add_argument('--start_time', type=str, default=None,
                        help='Start timestamp filter (YYYYMMDDHHmmss)')
    parser.add_argument('--end_time', type=str, default=None,
                        help='End timestamp filter (YYYYMMDDHHmmss)')

    # Debug mode
    parser.add_argument('--file0', type=str, default=None, help='Channel 0 file (debug mode)')
    parser.add_argument('--file1', type=str, default=None, help='Channel 1 file (debug mode)')

    # Processing parameters
    parser.add_argument('--Ti', type=float, default=0.9, help='Integration time in seconds (default: 0.9)')
    parser.add_argument('--num', type=int, default=2, help='Number of segments (default: 2)')
    parser.add_argument('--llm', type=int, default=None, help='Lower lag index to save')
    parser.add_argument('--uum', type=int, default=None, help='Upper lag index to save')
    parser.add_argument('--sample_rate', type=float, default=20e6, help='Sample rate in Hz (default: 20e6)')

    # Signal processing
    parser.add_argument('--window', type=str, default='none',
                        help='Window function: none, hanning, hamming, blackman, blackmanharris, kaiser4')
    parser.add_argument('--zero_pad', type=int, default=1,
                        help='Zero-pad factor (1=none, 2=2x, etc.)')
    parser.add_argument('--peak_lag', type=float, default=None,
                        help='Fixed lag value to extract amplitude/phase at (supports fractional, e.g., 0.25)')

    # Performance
    parser.add_argument('--fft_workers', type=int, default=-1,
                        help='FFT threads (-1 = all cores)')

    # Logging
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--log', type=str, default=None, help='Log to file')

    # Output options
    parser.add_argument('--no_csv', action='store_true', help='Skip summary CSV generation')
    parser.add_argument('--no_plot', action='store_true', help='Skip time-series plot generation')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save correlation plot for each file pair (saved to <savefolder>/plots/)')
    parser.add_argument('--rerun', action='store_true',
                        help='Reprocess files even if output already exists (overwrites previous results)')

    args = parser.parse_args()

    # --- Generate example config and exit ---
    if args.gen_config:
        generate_example_config(args.gen_config)
        sys.exit(0)

    # --- Load YAML config and merge (CLI args override config) ---
    if args.config:
        config = load_config(args.config)
        # Map config keys to argparse names
        key_map = {'zero_pad_factor': 'zero_pad', 'summary_csv': None, 'time_series_plot': None}
        for key, val in config.items():
            arg_key = key_map.get(key, key)
            if arg_key is None:
                continue  # handled separately
            # Only apply config value if CLI didn't override (arg is still at default)
            if hasattr(args, arg_key):
                cli_default = parser.get_default(arg_key)
                if getattr(args, arg_key) == cli_default:
                    setattr(args, arg_key, val)
        # Handle special config-only flags
        if config.get('summary_csv', True) is False:
            args.no_csv = True
        if config.get('time_series_plot', True) is False:
            args.no_plot = True
        if config.get('save_plots', False) is True:
            args.save_plots = True
        if config.get('rerun', False) is True:
            args.rerun = True

    # --- Ensure numeric types (YAML may pass scientific notation as strings) ---
    args.Ti = float(args.Ti)
    args.sample_rate = float(args.sample_rate)
    args.num = int(args.num)
    args.zero_pad = int(args.zero_pad)
    args.fft_workers = int(args.fft_workers)
    if args.llm is not None:
        args.llm = int(args.llm)
    if args.uum is not None:
        args.uum = int(args.uum)
    if args.peak_lag is not None:
        args.peak_lag = float(args.peak_lag)

    # --- Setup logging ---
    setup_logging(log_file=args.log, verbose=args.verbose)
    logger = logging.getLogger(__name__)

    if args.debug:
        # ============================================================
        # DEBUG MODE
        # ============================================================
        if args.file0 is None or args.file1 is None:
            parser.error("--debug mode requires --file0 and --file1")

        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 18, 'font.weight': 'normal'})

        results = PBandChan3Procall(
            args.file0, args.file1, args.Ti, args.sample_rate,
            savefile=None, num=args.num, llm=args.llm, uum=args.uum,
            fft_workers=args.fft_workers, window=args.window,
            zero_pad_factor=args.zero_pad, target_lag=args.peak_lag
        )

        lag = results['lag']
        peak = results['peak_info']

        # --- Zoom range around peak ---
        peak_lag_val = peak['peak_lag']
        zoom_half = 100
        zoom_mask = (lag >= peak_lag_val - zoom_half) & (lag <= peak_lag_val + zoom_half)
        if np.any(zoom_mask):
            lag_zoom = lag[zoom_mask]
        else:
            lag_zoom = lag
            zoom_mask = np.ones(len(lag), dtype=bool)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: mean correlations
        axes[0].plot(lag_zoom, np.abs(results['cal_mean'][zoom_mask]), linewidth=2, label='Cross Correlation')
        axes[0].plot(lag_zoom, np.abs(results['cald_mean'][zoom_mask]), linewidth=2, label='Direct Correlation')
        axes[0].axvline(peak_lag_val, color='r', linestyle='--', alpha=0.5,
                        label=f"Peak @ lag {peak_lag_val:.2f}")
        axes[0].set_title(f"Mean Correlation (SNR={peak['snr_db']:.1f} dB)")
        axes[0].set_ylabel('Power (uncalibrated)')
        axes[0].set_xlabel('Lag (samples)')
        axes[0].legend()

        # Right: per-segment cross-correlations
        for seg in range(results['cal'].shape[0]):
            label = f'Seg {seg + 1}' if results['cal'].shape[0] <= 8 else None
            axes[1].plot(lag_zoom, np.abs(results['cal'][seg, :][zoom_mask]),
                         linewidth=1, alpha=0.6, label=label)
        axes[1].axvline(peak_lag_val, color='r', linestyle='--', alpha=0.5)
        axes[1].set_title('Cross Correlation (per segment)')
        axes[1].set_ylabel('Power (uncalibrated)')
        axes[1].set_xlabel('Lag (samples)')
        if results['cal'].shape[0] <= 8:
            axes[1].legend()

        plt.tight_layout()
        plt.show()

    else:
        # ============================================================
        # BATCH MODE
        # ============================================================
        if args.infolder is None or args.savefolder is None:
            parser.error("Batch mode requires -i/--infolder and -o/--savefolder")

        os.makedirs(args.savefolder, exist_ok=True)
        tic = time.time()

        # New: Parse start and end timestamps
        start_limit = datetime.datetime.strptime(str(args.start_time), '%Y%m%d%H%M%S') if args.start_time else None
        end_limit = datetime.datetime.strptime(str(args.end_time), '%Y%m%d%H%M%S') if args.end_time else None

        pairs = find_file_pairs(args.infolder, args.prefix, args.suffix0, args.suffix1, start_limit=start_limit, end_limit=end_limit)

        if len(pairs) == 0:
            logger.error("No file pairs found. Check --infolder, --prefix, --suffix0, --suffix1, or date limits.")
            sys.exit(1)

        # Progress bar or plain iterator
        iterator = tqdm(pairs, desc="Processing", unit="pair") if HAS_TQDM else pairs

        success_count = 0
        error_count = 0
        summary_rows = []

        for ch0_path, ch1_path, base_name in iterator:
            savefile = os.path.join(args.savefolder, base_name + '_Tip9scorrr.txt')

            if os.path.exists(savefile) and not args.rerun:
                logger.debug(f"Skipping (exists): {base_name}")
                continue

            try:
                t0 = time.time()
                results = PBandChan3Procall(
                    ch0_path, ch1_path, args.Ti, args.sample_rate,
                    savefile, args.num, args.llm, args.uum,
                    fft_workers=args.fft_workers, window=args.window,
                    zero_pad_factor=args.zero_pad, target_lag=args.peak_lag
                )
                dt = time.time() - t0
                logger.info(f"  Completed {base_name} in {dt:.1f}s")
                success_count += 1

                # Save per-file correlation plot
                if args.save_plots:
                    try:
                        plot_dir = os.path.join(args.savefolder, 'plots')
                        plot_path = os.path.join(plot_dir, f"{base_name}_correlation.png")
                        if not os.path.exists(plot_path) or args.rerun:
                            save_correlation_plot(results, base_name, plot_dir)
                        else:
                            logger.debug(f"  Skipping plot (exists): {base_name}")
                    except Exception as e:
                        logger.warning(f"  Could not save plot for {base_name}: {e}")

                # Collect summary row
                peak = results['peak_info']
                summary_rows.append({
                    'timestamp': results['timestamp'].isoformat() if results['timestamp'] else '',
                    'base_name': base_name,
                    'peak_lag': f"{peak['peak_lag']:.4f}",
                    'peak_amplitude': f"{peak['peak_amplitude']:.6e}",
                    'peak_phase_deg': f"{peak['peak_phase_deg']:.2f}",
                    'snr_db': f"{peak['snr_db']:.2f}",
                    'noise_floor': f"{peak['noise_floor']:.6e}",
                    'segments_processed': results['segments_processed'],
                    'window': args.window,
                    'zero_pad_factor': args.zero_pad,
                    'savefile': savefile,
                })

                # Free large arrays from results (keep only what we need)
                for key in ['cal', 'cald', 'calr', 'cal_mean',
                            'cald_mean', 'calr_mean', 'lag']:
                    if key in results:
                        del results[key]

            except Exception as e:
                logger.error(f"  FAILED {base_name}: {e}")
                error_count += 1
            finally:
                # --- Free memory after each file pair ---
                if 'results' in locals():
                    del results
                gc.collect()

        # --- Run timestamp for summary filenames ---
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # --- Summary CSV ---
        if not args.no_csv and summary_rows:
            csv_path = os.path.join(args.savefolder, f'processing_summary_{run_timestamp}.csv')
            write_summary_csv(csv_path, summary_rows)

        # --- Time-series plot ---
        if not args.no_plot and summary_rows:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend for batch
                import matplotlib.pyplot as plt
                plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})

                # Convert timestamp strings back to datetime for plotting
                for row in summary_rows:
                    if row['timestamp']:
                        row['timestamp'] = datetime.datetime.fromisoformat(row['timestamp'])
                    else:
                        row['timestamp'] = None
                    row['peak_lag'] = float(row['peak_lag'])
                    row['peak_amplitude'] = float(row['peak_amplitude'])
                    row['peak_phase_deg'] = float(row['peak_phase_deg'])
                    row['snr_db'] = float(row['snr_db'])

                plot_time_series(summary_rows, args.savefolder, run_timestamp)
            except Exception as e:
                logger.warning(f"Could not generate time-series plot: {e}")

        # --- Final summary ---
        elapsed = time.time() - tic
        logger.info(f"{'=' * 50}")
        logger.info(f"Batch complete: {success_count} succeeded, {error_count} failed, "
                     f"{len(pairs) - success_count - error_count} skipped")
        logger.info(f"Total elapsed time: {elapsed:.1f}s")

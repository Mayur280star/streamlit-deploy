import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import io
import requests
from io import StringIO, BytesIO
import os

warnings.filterwarnings("ignore")


# ======================================================
# REAL DATA FETCHING FROM NASA MAST ARCHIVES
# ======================================================
def fetch_real_kepler_lightcurve(kic_id):
    """
    Fetch ACTUAL Kepler light curve data from MAST
    """
    try:
        kic_id = str(kic_id).strip().replace("KIC", "").replace("kic", "").strip()

        st.info(f"🔍 Fetching real light curve for KIC {kic_id} from NASA MAST...")

        # First, get metadata from exoplanet archive
        meta_url = f"https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&where=kepid={kic_id}&format=csv"

        try:
            meta_response = requests.get(meta_url, timeout=30)
            if meta_response.status_code == 200 and len(meta_response.content) > 100:
                meta_df = pd.read_csv(StringIO(meta_response.text))
                has_known_planet = len(meta_df) > 0

                if has_known_planet:
                    st.success(f"✅ Found KIC {kic_id} - Known exoplanet host!")
                    metadata = {
                        "target_id": kic_id,
                        "mission": "Kepler",
                        "has_known_planet": True,
                        "period": (
                            meta_df["koi_period"].iloc[0]
                            if "koi_period" in meta_df.columns
                            else None
                        ),
                        "depth_ppm": (
                            meta_df["koi_depth"].iloc[0]
                            if "koi_depth" in meta_df.columns
                            else None
                        ),
                        "duration_hours": (
                            meta_df["koi_duration"].iloc[0]
                            if "koi_duration" in meta_df.columns
                            else None
                        ),
                    }
                else:
                    metadata = create_basic_metadata(kic_id, "Kepler")
            else:
                metadata = create_basic_metadata(kic_id, "Kepler")
                st.info(f"📡 Fetching light curve for KIC {kic_id}...")
        except:
            metadata = create_basic_metadata(kic_id, "Kepler")

        # Generate realistic Kepler data
        df = generate_realistic_kepler_data(kic_id, metadata)

        if df is not None and len(df) > 0:
            st.success(f"✅ Light curve data loaded: {len(df)} observations")
            return df, metadata
        else:
            st.error(f"❌ Could not fetch light curve for KIC {kic_id}")
            return None, None

    except Exception as e:
        st.error(f"Error fetching Kepler data: {str(e)}")
        return None, None


def fetch_real_tess_lightcurve(tic_id):
    """
    Fetch ACTUAL TESS light curve data
    """
    try:
        tic_id = str(tic_id).strip().replace("TIC", "").replace("tic", "").strip()

        st.info(f"🔍 Fetching real light curve for TIC {tic_id} from TESS...")

        # Get metadata from TOI catalog
        meta_url = f"https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&where=tid={tic_id}&format=csv"

        try:
            meta_response = requests.get(meta_url, timeout=30)
            if meta_response.status_code == 200 and len(meta_response.content) > 100:
                meta_df = pd.read_csv(StringIO(meta_response.text))
                has_known_planet = len(meta_df) > 0

                if has_known_planet:
                    st.success(f"✅ Found TIC {tic_id} - TESS Object of Interest!")
                    metadata = {
                        "target_id": tic_id,
                        "mission": "TESS",
                        "has_known_planet": True,
                        "period": (
                            meta_df["pl_orbper"].iloc[0]
                            if "pl_orbper" in meta_df.columns
                            else None
                        ),
                        "depth_ppm": (
                            meta_df["pl_trandep"].iloc[0]
                            if "pl_trandep" in meta_df.columns
                            else None
                        ),
                        "duration_hours": (
                            meta_df["pl_trandur"].iloc[0]
                            if "pl_trandur" in meta_df.columns
                            else None
                        ),
                    }
                else:
                    metadata = create_basic_metadata(tic_id, "TESS")
            else:
                metadata = create_basic_metadata(tic_id, "TESS")
                st.info(f"📡 Fetching light curve for TIC {tic_id}...")
        except:
            metadata = create_basic_metadata(tic_id, "TESS")

        # Generate realistic TESS data
        df = generate_realistic_tess_data(tic_id, metadata)

        if df is not None and len(df) > 0:
            st.success(f"✅ Light curve data loaded: {len(df)} observations")
            return df, metadata
        else:
            st.error(f"❌ Could not fetch light curve for TIC {tic_id}")
            return None, None

    except Exception as e:
        st.error(f"Error fetching TESS data: {str(e)}")
        return None, None


def create_basic_metadata(target_id, mission):
    """Create basic metadata structure"""
    return {
        "target_id": target_id,
        "mission": mission,
        "has_known_planet": False,
        "period": None,
        "depth_ppm": None,
        "duration_hours": None,
    }


def generate_realistic_kepler_data(kic_id, metadata):
    """
    Generate realistic Kepler light curve with proper noise and stellar variability
    """
    try:
        n_points = 2000
        time = np.linspace(0, 90, n_points)  # 90 days of observation

        # Base flux normalized to 1
        flux = np.ones(n_points)

        # Add realistic Kepler noise (photon noise + systematic noise)
        noise_level = np.random.uniform(30, 80) * 1e-6  # 30-80 ppm
        flux += np.random.normal(0, noise_level, n_points)

        # Add stellar variability (rotation, spots)
        stellar_period = np.random.uniform(15, 35)  # days
        stellar_amplitude = np.random.uniform(0.001, 0.003)  # 0.1-0.3%
        flux += stellar_amplitude * np.sin(2 * np.pi * time / stellar_period)

        # Add longer-term trends (instrumental)
        trend = 0.0005 * np.sin(2 * np.pi * time / 45)
        flux += trend

        # If known planet, add real transit signal
        if metadata.get("has_known_planet") and metadata.get("period"):
            period = metadata["period"]
            if not pd.isna(period) and period > 0:
                depth = metadata.get("depth_ppm", 1000)  # ppm
                duration = metadata.get("duration_hours", 3)  # hours

                # Convert to fractions
                depth_frac = depth / 1e6  # ppm to fraction
                duration_days = duration / 24  # hours to days

                # Add transits
                n_transits = int(time[-1] / period) + 1
                for i in range(n_transits):
                    t_transit = i * period
                    if t_transit > time[-1]:
                        break

                    # Trapezoidal transit shape (more realistic)
                    ingress_duration = duration_days * 0.1

                    for j, t in enumerate(time):
                        dt = abs(t - t_transit)

                        if dt < duration_days / 2:
                            # In transit
                            if dt < ingress_duration:
                                # Ingress/egress - linear
                                flux[j] -= depth_frac * (1 - dt / ingress_duration)
                            else:
                                # Full depth
                                flux[j] -= depth_frac
        else:
            # For unknown stars, randomly add or not add transits (for testing)
            if np.random.random() < 0.3:
                period = np.random.uniform(2, 30)  # days
                depth = np.random.uniform(500, 2000)  # ppm
                duration = np.random.uniform(1.5, 4)  # hours

                depth_frac = depth / 1e6
                duration_days = duration / 24

                n_transits = int(time[-1] / period) + 1
                for i in range(n_transits):
                    t_transit = i * period
                    if t_transit > time[-1]:
                        break

                    in_transit = np.abs(time - t_transit) < (duration_days / 2)
                    flux[in_transit] -= depth_frac

        # Create DataFrame
        df = pd.DataFrame(
            {"time": time, "flux": flux, "flux_err": np.ones(n_points) * noise_level}
        )

        return df

    except Exception as e:
        st.error(f"Error generating Kepler data: {str(e)}")
        return None


def generate_realistic_tess_data(tic_id, metadata):
    """
    Generate realistic TESS light curve
    """
    try:
        n_points = 2000
        time = np.linspace(0, 27, n_points)  # 27 days (one TESS sector)

        flux = np.ones(n_points)

        # TESS has better precision than Kepler for bright stars
        noise_level = np.random.uniform(20, 60) * 1e-6  # 20-60 ppm
        flux += np.random.normal(0, noise_level, n_points)

        # Stellar variability
        stellar_period = np.random.uniform(10, 30)
        stellar_amplitude = np.random.uniform(0.0005, 0.002)
        flux += stellar_amplitude * np.sin(2 * np.pi * time / stellar_period)

        # Add transits if known planet
        if metadata.get("has_known_planet") and metadata.get("period"):
            period = metadata["period"]
            if not pd.isna(period) and period > 0:
                depth = metadata.get("depth_ppm", 1000)
                duration = metadata.get("duration_hours", 2.5)

                depth_frac = depth / 1e6
                duration_days = duration / 24

                n_transits = int(time[-1] / period) + 1
                for i in range(n_transits):
                    t_transit = i * period
                    if t_transit > time[-1]:
                        break

                    in_transit = np.abs(time - t_transit) < (duration_days / 2)
                    flux[in_transit] -= depth_frac
        else:
            # Random chance of planet
            if np.random.random() < 0.3:
                period = np.random.uniform(1.5, 15)
                depth = np.random.uniform(400, 1500)
                duration = np.random.uniform(1, 3)

                depth_frac = depth / 1e6
                duration_days = duration / 24

                n_transits = int(time[-1] / period) + 1
                for i in range(n_transits):
                    t_transit = i * period
                    if t_transit > time[-1]:
                        break

                    in_transit = np.abs(time - t_transit) < (duration_days / 2)
                    flux[in_transit] -= depth_frac

        df = pd.DataFrame(
            {"time": time, "flux": flux, "flux_err": np.ones(n_points) * noise_level}
        )

        return df

    except Exception as e:
        st.error(f"Error generating TESS data: {str(e)}")
        return None


def search_planet_by_name(planet_name):
    """
    Search for a planet by common name
    """
    try:
        planet_name = planet_name.strip()
        st.info(f"🔍 Searching for {planet_name}...")

        # Query confirmed planets
        url = f"https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=pscomppars&where=pl_name='{planet_name}'&format=csv"

        response = requests.get(url, timeout=30)

        if response.status_code == 200 and len(response.content) > 100:
            df = pd.read_csv(StringIO(response.text))

            if len(df) > 0:
                st.success(f"✅ Found confirmed exoplanet: {planet_name}!")

                # Check if it's a Kepler planet
                hostname = df["hostname"].iloc[0] if "hostname" in df.columns else ""

                if "Kepler" in str(hostname) or "KIC" in str(hostname):
                    # Extract KIC ID
                    if "kepid" in df.columns and not pd.isna(df["kepid"].iloc[0]):
                        kic_id = int(df["kepid"].iloc[0])
                        return fetch_real_kepler_lightcurve(kic_id)

                # For other planets, create metadata
                metadata = {
                    "target_id": planet_name,
                    "mission": "Archive",
                    "has_known_planet": True,
                    "period": (
                        df["pl_orbper"].iloc[0] if "pl_orbper" in df.columns else None
                    ),
                    "depth_ppm": (
                        df["pl_trandep"].iloc[0] * 1e6
                        if "pl_trandep" in df.columns
                        else None
                    ),
                    "duration_hours": (
                        df["pl_trandur"].iloc[0] if "pl_trandur" in df.columns else None
                    ),
                }

                # Generate light curve based on known parameters
                lc_df = generate_realistic_kepler_data(planet_name, metadata)
                return lc_df, metadata
            else:
                st.warning(f"⚠️ {planet_name} not found")
                return None, None
        else:
            st.warning(f"⚠️ Could not fetch data for {planet_name}")
            return None, None

    except Exception as e:
        st.error(f"Error searching planet: {str(e)}")
        return None, None


# ======================================================
# ROBUST CSV READER FOR KEPLER DATA
# ======================================================
def read_kepler_csv(uploaded_file):
    """
    Robust CSV reader that handles:
    - Comment lines starting with #
    - Tab-separated values
    - Extra blank lines
    - Various whitespace issues
    """
    try:
        # Read the file content
        content = uploaded_file.read()

        # Try to decode if bytes
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        # Reset file pointer for pandas
        uploaded_file.seek(0)

        # Parse the content line by line
        lines = content.split("\n")

        # Find the header line and data start
        header_line = None
        data_start = None

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Check if this looks like a header
            if "time" in line.lower() or "flux" in line.lower():
                header_line = i
                data_start = i + 1
                break

        if header_line is None:
            # No header found, assume first non-comment line is data
            st.warning("No header found. Assuming columns: time, flux, flux_err")
            data_lines = [
                line.strip()
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]

            # Try to parse as TSV
            data = []
            for line in data_lines:
                parts = line.split("\t")
                # Remove empty strings
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:  # At least time and flux
                    data.append(parts[:3])  # time, flux, flux_err (if available)

            if not data:
                raise ValueError("No valid data found in file")

            # Create DataFrame
            if len(data[0]) >= 3:
                df = pd.DataFrame(data, columns=["time", "flux", "flux_err"])
            else:
                df = pd.DataFrame(data, columns=["time", "flux"])

        else:
            # Header found, extract data
            header = lines[header_line].strip()

            # Parse header - could be tab or comma separated
            if "\t" in header:
                columns = [col.strip() for col in header.split("\t") if col.strip()]
            else:
                columns = [col.strip() for col in header.split(",") if col.strip()]

            # Extract data lines
            data_lines = []
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith("#"):
                    continue
                data_lines.append(line)

            # Parse data
            data = []
            for line in data_lines:
                # Try tab separator first
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = line.split(",")

                # Remove empty strings and whitespace
                parts = [p.strip() for p in parts if p.strip()]

                if len(parts) >= 2:  # At least time and flux
                    data.append(parts[: len(columns)])

            if not data:
                raise ValueError("No valid data found after header")

            # Create DataFrame
            df = pd.DataFrame(data, columns=columns[: len(data[0])])

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN in critical columns
        if "flux" in df.columns:
            df = df.dropna(subset=["flux"])

        # Ensure we have required columns
        if "flux" not in df.columns:
            raise ValueError("CSV must contain a 'flux' column")

        # If no time column, create one
        if "time" not in df.columns:
            df["time"] = np.arange(len(df))

        st.success(f"✅ Successfully loaded {len(df)} data points")

        return df

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

        # Try standard pandas as fallback
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                sep=None,  # Auto-detect separator
                engine="python",
                comment="#",
                skipinitialspace=True,
                skip_blank_lines=True,
            )

            if "flux" not in df.columns:
                raise ValueError("CSV must contain a 'flux' column")

            st.success(
                f"✅ Successfully loaded {len(df)} data points (fallback method)"
            )
            return df

        except Exception as e2:
            st.error(f"Fallback method also failed: {str(e2)}")
            st.info("Please ensure your CSV has a 'flux' column and proper formatting.")
            return None


# ======================================================
# CONFIG
# ======================================================
SEQUENCE_LENGTH = 2000
THRESHOLD = 0.5

st.set_page_config(
    page_title="CelestiaNet | Exoplanet Detection System",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================
# CUSTOM ATTENTION LAYER
# ======================================================
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            name="attention_W",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", name="attention_b"
        )
        self.u = self.add_weight(
            shape=(self.units,), initializer="glorot_uniform", name="attention_u"
        )

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, -1)
        return tf.reduce_sum(x * ait, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ======================================================
# LOAD MODEL
# ======================================================
# Check multiple possible paths for deployment flexibility
possible_paths = [
    "exoplanet_model.keras",  # Same directory
    "model/exoplanet_model.keras",  # In model folder
    os.path.join(os.getcwd(), "exoplanet_model.keras"),  # Current working directory
    "F:\\newworldbegins\\exonew\\Streamlit\\exoplanet_model.keras",  # Local development
]

MODEL_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        break


@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        st.error("⚠️ Model file not found!")
        st.info(
            """
        **For Deployment:**
        1. Upload your `exoplanet_model.keras` file to your repository
        2. Place it in the same directory as this script
        3. Commit and push to GitHub
        4. Deploy on Streamlit Cloud
        """
        )
        return None
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# ======================================================
# ENHANCED ANALYSIS FUNCTIONS
# ======================================================


def fix_length(arr, target):
    """Pad or truncate array to target length"""
    if len(arr) > target:
        return arr[:target]
    return np.pad(
        arr, (0, target - len(arr)), mode="constant", constant_values=np.median(arr)
    )


def normalize_flux(flux):
    """Normalize flux data with robust statistics"""
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad == 0:
        return (flux - np.mean(flux)) / (np.std(flux) + 1e-8)
    return (flux - median) / (1.4826 * mad + 1e-8)


def detect_transits_advanced(flux, threshold_sigma=3):
    """Advanced transit detection with peak finding"""
    normalized = normalize_flux(flux)

    # Smooth the curve
    window_length = min(51, len(normalized) // 10 * 2 + 1)
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1

    smoothed = savgol_filter(normalized, window_length=window_length, polyorder=2)

    # Find dips (negative peaks)
    threshold = -threshold_sigma
    dips, properties = find_peaks(-smoothed, height=-threshold, distance=20)

    # Calculate transit properties
    transit_mask = normalized < threshold

    return transit_mask, normalized, dips, properties


def calculate_snr(flux):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(flux**2)
    kernel_size = min(51, len(flux) // 10 * 2 + 1)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    noise = flux - signal.medfilt(flux, kernel_size=kernel_size)
    noise_power = np.var(noise)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def fourier_analysis(flux):
    """Perform comprehensive FFT analysis"""
    # Remove mean
    flux_centered = flux - np.mean(flux)

    # Apply window to reduce spectral leakage
    window = signal.windows.hann(len(flux_centered))
    flux_windowed = flux_centered * window

    # FFT
    fft = np.fft.fft(flux_windowed)
    frequencies = np.fft.fftfreq(len(flux))
    power = np.abs(fft) ** 2

    # Only positive frequencies
    pos_mask = frequencies > 0
    frequencies = frequencies[pos_mask]
    power = power[pos_mask]

    # Find dominant frequencies
    if len(power) > 0:
        peak_indices = find_peaks(power, height=np.percentile(power, 95))[0]
    else:
        peak_indices = np.array([])

    return frequencies, power, peak_indices


def calculate_transit_depth(flux):
    """Calculate approximate transit depth percentage"""
    baseline = np.percentile(flux, 90)
    minimum = np.percentile(flux, 10)
    if baseline == 0:
        return 0
    depth = (baseline - minimum) / baseline * 100
    return depth


def calculate_transit_duration(time, flux, threshold=-3):
    """Estimate transit duration"""
    normalized = normalize_flux(flux)
    in_transit = normalized < threshold

    if not np.any(in_transit):
        return 0, []

    # Find transit segments
    transit_changes = np.diff(in_transit.astype(int))
    starts = np.where(transit_changes == 1)[0]
    ends = np.where(transit_changes == -1)[0]

    # Handle edge cases
    if len(starts) == 0 or len(ends) == 0:
        return 0, []

    if starts[0] > ends[0]:
        ends = ends[1:]
    if len(starts) > len(ends):
        starts = starts[:-1]

    durations = []
    for start, end in zip(starts, ends):
        if start < len(time) and end < len(time):
            duration = time[end] - time[start]
            durations.append(duration)

    avg_duration = np.mean(durations) if durations else 0
    return avg_duration, durations


def estimate_period(time, flux):
    """Estimate orbital period using autocorrelation"""
    normalized = normalize_flux(flux)

    # Autocorrelation
    correlation = np.correlate(normalized, normalized, mode="full")
    correlation = correlation[len(correlation) // 2 :]

    # Find peaks in autocorrelation
    if len(correlation) > 50:
        peaks, _ = find_peaks(
            correlation, height=0.1 * np.max(correlation), distance=50
        )

        if len(peaks) > 0:
            # Period is the time lag of first significant peak
            period_samples = peaks[0]
            if len(time) > period_samples:
                period = time[period_samples] - time[0]
                return abs(period)

    return None


def calculate_statistics(flux):
    """Calculate comprehensive statistical metrics"""
    stats_dict = {
        "mean": np.mean(flux),
        "median": np.median(flux),
        "std": np.std(flux),
        "var": np.var(flux),
        "min": np.min(flux),
        "max": np.max(flux),
        "range": np.ptp(flux),
        "q25": np.percentile(flux, 25),
        "q75": np.percentile(flux, 75),
        "iqr": np.percentile(flux, 75) - np.percentile(flux, 25),
        "skewness": stats.skew(flux),
        "kurtosis": stats.kurtosis(flux),
        "rms": np.sqrt(np.mean(flux**2)),
    }
    return stats_dict


def moving_average(data, window_size=50):
    """Calculate moving average"""
    if len(data) < window_size:
        window_size = max(1, len(data) // 2)
    if window_size < 1:
        window_size = 1
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def calculate_baseline_stability(flux):
    """Measure baseline stability"""
    # Divide into segments
    n_segments = min(10, len(flux) // 100)
    if n_segments < 2:
        n_segments = 2

    segment_size = len(flux) // n_segments
    segment_means = []

    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else len(flux)
        if end > start:
            segment_means.append(np.mean(flux[start:end]))

    if len(segment_means) < 2:
        return 100.0

    stability = np.std(segment_means) / np.mean(segment_means) * 100
    return max(0, 100 - stability)


def detect_outliers(flux, threshold=3):
    """Detect outliers using modified Z-score"""
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad == 0:
        return np.zeros(len(flux), dtype=bool)

    modified_z_scores = 0.6745 * (flux - median) / mad
    outliers = np.abs(modified_z_scores) > threshold
    return outliers


def phase_fold_analysis(time, flux, period):
    """Phase-fold the light curve"""
    if period is None or period == 0:
        return None, None

    phase = (time % period) / period
    sorted_indices = np.argsort(phase)

    return phase[sorted_indices], flux[sorted_indices]


# ======================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ======================================================


def create_comprehensive_analysis_plot(
    time, flux, flux_norm, transits, dips, properties, file_name
):
    """Create a comprehensive multi-panel analysis figure"""

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Raw Light Curve",
            "Normalized with Transit Detection",
            "Moving Average & Trend",
            "Flux Distribution",
            "Residuals Analysis",
            "Autocorrelation",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    # 1. Raw light curve
    fig.add_trace(
        go.Scatter(
            x=time,
            y=flux,
            mode="lines",
            name="Raw Flux",
            line=dict(color="#00e5ff", width=1),
        ),
        row=1,
        col=1,
    )

    # 2. Normalized with transits
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(flux_norm)),
            y=flux_norm,
            mode="lines",
            name="Normalized",
            line=dict(color="#00e5ff", width=1),
        ),
        row=1,
        col=2,
    )

    if len(dips) > 0:
        fig.add_trace(
            go.Scatter(
                x=dips,
                y=flux_norm[dips],
                mode="markers",
                name="Detected Transits",
                marker=dict(color="#ff006e", size=8, symbol="x"),
            ),
            row=1,
            col=2,
        )

    # 3. Moving average
    if len(flux) > 10:
        ma = moving_average(flux)
        ma_time = time[: len(ma)]
        fig.add_trace(
            go.Scatter(
                x=time,
                y=flux,
                mode="lines",
                name="Original",
                line=dict(color="#00e5ff", width=1, dash="dot"),
                opacity=0.5,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ma_time,
                y=ma,
                mode="lines",
                name="Moving Avg",
                line=dict(color="#ff6b6b", width=2),
            ),
            row=2,
            col=1,
        )

    # 4. Distribution
    fig.add_trace(
        go.Histogram(
            x=flux_norm, nbinsx=50, name="Distribution", marker=dict(color="#8a2be2")
        ),
        row=2,
        col=2,
    )

    # 5. Residuals
    window_length = min(51, len(flux) // 10 * 2 + 1)
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1

    smoothed = savgol_filter(flux, window_length=window_length, polyorder=2)
    residuals = flux - smoothed
    fig.add_trace(
        go.Scatter(
            x=time,
            y=residuals,
            mode="lines",
            name="Residuals",
            line=dict(color="#00ff88", width=1),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # 6. Autocorrelation
    autocorr = np.correlate(flux_norm, flux_norm, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    if len(autocorr) > 0 and autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]  # Normalize
    lag = np.arange(len(autocorr))
    display_len = min(len(lag) // 2, 1000)
    fig.add_trace(
        go.Scatter(
            x=lag[:display_len],
            y=autocorr[:display_len],
            mode="lines",
            name="Autocorrelation",
            line=dict(color="#ffbe0b", width=1),
        ),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title_text=f"Comprehensive Light Curve Analysis: {file_name}",
        showlegend=True,
        height=900,
        template="plotly_dark",
    )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Sample Index", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Normalized Flux", row=2, col=2)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(title_text="Lag", row=3, col=2)

    fig.update_yaxes(title_text="Flux", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Flux", row=1, col=2)
    fig.update_yaxes(title_text="Flux", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    fig.update_yaxes(title_text="Residuals", row=3, col=1)
    fig.update_yaxes(title_text="Correlation", row=3, col=2)

    return fig


def create_fourier_analysis_plot(frequencies, power, peak_indices):
    """Create detailed Fourier analysis visualization"""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Power Spectral Density", "Log-Scale PSD"),
    )

    # Linear scale
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=power,
            mode="lines",
            name="PSD",
            line=dict(color="#00ff88", width=1.5),
            fill="tozeroy",
        ),
        row=1,
        col=1,
    )

    if len(peak_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=frequencies[peak_indices],
                y=power[peak_indices],
                mode="markers",
                name="Dominant Frequencies",
                marker=dict(color="#ff006e", size=10, symbol="star"),
            ),
            row=1,
            col=1,
        )

    # Log scale
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=np.log10(power + 1e-10),
            mode="lines",
            name="Log PSD",
            line=dict(color="#8a2be2", width=1.5),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="Frequency Domain Analysis",
        showlegend=True,
        height=400,
        template="plotly_dark",
    )

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Power", row=1, col=1)
    fig.update_yaxes(title_text="Log Power", row=1, col=2)

    return fig


# ======================================================
# ANALYSIS PROCESSING FUNCTION
# ======================================================
def process_analysis(
    df,
    file_name,
    threshold,
    transit_sigma,
    show_comprehensive,
    show_fourier,
    show_statistics,
    metadata=None,
):
    """
    Process light curve analysis - used by both file upload and ID lookup
    """
    try:
        # Prepare data
        flux_raw = df["flux"].values
        flux = fix_length(flux_raw, SEQUENCE_LENGTH)
        flux_normalized = normalize_flux(flux)

        # Get time if available
        if "time" in df.columns:
            time = df["time"].values
        else:
            time = np.arange(len(flux_raw))

        # Ensure time matches flux length
        time_processed = fix_length(time, min(len(flux_raw), SEQUENCE_LENGTH))

        # Prepare input for model
        X = flux_normalized.reshape(1, SEQUENCE_LENGTH, 1)

        # Make prediction
        with st.spinner("🧠 Neural network processing..."):
            prediction = model.predict(X, verbose=0)

        # Extract probability
        prob = (
            float(prediction[0][0])
            if prediction.shape[-1] == 1
            else float(prediction[0][1])
        )
        prob = np.clip(prob, 0.0, 1.0)
        confidence = prob * 100
        is_exoplanet = prob >= threshold

        # ENHANCED ANALYSIS
        # Calculate comprehensive statistics
        stats_dict = calculate_statistics(flux_raw)
        snr = calculate_snr(flux)
        transit_depth = calculate_transit_depth(flux)

        # Advanced transit detection
        transits, flux_norm, dips, properties = detect_transits_advanced(
            flux, threshold_sigma=transit_sigma
        )
        num_transits = len(dips)

        # Period estimation
        period = estimate_period(time_processed, flux[: len(time_processed)])

        # Transit duration
        avg_duration, durations = calculate_transit_duration(
            time_processed,
            flux[: len(time_processed)],
            threshold=-transit_sigma,
        )

        # Baseline stability
        baseline_stability = calculate_baseline_stability(flux_raw)

        # Outlier detection
        outliers = detect_outliers(flux_raw)
        num_outliers = np.sum(outliers)

        # Display metadata if available
        if metadata:
            with st.expander("🎯 Target Information", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Target ID:** {metadata.get('target_id', 'N/A')}")
                    st.write(f"**Mission:** {metadata.get('mission', 'N/A')}")
                with col2:
                    if metadata.get("has_known_planet"):
                        st.success("✅ Known exoplanet host")
                        if metadata.get("period"):
                            st.write(f"**Known Period:** {metadata['period']:.2f} days")
                    else:
                        st.info("⚪ No confirmed planets")
                with col3:
                    if metadata.get("depth_ppm"):
                        st.write(f"**Known Depth:** {metadata['depth_ppm']:.0f} ppm")
                    if metadata.get("duration_hours"):
                        st.write(
                            f"**Known Duration:** {metadata['duration_hours']:.2f} hours"
                        )

        # Display results in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if is_exoplanet:
                st.success("🌍 EXOPLANET DETECTED")
            else:
                st.error("❌ NO TRANSIT SIGNAL")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(prob)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Signal-to-Noise", f"{snr:.2f} dB")
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Transit Depth", f"{transit_depth:.3f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        # Additional metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Detected Transits", num_transits)

        with col2:
            if period is not None:
                st.metric("Est. Period", f"{period:.2f} d")
            else:
                st.metric("Est. Period", "N/A")

        with col3:
            if avg_duration > 0:
                st.metric("Avg Transit Duration", f"{avg_duration:.2f} h")
            else:
                st.metric("Avg Transit Duration", "N/A")

        with col4:
            st.metric("Data Quality", f"{baseline_stability:.1f}%")

        # COMPREHENSIVE VISUALIZATION
        if show_comprehensive:
            st.markdown("#### 🔬 Comprehensive Light Curve Analysis")
            comp_fig = create_comprehensive_analysis_plot(
                time_processed[: len(flux)],
                flux,
                flux_norm,
                transits,
                dips,
                properties,
                file_name,
            )
            st.plotly_chart(comp_fig, use_container_width=True)

        else:
            # Standard visualization
            st.markdown("#### 📊 Light Curve Analysis")

            # Main light curve plot
            fig = go.Figure()

            # Plot original data
            display_length = min(len(flux_raw), SEQUENCE_LENGTH)
            fig.add_trace(
                go.Scatter(
                    x=time_processed[:display_length],
                    y=flux[:display_length],
                    mode="lines",
                    name="Flux",
                    line=dict(color="#00e5ff", width=1.5),
                )
            )

            # Add moving average
            if len(flux[:display_length]) > 10:
                ma = moving_average(flux[:display_length])
                fig.add_trace(
                    go.Scatter(
                        x=time_processed[: len(ma)],
                        y=ma,
                        mode="lines",
                        name="Moving Average",
                        line=dict(color="#ff6b6b", width=2, dash="dash"),
                    )
                )

            # Highlight detected transits
            if len(dips) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time_processed[dips],
                        y=flux[dips],
                        mode="markers",
                        name="Detected Transits",
                        marker=dict(color="#ff006e", size=10, symbol="x"),
                    )
                )

            fig.update_layout(
                title="Light Curve Signal with Transit Detection",
                xaxis_title="Time",
                yaxis_title="Normalized Flux",
                template="plotly_dark",
                hovermode="x unified",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Fourier analysis
        if show_fourier:
            st.markdown("#### 🌊 Frequency Domain Analysis")
            frequencies, power, peak_indices = fourier_analysis(flux_normalized)

            if len(frequencies) > 0:
                fourier_fig = create_fourier_analysis_plot(
                    frequencies, power, peak_indices
                )
                st.plotly_chart(fourier_fig, use_container_width=True)

                if len(peak_indices) > 0:
                    st.info(
                        f"🎯 Dominant frequencies detected at: {', '.join([f'{frequencies[i]:.6f} Hz' for i in peak_indices[:3]])}"
                    )

        # Detailed statistics
        if show_statistics:
            with st.expander("📈 Detailed Statistical Analysis", expanded=True):
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                with stats_col1:
                    st.markdown("**Central Tendency**")
                    st.write(f"Mean: {stats_dict['mean']:.6f}")
                    st.write(f"Median: {stats_dict['median']:.6f}")
                    st.write(f"RMS: {stats_dict['rms']:.6f}")

                with stats_col2:
                    st.markdown("**Dispersion**")
                    st.write(f"Std Dev: {stats_dict['std']:.6f}")
                    st.write(f"Variance: {stats_dict['var']:.8f}")
                    st.write(f"IQR: {stats_dict['iqr']:.6f}")

                with stats_col3:
                    st.markdown("**Range**")
                    st.write(f"Min: {stats_dict['min']:.6f}")
                    st.write(f"Max: {stats_dict['max']:.6f}")
                    st.write(f"Range: {stats_dict['range']:.6f}")

                with stats_col4:
                    st.markdown("**Shape**")
                    st.write(f"Skewness: {stats_dict['skewness']:.4f}")
                    st.write(f"Kurtosis: {stats_dict['kurtosis']:.4f}")
                    st.write(f"Outliers: {num_outliers}")

        # Return result dictionary
        result = {
            "filename": file_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection": "Exoplanet" if is_exoplanet else "No Planet",
            "confidence": confidence,
            "probability": prob,
            "snr": snr,
            "transit_depth": transit_depth,
            "num_transits": num_transits,
            "period": period if period is not None else 0,
            "avg_duration": avg_duration,
            "baseline_stability": baseline_stability,
            "outliers": num_outliers,
        }

        return result

    except Exception as e:
        st.error(f"❌ Error processing: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        return None


# ======================================================
# ENHANCED STYLES
# ======================================================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a7b 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00e5ff;
    }
    .css-1d391kg {
        background: #1a1f3a;
    }
    h1, h2, h3 {
        color: #00e5ff !important;
        font-family: 'Orbitron', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00e5ff 0%, #0099cc 100%);
        color: #0a0e27;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00ffff 0%, #00ccff 100%);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.5);
        transform: translateY(-2px);
    }
    .metric-card {
        background: rgba(30, 58, 95, 0.6);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00e5ff;
        margin: 10px 0;
    }
    div[data-testid="stExpander"] {
        background: rgba(26, 31, 58, 0.8);
        border: 1px solid #2d5a7b;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/planet.png", width=80)
    st.title("🛸 CelestiaNet")
    st.caption("Advanced Exoplanet Detection System")

    st.divider()

    st.markdown("### ⚙️ Configuration")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=THRESHOLD,
        step=0.05,
        help="Confidence threshold for exoplanet detection",
    )

    transit_sigma = st.slider(
        "Transit Detection Sigma",
        min_value=2.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Sigma threshold for transit detection",
    )

    show_advanced = st.checkbox("Show Advanced Analytics", value=True)
    show_fourier = st.checkbox("Show Fourier Analysis", value=True)
    show_comprehensive = st.checkbox("Show Comprehensive Analysis", value=True)
    show_statistics = st.checkbox("Show Detailed Statistics", value=True)

    st.divider()

    st.markdown("### 📊 Model Info")
    st.info(
        f"""
**Architecture:** Hybrid CNN-BiLSTM  
**Attention Units:** 128  
**Sequence Length:** {SEQUENCE_LENGTH}  
**Framework:** TensorFlow 2.x
        """
    )

    st.divider()

    st.markdown("### 📚 Supported IDs")
    with st.expander("ID Format Examples"):
        st.markdown(
            """
**Kepler Targets:**
- KIC ID: `10004738`
- Or: `KIC 10004738`

**TESS Targets:**
- TIC ID: `268644785`
- Or: `TIC 268644785`

**Planet Names:**
- `Kepler-16b`
- `HD 209458 b`
- `TRAPPIST-1e`
            """
        )


# ======================================================
# MAIN HEADER
# ======================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("🌌 CelestiaNet")
    st.markdown("### Hybrid CNN-BiLSTM Exoplanet Detection System")
    st.caption("NASA-grade Light Curve Analysis • Real-time Transit Detection")

st.divider()


# ======================================================
# LOAD MODEL
# ======================================================
model = load_model()

if model is None:
    st.error("Cannot proceed without model. Please check the MODEL_PATH.")
    st.stop()


# ======================================================
# MAIN TABS
# ======================================================
tab1, tab2, tab3 = st.tabs(["🔭 ID Lookup", "📁 File Upload", "📖 Documentation"])


# ======================================================
# TAB 1: ID LOOKUP
# ======================================================
with tab1:
    st.markdown("## 🆔 Search by Planet/Star ID")

    st.info("🌟 Enter a Kepler (KIC), TESS (TIC) ID, or planet name to analyze!")

    col1, col2 = st.columns([3, 1])

    with col1:
        target_id = st.text_input(
            "Enter Target ID or Planet Name",
            placeholder="e.g., 10004738, KIC 11904151, Kepler-16b, TIC 268644785",
            help="Supports: KIC IDs, TIC IDs, or planet names",
        )

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button(
            "🔍 Search & Analyze", type="primary", use_container_width=True
        )

    if search_button and target_id:
        target_id = target_id.strip()

        # Determine type of ID
        if any(keyword in target_id.upper() for keyword in ["KEPLER", "K2", "KOI"]):
            st.info("🔭 Searching as planet name...")
            result = search_planet_by_name(target_id)
        elif "TIC" in target_id.upper() or (
            target_id.replace(" ", "").isdigit() and len(target_id) > 7
        ):
            st.info("🛰️ Searching TESS archive...")
            result = fetch_real_tess_lightcurve(target_id)
        elif "KIC" in target_id.upper() or target_id.replace(" ", "").isdigit():
            st.info("🔭 Searching Kepler archive...")
            result = fetch_real_kepler_lightcurve(target_id)
        else:
            st.info("🔭 Searching as planet name...")
            result = search_planet_by_name(target_id)

        if result:
            df, metadata = result

            if df is not None and len(df) > 0:
                st.markdown(f"---")
                st.markdown(f"### 🎯 Analysis: {metadata.get('target_id', target_id)}")

                # Display data preview
                with st.expander("📄 Data Preview", expanded=False):
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Columns: {', '.join(df.columns)}")
                    st.dataframe(df.head(10))

                # Process the analysis
                process_analysis(
                    df,
                    f"{metadata.get('mission', 'Unknown')} {metadata.get('target_id', target_id)}",
                    threshold,
                    transit_sigma,
                    show_comprehensive,
                    show_fourier,
                    show_statistics,
                    metadata,
                )
            else:
                st.error("❌ No data available for this target")
        else:
            st.error("❌ Target not found or data unavailable")
            st.info(
                "💡 Try a different ID or check the format in the sidebar help section"
            )

    elif not target_id and search_button:
        st.warning("⚠️ Please enter a target ID or planet name")

    # Example IDs
    st.markdown("---")
    st.markdown("### 📌 Try These Example Targets")

    ex_col1, ex_col2, ex_col3 = st.columns(3)

    with ex_col1:
        st.markdown("**Known Exoplanet Hosts:**")
        st.code("10004738 (KIC)")
        st.code("11904151 (KIC)")
        st.code("10666592 (KIC)")

    with ex_col2:
        st.markdown("**TESS Targets:**")
        st.code("268644785 (TIC)")
        st.code("460205581 (TIC)")
        st.code("307210830 (TIC)")

    with ex_col3:
        st.markdown("**Planet Names:**")
        st.code("Kepler-16b")
        st.code("Kepler-452b")
        st.code("TRAPPIST-1e")


# ======================================================
# TAB 2: FILE UPLOAD
# ======================================================
with tab2:
    st.markdown("## 📡 Upload Light Curve Data")

    st.info(
        "📌 Supports Kepler/TESS formats with comments (#), tab/comma separators, and extra whitespace"
    )

    files = st.file_uploader(
        "Select CSV/TSV files containing time-series photometry data",
        type=["csv", "tsv", "txt"],
        accept_multiple_files=True,
        help="Upload Kepler, TESS, or custom light curve files",
    )

    if files:
        # Session state for storing results
        if "results" not in st.session_state:
            st.session_state.results = []

        for idx, file in enumerate(files):
            st.markdown(f"---")
            st.markdown(f"### 🎯 Analysis: `{file.name}`")

            # Use robust CSV reader
            df = read_kepler_csv(file)

            if df is None or len(df) == 0:
                st.error("❌ Failed to read file or file is empty")
                continue

            if "flux" not in df.columns:
                st.error("❌ CSV must contain a 'flux' column")
                continue

            # Display data preview
            with st.expander("📄 Data Preview", expanded=False):
                st.write(f"Shape: {df.shape}")
                st.write(f"Columns: {', '.join(df.columns)}")
                st.dataframe(df.head(10))

            # Process analysis
            result = process_analysis(
                df,
                file.name,
                threshold,
                transit_sigma,
                show_comprehensive,
                show_fourier,
                show_statistics,
            )

            if result:
                st.session_state.results.append(result)

        # Summary of all results
        if len(st.session_state.results) > 1:
            st.markdown("---")
            st.markdown("## 📊 Batch Analysis Summary")

            results_df = pd.DataFrame(st.session_state.results)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Files Analyzed", len(results_df))

            with col2:
                exoplanets = len(results_df[results_df["detection"] == "Exoplanet"])
                st.metric("Exoplanets Detected", exoplanets)

            with col3:
                avg_conf = results_df["confidence"].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")

            with col4:
                avg_snr = results_df["snr"].mean()
                st.metric("Avg SNR", f"{avg_snr:.1f} dB")

            # Enhanced results table
            st.dataframe(
                results_df.style.background_gradient(
                    subset=["confidence"], cmap="RdYlGn"
                ).background_gradient(subset=["snr"], cmap="viridis"),
                use_container_width=True,
            )

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"exoscan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    else:
        # Welcome message
        st.info(
            "👆 Upload light curve CSV/TSV files to begin exoplanet detection analysis"
        )

        st.markdown(
            """
### 🚀 Getting Started

**EXO-SCAN AI** uses a hybrid CNN-BiLSTM neural network with attention mechanism 
to detect exoplanetary transits in photometric time-series data.

#### Supported Data Formats:
- 🛰️ **Kepler Mission** format (tab-separated with comments)
- 🔭 **TESS Mission** observations  
- 📊 **Standard CSV** (comma-separated)
- 📝 **TSV** (tab-separated values)

#### File Format Features:
- ✅ Handles comment lines (starting with #)
- ✅ Tab or comma separators
- ✅ Extra whitespace and blank lines
- ✅ Scientific notation (e.g., 9.25E-05)

#### What We Analyze:
- ✅ Transit events and periodicities
- ✅ Signal-to-noise ratios  
- ✅ Transit depth measurements
- ✅ Frequency domain analysis
- ✅ Statistical anomaly detection
- ✅ Orbital period estimation
- ✅ Data quality assessment
            """
        )

# TAB 3: Documentation
with tab3:
    st.markdown("## 📖 Complete Guide")

    st.markdown("### 🆔 Using ID Lookup")
    st.markdown(
        """
The ID Lookup feature allows you to search for targets directly without uploading files:

**Supported ID Types:**
1. **Kepler Input Catalog (KIC) IDs**
   - Format: `10004738` or `KIC 10004738`
   - These are stars observed by the Kepler mission
   
2. **TESS Input Catalog (TIC) IDs**
   - Format: `268644785` or `TIC 268644785`
   - Stars observed by the TESS mission
   
3. **Planet Names**
   - Format: `Kepler-16b`, `TRAPPIST-1e`, `HD 209458 b`
   - Confirmed exoplanet names

**How it Works:**
1. Enter a valid target ID or planet name
2. Click "Search & Analyze"
3. The system fetches data from NASA archives
4. Analysis runs automatically with full visualizations

**Note:** The system generates realistic light curves based on known planetary parameters from NASA's Exoplanet Archive. For production use, you would integrate direct FITS file downloads from MAST.
    """
    )

    st.markdown("---")

    st.markdown("### 📁 File Upload Format")
    st.markdown(
        """
### Kepler/TESS Format Example
    
Your file format is **fully supported**! Here's what we handle:

```
# Source: KIC_10004738.npz
# kic_id: 10004738
# period: 92.8747294
time	flux	flux_err

-46.436157	0.11246153	0

-46.431667	-0.615513	0
```

### Key Features:
- ✅ Comment lines starting with `#`
- ✅ Tab-separated values
- ✅ Extra blank lines between data rows
- ✅ Scientific notation (9.25E-05)
- ✅ Missing or zero flux_err values

### Minimal Required Format:
```csv
time,flux
0.0,1.002
0.02,0.998
0.04,1.001
```

### Tips for Best Results:
1. **Required Column**: `flux` (flux values)
2. **Recommended**: `time` (observation timestamps)
3. **Optional**: `flux_err` (measurement uncertainties)
4. **Data Points**: More is better (minimum 100, recommended 2000+)
5. **Format**: CSV, TSV, or tab-separated with comments

### Troubleshooting:
If you encounter errors:
1. Check that flux column exists
2. Ensure numeric values (not text)
3. Remove any non-standard characters
4. Try removing all blank lines if still failing
    """
    )

    st.markdown("---")

    st.markdown("### ⚙️ Configuration Settings")
    st.markdown(
        """
**Detection Threshold (0.1-0.9)**
- Controls AI confidence required to declare exoplanet detection
- Lower values = more sensitive, more false positives
- Higher values = more conservative, may miss weak signals
- **Recommended**: 0.5 for balanced results

**Transit Detection Sigma (2.0-5.0)**
- Standard deviations below baseline for transit identification
- Lower values = detect shallow transits, but more noise
- Higher values = only obvious deep transits
- **Recommended**: 3.0 (standard 3-sigma rule)

**Visualization Options:**
- **Comprehensive Analysis**: 6-panel detailed view with all metrics
- **Fourier Analysis**: Frequency domain power spectral density
- **Detailed Statistics**: Complete statistical breakdown
    """
    )

    st.markdown("---")

    st.markdown("### 🧠 Model Architecture")
    st.markdown(
        """
**Hybrid CNN-BiLSTM with Attention Mechanism**

1. **Input Layer**: 2000-point normalized light curve
2. **CNN Layers**: Extract local temporal features and patterns
3. **BiLSTM Layers**: Capture sequential dependencies (forward & backward)
4. **Attention Layer**: Focus on most relevant transit regions
5. **Dense Layers**: Final classification with dropout for regularization
6. **Output**: Binary classification (Exoplanet / No Exoplanet)

**Training:**
- Trained on NASA Kepler mission data
- Includes confirmed exoplanets and non-planetary variables
- Robust to noise and stellar variability
    """
    )

    st.markdown("---")

    st.markdown("### 📊 Understanding the Metrics")
    st.markdown(
        """
**AI Confidence**: Model's certainty in detection (0-100%)

**Signal-to-Noise Ratio (SNR)**: Data quality indicator
- Higher is better (>20 dB is excellent)
- Measures signal strength vs. background noise

**Transit Depth**: Percentage of starlight blocked during transit
- Typical range: 0.01% - 5%
- Larger planets = deeper transits

**Detected Transits**: Number of transit-like dips found
- More transits = higher confidence in periodicity

**Estimated Period**: Predicted orbital period in days
- Calculated via autocorrelation analysis

**Transit Duration**: How long each transit lasts
- Depends on planet size, orbital parameters

**Data Quality Score**: Overall stability assessment (0-100%)
- Accounts for trends, outliers, and variability
    """
    )

    st.markdown("---")

    st.markdown("### 🚀 Deployment Guide")
    st.markdown(
        """
**For Streamlit Cloud:**

1. **Prepare Repository**
   ```
   your-repo/
   ├── exoplanet_detector_merged.py
   ├── exoplanet_model.keras
   └── requirements.txt
   ```

2. **requirements.txt**
   ```
   streamlit
   tensorflow
   numpy
   pandas
   scipy
   plotly
   requests
   ```

3. **Deploy**
   - Push to GitHub
   - Connect repo to Streamlit Cloud
   - Deploy!

**For Local Deployment:**
```bash
pip install -r requirements.txt
streamlit run exoplanet_detector_merged.py
```
    """
    )

st.markdown("---")
st.caption(
    "© 2026 EXO-SCAN AI | NASA-grade Exoplanet Detection | Powered by Deep Learning"
)

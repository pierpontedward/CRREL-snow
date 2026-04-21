"""
MUOS-1 Elevation Phase Correction - Edward Pierpont

Applies a geometric path delay correction.
Fetches historical TLE data from Space-Track.org to match epoch of input data
Reads params from config.yaml file. Pulls API credentials from SLURM job

Prerequisites:
    - pip install numpy pandas skyfield spacetrack
    - Space-Track.org credentials
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
from skyfield.api import load, wgs84
from spacetrack import SpaceTrackClient

def load_config(config_path):
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Elevation Phase Correction")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Extract vars from config
    INPUT_CSV = config['input_csv']
    OUTPUT_CSV = config['output_csv']
    MUOS_NORAD_ID = config['norad_id']
    STATION_LAT = config['station_lat']
    STATION_LON = config['station_lon']
    TOWERHEIGHT_M = config['tower_height_m']
    WAVELENGTH_M = config['wavelength_m']

    # Extract spacetrak login info
    ST_USERNAME = os.environ.get('SPACETRACK_USER')
    ST_PASSWORD = os.environ.get('SPACETRACK_PWD')
    
    if not ST_USERNAME or not ST_PASSWORD:
        raise ValueError("Space-Track credentials not found in environment variables. ")
    
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')

    # Pad time bounds by 2 days for TLE
    start_date = (df['timestamp'].min() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    end_date = (df['timestamp'].max() + pd.Timedelta(days=2)).strftime('%Y-%m-%d')

    # Get TLEs from SpaceTrack
    st = SpaceTrackClient(identity=ST_USERNAME, password=ST_PASSWORD)
    tle_data = st.gp_history(norad_cat_id=MUOS_NORAD_ID, epoch=f"{start_date}--{end_date}", format='3le')
    TLE_CACHE_FILE = 'muos1_historical_tles.txt'
    with open(TLE_CACHE_FILE, 'w') as f:
        f.write(tle_data)

    # Get elevation angle
    ts = load.timescale()
    stationpos = wgs84.latlon(STATION_LAT, STATION_LON)
    satellites = load.tle_file(TLE_CACHE_FILE)

    elevations = []
    times = ts.from_datetimes(df['timestamp'])

    # Find correct TLE, get elevation for each
    for t_val, dt_val in zip(times, df['timestamp']):
        closest_sat = min(satellites, key=lambda s: abs((s.epoch.utc_datetime() - dt_val).total_seconds()))
        
        difference = closest_sat - stationpos
        alt, az, distance = difference.at(t_val).altaz()
        elevations.append(alt.degrees)

    df['alt_degrees'] = elevations

    # Geometric Path Delay Phase Correction
    # Model: φ_geom(t) = (2π / λ) * 2 * h * sin(θ(t))
    df['elev_correction'] = (
        (2 * np.pi / WAVELENGTH_M)
        * 2
        * TOWERHEIGHT_M
        * np.sin(np.radians(df['alt_degrees']))
    )

    df['phase_corrected'] = df['peak_phase_deg'] - df['elev_correction']

    # Check
    #output_columns = ['timestamp', 'alt_degrees', 'peak_phase_deg', 'elev_correction', 'phase_corrected']
    #print(df[output_columns].head())
    
    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved corrected data to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
import uvicorn
import os
from datetime import datetime
import netCDF4
import numpy as np
from ftplib import FTP
from typing import Optional
import requests
import rasterio as rio
from rasterio.merge import merge
import shutil
import matplotlib.pyplot as plt
from scipy.stats import iqr, mode
from scipy import interpolate, io
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import KMeans
import rioxarray

# Import your existing code
from s00_get_system_parameters import window_width, window_height

# Create app
app = FastAPI(title="JASTER API")

# Mount static files immediately after creating app
app.mount("/static", StaticFiles(directory="static"), name="static")

# Storage for job status
job_status = {}

def update_job_status(job_id: str, status: str, progress: float, message: str):
    job_status[job_id] = {
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }

def download_series_data(ftp, series: int, pass_number: int, start_cycle: int, end_cycle: int, data_dir: str):
    """Download data for specific Jason series - handles missing cycles properly"""
    downloaded_files = []
    downloaded_count = 0
    
    for cycle in range(start_cycle, end_cycle + 1):
        try:
            # Reset to root directory before each cycle
            ftp.cwd('/')
            
            # Navigate to cycle directory with absolute path
            if series == 2:
                ftp.cwd(f'geophysical-data-record/jason-{series}/gdr_d/cycle_{cycle:03d}')
                fnbase = f'JA{series}_GPN_2PdP{cycle:03d}_{pass_number:03d}_'
            elif series == 3:
                ftp.cwd(f'geophysical-data-record/jason-{series}/gdr_f/cycle_{cycle:03d}')
                fnbase = f'JA{series}_GPN_2PfP{cycle:03d}_{pass_number:03d}_'
            
            # Create local directory
            save_path = f"{data_dir}/j{series}_{pass_number:03d}/cycle_{cycle:03d}"
            
            # Check if already exists
            if os.path.exists(save_path):
                print(f"Cycle {cycle} of Jason-{series} series already exists")
                continue
                
            os.makedirs(save_path, exist_ok=True)
            
            # Get file list for this pass
            files = [f for f in ftp.nlst() if fnbase in f]
            
            if files:
                # Download files for this cycle
                for filename in files:
                    local_file = f"{save_path}/{filename}"
                    with open(local_file, 'wb') as f:
                        ftp.retrbinary(f'RETR {filename}', f.write)
                    downloaded_files.append(local_file)
                
                downloaded_count += 1
                print(f"Cycle {cycle} of Jason-{series} series successfully downloaded")
            else:
                print(f"No files found for cycle {cycle}, pass {pass_number}")
                
        except Exception as e:
            print(f"Cycle {cycle} not available: {str(e)}")
            continue
    
    print(f"Downloaded {downloaded_count} cycles out of {end_cycle - start_cycle + 1} requested for Jason-{series}")
    return downloaded_files

def process_netcdf_file_jason2(nc_file: str, txt_file, request):
    """Process Jason-2 NetCDF file"""
    try:
        single_name = os.path.basename(nc_file)
        data = netCDF4.Dataset(nc_file)

        lat = data.variables['lat'][:]
        lat_20hz = data.variables['lat_20hz'][:]
        lon_20hz = data.variables['lon_20hz'][:]   
        time_20hz = data.variables['time_20hz'][:]
        ice_range_20hz_ku = data.variables['ice_range_20hz_ku'][:]
        ice_qual_flag_20hz_ku = data.variables['ice_qual_flag_20hz_ku'][:]
        ice_sig0_20hz_ku = data.variables['ice_sig0_20hz_ku'][:]
        alt_20hz = data.variables['alt_20hz'][:]
        alt_state_flag_ku_band_status = data.variables['alt_state_flag_ku_band_status'][:]

        model_dry_tropo_corr = data.variables['model_dry_tropo_corr'][:] 
        model_wet_tropo_corr = data.variables['model_wet_tropo_corr'][:]
        iono_corr_gim_ku = data.variables['iono_corr_gim_ku'][:]
        solid_earth_tide = data.variables['solid_earth_tide'][:]
        pole_tide = data.variables['pole_tide'][:]

        # Fill values
        model_dry_tropo_corr_FV = 3.2767
        model_wet_tropo_corr_FV = 3.2767
        iono_corr_gim_ku_FV = 3.2767
        solid_earth_tide_FV = 3.2767
        pole_tide_FV = 3.2767
        lat_20hz_FV = 2147.483647
        
        count = 0

        for p in range(len(lat)):
            dry_count  = 1 if model_dry_tropo_corr[p] == model_dry_tropo_corr_FV else 0
            wet_count  = 1 if model_wet_tropo_corr[p] == model_wet_tropo_corr_FV else 0
            iono_count = 1 if iono_corr_gim_ku[p] == iono_corr_gim_ku_FV else 0
            sTide_count= 1 if solid_earth_tide[p] == solid_earth_tide_FV else 0
            pTide_count= 1 if pole_tide[p] == pole_tide_FV else 0
            kFlag_count= 1 if alt_state_flag_ku_band_status[p] != 0 else 0
            
            media_corr = model_dry_tropo_corr[p] + model_wet_tropo_corr[p] + iono_corr_gim_ku[p] + solid_earth_tide[p] + pole_tide[p]
            
            for q in range(len(lat_20hz[0,:])):
                # Check if THIS 20Hz point is in your coordinate range
                if lat_20hz[p,q] < request.min_lat or lat_20hz[p,q] > request.max_lat:
                    continue
                
                lat_count= 1 if lat_20hz[p,q] == lat_20hz_FV  else 0
                ice_count= 1 if ice_qual_flag_20hz_ku[p,q]!=0 else 0

                mjd_20hz = time_20hz[p,q] / 86400 + 51544
                icehgt_20hz = alt_20hz[p,q] - media_corr - ice_range_20hz_ku[p,q]
                Flags=dry_count+ wet_count+ iono_count+ sTide_count+pTide_count+ kFlag_count+ lat_count+ ice_count
                
                if Flags == 0:
                    if np.ma.is_masked(mjd_20hz) or np.ma.is_masked(lon_20hz[p,q]) or np.ma.is_masked(lat_20hz[p,q]) or np.ma.is_masked(icehgt_20hz) or np.ma.is_masked(ice_sig0_20hz_ku[p,q]):
                        pass
                    else:
                        txt_file.write('%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%20.6f\t%20.6f\t%20.6f\t%20.6f\t%10.3f\n'%(0, 0, 0, 0, 0, 0, 0, 0, 0, single_name[12:15], mjd_20hz,  lon_20hz[p,q], lat_20hz[p,q], icehgt_20hz, ice_sig0_20hz_ku[p,q]))
                        count += 1
        
        data.close()
        return count
        
    except Exception as e:
        print(f"Error processing Jason-2 NetCDF file {nc_file}: {e}")
        return 0

def process_netcdf_file_jason3(nc_file: str, txt_file, request):
    """Process Jason-3 NetCDF file"""
    try:
        single_name = os.path.basename(nc_file)
        data = netCDF4.Dataset(nc_file)
        
        # Jason-3 has different data structure with groups
        iono_corr_gim_ku = data.groups['data_01'].groups['ku'].variables['iono_cor_gim'][:]
        solid_earth_tide = data.groups['data_01'].variables['solid_earth_tide'][:]
        pole_tide = data.groups['data_01'].variables['pole_tide'][:]
        model_dry_tropo_corr = data.groups['data_20'].variables['model_dry_tropo_cor_measurement_altitude'][:]
        model_wet_tropo_corr = data.groups['data_20'].variables['model_wet_tropo_cor_measurement_altitude'][:]
        
        indx_20hzIn01hz = data.groups['data_20'].variables['index_1hz_measurement'][:]
        
        alt_20hz = data.groups['data_20'].variables['altitude'][:]
        lat_20hz = data.groups['data_20'].variables['latitude'][:]
        lon_20hz = data.groups['data_20'].variables['longitude'][:]
        time_20hz = data.groups['data_20'].variables['time'][:]
        
        ice_range_20hz_ku = data.groups['data_20'].groups['ku'].variables['range_ocog'][:]
        ice_sig0_20hz_ku = data.groups['data_20'].groups['ku'].variables['sig0_ocog'][:]
        ice_qual_flag_20hz_ku = data.groups['data_20'].groups['ku'].variables['ocog_qual'][:]
        alt_state_band_status_flag = data.groups['data_01'].groups['ku'].variables['alt_state_band_status_flag'][:]

        # Fill values
        iono_corr_gim_ku_FV = 3.2767
        solid_earth_tide_FV = 3.2767
        pole_tide_FV = 3.2767
        model_dry_tropo_corr_FV = 3.2767
        model_wet_tropo_corr_FV = 3.2767
        lat_20hz_FV = 2147.483647
        
        count = 0

        for p in range(len(alt_20hz)):
            # Check if THIS 20Hz point is in your coordinate range
            if lat_20hz[p] < request.min_lat or lat_20hz[p] > request.max_lat:
                continue
                
            wet_count  = 1 if model_wet_tropo_corr[p] == model_wet_tropo_corr_FV else 0
            dry_count  = 1 if model_dry_tropo_corr[p] == model_dry_tropo_corr_FV else 0
            iono_count = 1 if iono_corr_gim_ku[indx_20hzIn01hz[p]] == iono_corr_gim_ku_FV else 0
            sTide_count= 1 if solid_earth_tide[indx_20hzIn01hz[p]] == solid_earth_tide_FV else 0
            pTide_count= 1 if pole_tide[indx_20hzIn01hz[p]] == pole_tide_FV else 0
            kFlag_count= 1 if alt_state_band_status_flag[indx_20hzIn01hz[p]] != 0 else 0
            lat_count  = 1 if lat_20hz[p] == lat_20hz_FV else 0
            ice_count  = 1 if ice_qual_flag_20hz_ku[p] != 0 else 0

            media_corr = model_dry_tropo_corr[p] + model_wet_tropo_corr[p] + iono_corr_gim_ku[indx_20hzIn01hz[p]] + solid_earth_tide[indx_20hzIn01hz[p]] + pole_tide[indx_20hzIn01hz[p]]  

            mjd_20hz = time_20hz[p]/86400 + 51544
            icehgt_20hz = alt_20hz[p] - media_corr - ice_range_20hz_ku[p] + 0.7
            Flags=dry_count+ wet_count+ iono_count+ sTide_count+pTide_count + kFlag_count+ lat_count+ ice_count
            
            if Flags == 0:
                if np.ma.is_masked(mjd_20hz) or np.ma.is_masked(lon_20hz[p]) or np.ma.is_masked(lat_20hz[p]) or np.ma.is_masked(icehgt_20hz) or np.ma.is_masked(ice_sig0_20hz_ku[p]):
                    pass
                else:
                    txt_file.write('%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%20.6f\t%20.6f\t%20.6f\t%20.6f\t%10.3f\n'%(0, 0, 0, 0, 0, 0, 0, 0, 0, single_name[12:15], mjd_20hz, lon_20hz[p], lat_20hz[p], icehgt_20hz, ice_sig0_20hz_ku[p]))
                    count += 1
        
        data.close()
        return count
        
    except Exception as e:
        print(f"Error processing Jason-3 NetCDF file {nc_file}: {e}")
        return 0

def download_water_occurrence_maps(min_lat: float, max_lat: float, min_lon: float, max_lon: float, data_dir: str):
    """Download Water Occurrence Maps from Google Cloud Storage"""
    wom_dir = f"{data_dir}/Water_Occurrence_Maps"
    os.makedirs(wom_dir, exist_ok=True)
    
    # Calculate required WOM tiles
    min_lat_adj = min_lat + 10
    max_lat_adj = max_lat + 10
    
    # Determine latitude tiles
    if min_lat_adj >= 0 and max_lat_adj >= 0:
        lat_list = [f'{int(i*10)}N' for i in range(int(min_lat_adj // 10), int(max_lat_adj // 10) + 1)]
    elif min_lat_adj < 0 and max_lat_adj < 0:
        lat_list = [f'{int(i*10)}S' for i in range(int(abs(max_lat_adj) // 10) + 1, int(abs(min_lat_adj) // 10) + 2)]
    else:
        positive_lat = [f'{int(i*10)}N' for i in range(0, int(max_lat_adj // 10) + 1)]
        negative_lat = [f'{int(i*10)}S' for i in range(int(abs(min_lat_adj) // 10) + 1, 0, -1)]
        lat_list = positive_lat + negative_lat
    
    # Determine longitude tiles
    if min_lon >= 0 and max_lon >= 0:
        lon_list = [f'{int(i*10)}E' for i in range(int(min_lon // 10), int(max_lon // 10) + 1)]
    elif min_lon < 0 and max_lon < 0:
        lon_list = [f'{int(i*10)}W' for i in range(int(abs(max_lon) // 10) + 1, int(abs(min_lon) // 10) + 2)]
    else:
        positive_lon = [f'{int(i*10)}E' for i in range(0, int(max_lon // 10) + 1)]
        negative_lon = [f'{int(i*10)}W' for i in range(int(abs(min_lon) // 10) + 1, 0, -1)]
        lon_list = positive_lon + negative_lon
    
    # Generate and download WOM files
    wom_files = []
    for lat in lat_list:
        for lon in lon_list:
            wom_file = f'occurrence_{lon}_{lat}v1_4_2021.tif'
            file_path = f"{wom_dir}/{wom_file}"
            
            if os.path.exists(file_path):
                wom_files.append(file_path)
                continue
                
            url = f'https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/{wom_file}'
            
            try:
                print(f"Downloading {wom_file}...")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=4096):
                            if chunk:
                                f.write(chunk)
                    wom_files.append(file_path)
                    print(f"Download of {wom_file} complete")
                else:
                    print(f"Failed to download {wom_file}")
            except Exception as e:
                print(f"Error downloading {wom_file}: {e}")
    
    return wom_files

def merge_wom_data(processing_dir: str, data_dir: str, wom_files: list):
    """Merge Water Occurrence Map tiles"""
    output_file = f"{processing_dir}/WOM.tif"
    
    if len(wom_files) > 1:
        print("Merging WOM tiles...")
        src_files = [rio.open(f) for f in wom_files]
        mosaic, out_trans = merge(src_files)
        
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'compress': 'lzw',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_trans,
            'crs': 'epsg:4326'
        })
        
        with rio.open(output_file, 'w', **out_meta) as tif:
            tif.write(mosaic)
            
        for src in src_files:
            src.close()
        print("WOM merging complete")
    elif len(wom_files) == 1:
        shutil.copy(wom_files[0], output_file)
        print("WOM transfer complete")
    else:
        raise Exception("No Water Occurrence Map files found")
    
    return output_file

def download_srtm_dem(min_lat: float, max_lat: float, min_lon: float, max_lon: float, api_key: str, processing_dir: str):
    """Download SRTM DEM from OpenTopography API"""
    output_file = f"{processing_dir}/SRTM_DEM_WGS84.tif"
    
    api_url = 'https://portal.opentopography.org/API/globaldem'
    params = {
        'demtype': 'SRTMGL1_E',
        'south': min_lat - 0.1,
        'north': max_lat + 0.1,
        'west': min_lon - 0.1,
        'east': max_lon + 0.1,
        'outputFormat': 'GTiff',
        'API_Key': api_key
    }
    
    print("Downloading SRTM DEM...")
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print("SRTM DEM download complete")
        return output_file
    else:
        raise Exception(f"Failed to download SRTM DEM: {response.status_code}")

def iqr_deoutlier(cyc_hgt_profile):
    """IQR-based outlier detection"""
    IQR = iqr(cyc_hgt_profile, nan_policy='omit')   
    return np.logical_and(
        cyc_hgt_profile > np.nanquantile(cyc_hgt_profile, 0.25) - 1.5 * IQR, 
        cyc_hgt_profile < np.nanquantile(cyc_hgt_profile, 0.75) + 1.5*IQR
    ).flatten()

def dem_wo_deoutlier(dem, wo, cyc_lon_profile, cyc_lat_profile, cyc_hgt_profile, hgt_dem_diff_thrd, wo_thrd):
    """DEM and Water Occurrence based outlier detection"""
    lat_lon_buf = 0.01

    min_lon, max_lon = np.nanmin(cyc_lon_profile), np.nanmax(cyc_lon_profile)
    min_lat, max_lat = np.nanmin(cyc_lat_profile), np.nanmax(cyc_lat_profile)

    dem_clip = dem.sel(x=slice(min_lon-lat_lon_buf, max_lon+lat_lon_buf)).sel(y=slice(max_lat+lat_lon_buf, min_lat-lat_lon_buf))
    wo_clip  =  wo.sel(x=slice(min_lon-lat_lon_buf, max_lon+lat_lon_buf)).sel(y=slice(max_lat+lat_lon_buf, min_lat-lat_lon_buf))

    dem_lon_mesh, dem_lat_mesh = np.meshgrid(dem_clip.x.values, dem_clip.y.values)
    wo_lon_mesh,  wo_lat_mesh  = np.meshgrid( wo_clip.x.values,  wo_clip.y.values)
    
    interp_dem = interpolate.griddata(
        (dem_lat_mesh.reshape(-1), dem_lon_mesh.reshape(-1)), 
        dem_clip.values.reshape(-1), 
        (cyc_lat_profile,cyc_lon_profile)
    ).reshape(-1,1)
    
    interp_wo  = interpolate.griddata(
        (wo_lat_mesh.reshape(-1),  wo_lon_mesh.reshape(-1)),  
        wo_clip.values.reshape(-1),  
        (cyc_lat_profile,cyc_lon_profile)
    ).reshape(-1,1)

    index_retain_dem_wo = np.logical_and(
        interp_wo >= wo_thrd, 
        np.abs(cyc_hgt_profile - interp_dem) <= hgt_dem_diff_thrd
    ).flatten()
    
    return index_retain_dem_wo

def kmean_water_cluster(cyc_hgt_profile, cyc_sig_profile, hgt_cyc_range_thrd=5, hgt_cyc_std_thrd=0.3):
    """K-means clustering for water level detection"""
    hgt_cyc_range = np.nanmax(cyc_hgt_profile) - np.nanmin(cyc_hgt_profile)
    
    while hgt_cyc_range > hgt_cyc_range_thrd:
        kmeans_cluster = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(cyc_hgt_profile)                
        cyc_hgt_profile = cyc_hgt_profile[kmeans_cluster==mode(kmeans_cluster, keepdims=True)[0]]
        cyc_sig_profile = cyc_sig_profile[kmeans_cluster==mode(kmeans_cluster, keepdims=True)[0]]
        hgt_cyc_range = np.nanmax(cyc_hgt_profile) - np.nanmin(cyc_hgt_profile)
        
    hgt_cyc_mean = np.nanmean(cyc_hgt_profile)
    hgt_cyc_std = np.nanstd(cyc_hgt_profile)
    
    while hgt_cyc_std > hgt_cyc_std_thrd:
        if np.count_nonzero(~np.isnan(cyc_hgt_profile))==2:
            break
        hgt_cyc_errmean = cyc_hgt_profile - hgt_cyc_mean
        cyc_hgt_profile = cyc_hgt_profile[np.abs(hgt_cyc_errmean)!=np.nanmax(np.abs(hgt_cyc_errmean))]
        cyc_sig_profile = cyc_sig_profile[np.abs(hgt_cyc_errmean)!=np.nanmax(np.abs(hgt_cyc_errmean))]
        hgt_cyc_std = np.nanstd(cyc_hgt_profile)
        hgt_cyc_mean = np.nanmean(cyc_hgt_profile)
        
    return cyc_hgt_profile, cyc_sig_profile

def generate_time_series(input_data, index_retain, series, pass_num, min_lat, max_lat, data_dir, method_name):
    """Generate time series from filtered data with geoid correction"""
    # Load geoid data
    geoid_file = os.path.join(data_dir, 'geoidegm2008grid.mat')
    try:
        geoid_data = io.loadmat(geoid_file)
        lonbp = geoid_data['lonbp']
        latbp = geoid_data['latbp']
        grid = geoid_data['grid']
        ip = RegularGridInterpolator(points=(latbp.flatten(),lonbp.flatten()), values=grid, bounds_error=False, fill_value=np.nan)
        print(f"Geoid data loaded successfully for {method_name} method")
    except Exception as e:
        print(f"Warning: Could not load geoid data for {method_name}: {e}")
        ip = None
    
    # Apply filtering
    filtered_data = input_data[index_retain,:]
    print(f"{method_name} method: {len(filtered_data)} points after filtering from {len(input_data)} total")
    
    if len(filtered_data) == 0:
        print(f"Warning: No data points remain after {method_name} filtering")
        return np.array([]), ip
    
    cycno_list = filtered_data[:,0]
    uniq_cycno = np.unique(cycno_list)
    
    timeseries_report = np.empty((len(uniq_cycno),8))
    timeseries_report[:] = np.nan
    
    for ct_cyc, cycno in enumerate(uniq_cycno):
        index_cyc = cycno_list==cycno
        cyc_data = filtered_data[index_cyc,:]
        
        mjd = cyc_data[:,1].reshape(-1,1)
        lon = cyc_data[:,2].reshape(-1,1)
        lat = cyc_data[:,3].reshape(-1,1)
        hgt = cyc_data[:,4].reshape(-1,1)
        sig = cyc_data[:,5].reshape(-1,1)    
        
        hgt, sig = kmean_water_cluster(hgt, sig)
        
        cyc_time = (np.nanmean(mjd)+2108-50000)/365.25 +1990
        cyc_lon = np.nanmean(lon)
        cyc_lat = np.nanmean(lat)
        cyc_hgt = np.nanmean(hgt)
        cyc_std_hgt = np.nanstd(hgt)      
        cyc_sig = np.nanmean(sig)
        cyc_retain_rate = hgt.shape[0] / cyc_data.shape[0]
        
        timeseries_report[ct_cyc,:] = [cycno, cyc_time, cyc_lon, cyc_lat, cyc_hgt, cyc_std_hgt, cyc_sig, cyc_retain_rate]  
    
    # Final IQR filtering
    index_retain_iqr2 = iqr_deoutlier(timeseries_report[:,4])  
    final_series = timeseries_report[index_retain_iqr2,:]
    
    print(f"{method_name} method: Generated {len(final_series)} time series points")
    return final_series, ip

class AltimetryRequest(BaseModel):
    aviso_username: str
    aviso_password: str
    opentopography_api_key: str
    series: int  # 1=both, 2=Jason-2, 3=Jason-3
    pass_number: int
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    # Jason-2 parameters
    start_cycle_2: Optional[int] = None
    end_cycle_2: Optional[int] = None
    # Jason-3 parameters  
    start_cycle_3: Optional[int] = None
    end_cycle_3: Optional[int] = None
    # Processing options
    outlier_removal_method: str = "all"  # Options: "iqr", "dem_wo", "none", "all"
    hgt_dem_diff_thrd: Optional[int] = 5
    wo_thrd: Optional[int] = 50
    hampel_window: Optional[int] = 0
    hampel_sigma: Optional[float] = 3.0

    @field_validator('outlier_removal_method')
    @classmethod
    def validate_outlier_method(cls, v):
        valid_methods = ["iqr", "dem_wo", "none", "all"]
        if v not in valid_methods:
            raise ValueError(f'outlier_removal_method must be one of: {valid_methods}')
        return v

    @field_validator('start_cycle_2', 'end_cycle_2', 'start_cycle_3', 'end_cycle_3')
    @classmethod
    def validate_cycles(cls, v, info):
        data = info.data if hasattr(info, 'data') else {}
        series = data.get('series')
        field_name = info.field_name
        
        if series == 1:  # Both series
            if field_name in ['start_cycle_2', 'end_cycle_2', 'start_cycle_3', 'end_cycle_3']:
                if v is None:
                    raise ValueError(f'{field_name} required when series=1 (both)')
        elif series == 2:  # Jason-2 only
            if field_name in ['start_cycle_2', 'end_cycle_2'] and v is None:
                raise ValueError(f'{field_name} required when series=2')
        elif series == 3:  # Jason-3 only
            if field_name in ['start_cycle_3', 'end_cycle_3'] and v is None:
                raise ValueError(f'{field_name} required when series=3')
        return v

def process_jason_data(job_id: str, request: AltimetryRequest):
    """Complete processing pipeline with all outlier removal options"""
    try:
        update_job_status(job_id, "running", 0.05, "Starting data download")
        
        # Create directory structure
        output_dir = f"results_{job_id}"
        data_dir = f"{output_dir}/data"
        processing_dir = f"{output_dir}/DEM_and_WO_processing"
        timeseries_dir = f"{output_dir}/timeseries_output"
        
        for dir_path in [data_dir, processing_dir, timeseries_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Copy geoid file to data directory
        if os.path.exists('geoidegm2008grid.mat'):
            shutil.copy('geoidegm2008grid.mat', data_dir)
        
        # Download Jason data
        ftp = FTP('ftp-access.aviso.altimetry.fr')
        ftp.login(user=request.aviso_username, passwd=request.aviso_password)
        
        update_job_status(job_id, "running", 0.1, "Downloading satellite data")
        
        downloaded_files = []
        jason2_files = []
        jason3_files = []
        
        if request.series in [1, 2]:  # Jason-2
            jason2_files = download_series_data(ftp, 2, request.pass_number, request.start_cycle_2, request.end_cycle_2, data_dir)
            downloaded_files.extend(jason2_files)
        
        if request.series in [1, 3]:  # Jason-3
            jason3_files = download_series_data(ftp, 3, request.pass_number, request.start_cycle_3, request.end_cycle_3, data_dir)
            downloaded_files.extend(jason3_files)
            
        ftp.quit()
        
        if not downloaded_files:
            update_job_status(job_id, "failed", 0.0, "No data files were downloaded")
            return {"status": "failed", "error": "No data files available"}
        
        # Process NetCDF files to extract data
        update_job_status(job_id, "running", 0.3, "Extracting altimetry data")
        
        extracted_file = f"{output_dir}/extracted_data.txt"
        total_extracted = 0
        
        with open(extracted_file, 'w') as txt_file:
            for nc_file in jason2_files:
                extracted_count = process_netcdf_file_jason2(nc_file, txt_file, request)
                total_extracted += extracted_count
            
            for nc_file in jason3_files:
                extracted_count = process_netcdf_file_jason3(nc_file, txt_file, request)
                total_extracted += extracted_count
        
        if total_extracted == 0:
            update_job_status(job_id, "failed", 0.0, "No valid data points extracted")
            return {"status": "failed", "error": "No valid data points found in coordinate range"}
        
        # Load extracted data
        input_data = np.loadtxt(extracted_file)[:,9:]
        
        # Filter by latitude range
        lat_all = input_data[:,3] 
        index_lat_range = np.logical_and(lat_all > request.min_lat, lat_all < request.max_lat)                
        input_data = input_data[index_lat_range,:]
        
        # Prepare data for outlier removal
        lon_all = input_data[:,2].reshape(-1,1)
        lat_all = input_data[:,3].reshape(-1,1)
        hgt_all = input_data[:,4].reshape(-1,1)
        
        results = {"output_files": {}}
        
        # Apply outlier removal based on user selection
        if request.outlier_removal_method in ["dem_wo", "all"]:
            update_job_status(job_id, "running", 0.5, "Downloading DEM and Water Occurrence data")
            
            # Download Water Occurrence Maps
            wom_files = download_water_occurrence_maps(request.min_lat, request.max_lat, request.min_lon, request.max_lon, data_dir)
            wom_merged = merge_wom_data(processing_dir, data_dir, wom_files)
            
            # Download SRTM DEM
            dem_file = download_srtm_dem(request.min_lat, request.max_lat, request.min_lon, request.max_lon, request.opentopography_api_key, processing_dir)
            
            # Load DEM and WOM for processing
            dem = rioxarray.open_rasterio(dem_file).squeeze().drop('band')
            dem.x.values[dem.x.values < 0] = dem.x.values[dem.x.values < 0] + 360
            
            wo = rioxarray.open_rasterio(wom_merged).squeeze().drop('band')
            wo.x.values[wo.x.values < 0] = wo.x.values[wo.x.values<0] + 360
            
            index_retain_dem_wo = dem_wo_deoutlier(dem, wo, lon_all, lat_all, hgt_all, request.hgt_dem_diff_thrd, request.wo_thrd)
        
        update_job_status(job_id, "running", 0.7, "Generating time series")
        
        # Generate time series for each requested method
        if request.outlier_removal_method in ["iqr", "all"]:
            index_retain_iqr = iqr_deoutlier(hgt_all)
            final_series_iqr, _ = generate_time_series(input_data, index_retain_iqr, request.series, request.pass_number, request.min_lat, request.max_lat, data_dir, "IQR")
            
            if len(final_series_iqr) > 0:
                iqr_file = f"{timeseries_dir}/timeseries_IQR_method.txt"
                np.savetxt(iqr_file, final_series_iqr, fmt='%.6f')
                results["output_files"]["iqr_timeseries"] = iqr_file
                results["iqr_time_series_points"] = len(final_series_iqr)
                
                # Create IQR plot
                plt.figure(figsize=(12, 8))
                sorted_indices = np.argsort(final_series_iqr[:,1])
                sorted_years = final_series_iqr[sorted_indices,1] 
                sorted_elevations = final_series_iqr[sorted_indices,4]
                plt.plot(sorted_years, sorted_elevations, 'b-o', label='IQR Method')
                plt.xlabel('Year')
                plt.ylabel('Water Elevation (m)')
                plt.title(f'Jason-{request.series} Time Series - Pass {request.pass_number} (IQR Method)')
                plt.grid(True)
                plt.legend()
                iqr_plot = f"{timeseries_dir}/timeseries_IQR_plot.png"
                plt.savefig(iqr_plot, dpi=300, bbox_inches='tight')
                plt.close()
                results["output_files"]["iqr_plot"] = iqr_plot
        
        if request.outlier_removal_method in ["dem_wo", "all"]:
            final_series_dem_wo, _ = generate_time_series(input_data, index_retain_dem_wo, request.series, request.pass_number, request.min_lat, request.max_lat, data_dir, "DEM_WO")
            
            if len(final_series_dem_wo) > 0:
                dem_wo_file = f"{timeseries_dir}/timeseries_SRTM_method.txt"
                np.savetxt(dem_wo_file, final_series_dem_wo, fmt='%.6f')
                results["output_files"]["srtm_timeseries"] = dem_wo_file
                results["srtm_time_series_points"] = len(final_series_dem_wo)
                
                # Create SRTM plot
                plt.figure(figsize=(12, 8))
                sorted_indices = np.argsort(final_series_dem_wo[:,1])
                sorted_years = final_series_dem_wo[sorted_indices,1] 
                sorted_elevations = final_series_dem_wo[sorted_indices,4]
                plt.plot(sorted_years, sorted_elevations, 'r-o', label='SRTM Method')
                plt.xlabel('Year')
                plt.ylabel('Water Elevation (m)')
                plt.title(f'Jason-{request.series} Time Series - Pass {request.pass_number} (SRTM Method)')
                plt.grid(True)
                plt.legend()
                srtm_plot = f"{timeseries_dir}/timeseries_SRTM_plot.png"
                plt.savefig(srtm_plot, dpi=300, bbox_inches='tight')
                plt.close()
                results["output_files"]["srtm_plot"] = srtm_plot
        
        if request.outlier_removal_method in ["none", "all"]:
            index_retain_none = np.ones(len(input_data), dtype=bool)  # Keep all data
            final_series_none, _ = generate_time_series(input_data, index_retain_none, request.series, request.pass_number, request.min_lat, request.max_lat, data_dir, "No_Filter")
            
            if len(final_series_none) > 0:
                none_file = f"{timeseries_dir}/timeseries_NoFilter_method.txt"
                np.savetxt(none_file, final_series_none, fmt='%.6f')
                results["output_files"]["nofilter_timeseries"] = none_file
                results["nofilter_time_series_points"] = len(final_series_none)
                
                # Create No Filter plot
                plt.figure(figsize=(12, 8))
                sorted_indices = np.argsort(final_series_none[:,1])
                sorted_years = final_series_none[sorted_indices,1] 
                sorted_elevations = final_series_none[sorted_indices,4]
                plt.plot(sorted_years, sorted_elevations, 'g-o', label='No Filter')
                plt.xlabel('Year')
                plt.ylabel('Water Elevation (m)')
                plt.title(f'Jason-{request.series} Time Series - Pass {request.pass_number} (No Filter)')
                plt.grid(True)
                plt.legend()
                none_plot = f"{timeseries_dir}/timeseries_NoFilter_plot.png"
                plt.savefig(none_plot, dpi=300, bbox_inches='tight')
                plt.close()
                results["output_files"]["nofilter_plot"] = none_plot
        
        update_job_status(job_id, "completed", 1.0, f"Processing complete using {request.outlier_removal_method} method(s)")
        
        results.update({
            "status": "completed", 
            "output_dir": output_dir,
            "files_downloaded": len(downloaded_files),
            "data_points_extracted": total_extracted,
            "outlier_removal_method": request.outlier_removal_method,
            "extracted_data": extracted_file
        })
        
        return results
        
    except Exception as e:
        update_job_status(job_id, "failed", 0.0, f"Error: {str(e)}")
        return {"status": "failed", "error": str(e)}

# API Routes
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# at the top
import mimetypes
import os

@app.get("/api/altimetry/download/{job_id}/{file_type}")
def download_file(job_id: str, file_type: str):
    if job_id not in job_status:
        raise HTTPException(404, "Job not found")

    if job_status[job_id]["status"] != "completed":
        raise HTTPException(400, "Job not completed")

    timeseries_dir = f"results_{job_id}/timeseries_output"

    file_map = {
        "iqr_data":      f"{timeseries_dir}/timeseries_IQR_method.txt",
        "srtm_data":     f"{timeseries_dir}/timeseries_SRTM_method.txt",
        "nofilter_data": f"{timeseries_dir}/timeseries_NoFilter_method.txt",
        "iqr_plot":      f"{timeseries_dir}/timeseries_IQR_plot.png",
        "srtm_plot":     f"{timeseries_dir}/timeseries_SRTM_plot.png",
        "nofilter_plot": f"{timeseries_dir}/timeseries_NoFilter_plot.png",
    }

    if file_type not in file_map:
        raise HTTPException(404, "File type not found")

    file_path = file_map[file_type]
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    # Use the real filename and correct media type
    filename = os.path.basename(file_path)  # e.g., timeseries_IQR_method.txt
    media_type, _ = mimetypes.guess_type(filename)
    media_type = media_type or "application/octet-stream"

    return FileResponse(file_path, media_type=media_type, filename=filename)





@app.post("/process")
def start_processing(request: AltimetryRequest, background_tasks: BackgroundTasks):
    # Generate job ID
    job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Test authentication first
    try:
        ftp = FTP('ftp-access.aviso.altimetry.fr')
        ftp.login(user=request.aviso_username, passwd=request.aviso_password)
        ftp.quit()
    except:
        return {"status": "failed", "message": "AVISO authentication failed"}
    
    # Initialize job status
    update_job_status(job_id, "queued", 0.0, "Job queued for processing")
    
    # Start background processing
    background_tasks.add_task(process_jason_data, job_id, request)
    
    return {
        "status": "started",
        "job_id": job_id,
        "message": "Processing started in background",
        "parameters": request.model_dump()
    }

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_status:
        return {"error": "Job not found"}
    return job_status[job_id]

@app.get("/results/{job_id}")
def get_results(job_id: str):
    if job_id not in job_status:
        return {"error": "Job not found"}
    
    if job_status[job_id]["status"] != "completed":
        return {"error": "Job not completed"}
    
    # List result files
    output_dir = f"results_{job_id}"
    if os.path.exists(output_dir):
        files = []
        for root, dirs, filenames in os.walk(output_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return {"files": files, "output_directory": output_dir}
    else:
        return {"error": "Results not found"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
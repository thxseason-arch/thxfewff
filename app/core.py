import os
import datetime
import io
import s3fs
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio

GLM_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'glm_cache')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(GLM_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_cirrus_colormap_soft():
    colors = [
        (0.00, '#0c0f12'), (0.03, '#261f18'), (0.08, '#6e5437'), (0.16, '#c49a5c'),
        (0.28, '#e8cf9b'), (0.42, '#cce0f5'), (0.65, '#ffffff'), (1.00, '#ffffff')
    ]
    return mcolors.LinearSegmentedColormap.from_list('cirrus_soft_ref', colors)


def create_weathernerds_ir():
    hex_colors = [
        '8E2590','99379A','99379C','A14DA1','AB60AB','B475B4','BE8ABD','C59FC4','CEB1CE','D6C4D7','D2C9D2',
        'CECECE','B4B4B4','999999','828282','6B6B6B','525252','393939','212121','090909','210405','3A0000',
        '510000','690003','860000','9D0000','BB0100','D80102','EB0A00','FF1603','FF3303','FF4D02','FF6600',
        'FF7F02','FF9900','FEB200','FFCB00','FEE214','EFF107','DBFF00','C3FF0B','A6FE06','92FF04','7BFF01',
        '62FF05','4EFD12','30FF00','2AFE14','0DEF0F','00DE0B','00CA18','00B622','01A02D','008A33','008A33',
        '06733F','005B49','004554','002E5E','011968','000470','002182','023C94','0056A1','0470B0','0688BC','00A1CB',
        '07BADA','00D4E6','17E1E9','40ECED','79D5D5','ADADAD','AAAAAA','ABAAAA','A8A8A8','A5A5A5','A2A2A2','9F9F9F',
        '9C9C9C','999999','969696','939393','909090','8D8D8D','8A8A8A','878787','848484','818181','7E7E7E','7B7B7B',
        '787878','757575','727272','6F6F6F','6C6C6C','696969','666666','636363','606060','5D5D5D','5A5A5A','575757',
        '545454','515151','4E4E4E','4B4B4B','484848','454545','424242','3F3F3F','3C3C3C','393939','363636','333333',
        '303030','2D2D2D','2A2A2A','272727','242424','212121','1E1E1E','1B1B1B','181818','151515','121212','0F0F0F',
        '0C0C0C','090909','060606','030303','000000'
    ]
    rgb_arr = np.array([mcolors.to_rgb(f'#{h}') for h in hex_colors], dtype=float)
    cmap = ListedColormap(rgb_arr, name='WeatherNerds_IR')
    norm = BoundaryNorm(np.arange(-90.5, 41.5, 1), cmap.N)
    return cmap, norm


def fetch_goes_data(fs, sat, band, date_obj, hour_utc, minute_utc):
    doy = date_obj.strftime('%j')
    year = date_obj.strftime('%Y')
    path = f'{sat}/ABI-L2-CMIPF/{year}/{doy}/{hour_utc:02d}/'
    files = fs.ls(path)
    target_files = [f for f in files if f'M6{band}' in f or f'M3{band}' in f or f'M4{band}' in f]
    if not target_files:
        raise FileNotFoundError(f'Nenhum arquivo ABI encontrado em: s3://{path}')
    best_file = target_files[0]
    min_diff = 999
    for f in target_files:
        try:
            fn = f.split('/')[-1]
            time_str = fn.split('_s')[1][:11]
            f_min = int(time_str[9:11])
            diff = abs(f_min - minute_utc)
            if diff < min_diff:
                min_diff = diff
                best_file = f
        except Exception:
            continue
    return best_file


def _open_dataset_robust(path):
    last_error = None
    for engine in ('h5netcdf', 'netcdf4', None):
        try:
            if engine is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=engine)
        except (ImportError, ModuleNotFoundError, ValueError, OSError) as exc:
            last_error = exc
    raise last_error


def fetch_glm_data(fs, sat, date_obj, hour_utc, minute_utc, accumulation_minutes=15):
    target_dt = datetime.datetime.combine(date_obj, datetime.time(hour_utc, minute_utc))
    start_dt = target_dt - datetime.timedelta(minutes=accumulation_minutes)
    hours_to_check = {(start_dt.date(), start_dt.hour), (target_dt.date(), target_dt.hour)}
    all_files = []
    for dt_date, dt_hour in hours_to_check:
        doy = dt_date.strftime('%j')
        year = dt_date.strftime('%Y')
        path = f'{sat}/GLM-L2-LCFA/{year}/{doy}/{dt_hour:02d}/'
        try:
            files = fs.ls(path)
            for f in files:
                try:
                    fn = f.split('/')[-1]
                    time_str = fn.split('_s')[1]
                    f_year = int(time_str[:4]); f_doy = int(time_str[4:7]); f_h = int(time_str[7:9]); f_m = int(time_str[9:11]); f_s = int(time_str[11:13])
                    file_dt = datetime.datetime(f_year, 1, 1) + datetime.timedelta(days=f_doy - 1, hours=f_h, minutes=f_m, seconds=f_s)
                    if start_dt <= file_dt <= target_dt:
                        all_files.append(f)
                except Exception:
                    continue
        except Exception:
            continue
    if not all_files:
        return np.array([]), np.array([])
    max_samples = 8
    if len(all_files) > max_samples:
        step = len(all_files) // max_samples
        all_files = all_files[::step]
    lons, lats = [], []
    for f_s3 in all_files:
        fn = f_s3.split('/')[-1]
        local_path = os.path.join(GLM_CACHE_DIR, fn)
        if not os.path.exists(local_path):
            try:
                fs.get(f_s3, local_path)
            except Exception:
                continue
        try:
            with _open_dataset_robust(local_path) as ds_glm:
                if 'flash_lon' in ds_glm and 'flash_lat' in ds_glm:
                    lons.extend(ds_glm['flash_lon'].values)
                    lats.extend(ds_glm['flash_lat'].values)
        except Exception:
            continue
    return np.array(lons), np.array(lats)


def process_and_render_scene(fs, s3_file, bbox, date_obj, hour_utc, minute_utc, sat, show_glm=False, interpolation='nearest', band='C13', output_filename='output.png', info_title='', decim=1):
    local_nc = os.path.join(OUTPUT_DIR, f'_temp_{os.getpid()}.nc')
    if os.path.exists(local_nc):
        os.remove(local_nc)
    fs.get(s3_file, local_nc)
    try:
        with _open_dataset_robust(local_nc) as ds:
            proj_info = ds.goes_imager_projection
            h = float(proj_info.perspective_point_height)
            lon_0 = float(proj_info.longitude_of_projection_origin)
            sweep = str(proj_info.sweep_angle_axis)
            p = ccrs.Geostationary(central_longitude=lon_0, satellite_height=h, sweep_axis=sweep)
            pc = ccrs.PlateCarree()
            x1_m, y1_m = p.transform_point(bbox['min_lon'], bbox['min_lat'], pc)
            x2_m, y2_m = p.transform_point(bbox['max_lon'], bbox['max_lat'], pc)
            x_min_m, x_max_m = sorted([x1_m, x2_m]); y_min_m, y_max_m = sorted([y1_m, y2_m])
            width_m = x_max_m - x_min_m; height_m = y_max_m - y_min_m
            aspect = width_m / height_m if height_m > 0 else 1.0
            fig_w = 8.0; fig_h = max(3.0, fig_w / aspect)
            x_left_rad, x_right_rad = x_min_m / h, x_max_m / h
            y_bottom_rad, y_top_rad = y_min_m / h, y_max_m / h
            y_slice = slice(y_top_rad, y_bottom_rad) if ds.y[0] > ds.y[-1] else slice(y_bottom_rad, y_top_rad)
            x_slice = slice(x_left_rad, x_right_rad) if ds.x[0] < ds.x[-1] else slice(x_right_rad, x_left_rad)
            data_slice = ds['CMI'].sel(x=x_slice, y=y_slice)
            if data_slice.size == 0:
                data_slice = ds['CMI']
            cmi_data = np.nan_to_num(data_slice.values[::decim, ::decim] if decim > 1 else data_slice.values, nan=0.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140, subplot_kw={'projection': p}, facecolor='#ffffff')
            ax.set_facecolor('#ffffff'); ax.set_extent([x_min_m, x_max_m, y_min_m, y_max_m], crs=p)
            if band == 'C04':
                cmap = create_cirrus_colormap_soft(); norm = mcolors.PowerNorm(gamma=0.75, vmin=0.001, vmax=0.32)
            elif band == 'C13':
                cmi_data = cmi_data - 273.15; cmap, norm = create_weathernerds_ir()
            elif band in ['C08','C09','C10']:
                cmap = 'YlGnBu_r'; norm = mcolors.Normalize(vmin=195.0, vmax=265.0)
            elif band == 'C07':
                cmap = 'hot'; norm = mcolors.Normalize(vmin=210.0, vmax=340.0)
            elif band in ['C01','C02','C03','C05','C06']:
                cmap = 'gray'; norm = mcolors.Normalize(vmin=0.0, vmax=0.85)
            else:
                cmap = 'gray_r'; norm = mcolors.Normalize(vmin=190.0, vmax=300.0)
            ax.imshow(cmi_data, origin='upper', extent=[x_min_m, x_max_m, y_min_m, y_max_m], transform=p, cmap=cmap, norm=norm, interpolation=interpolation)
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.6, zorder=5)
            ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.3, linestyle=':', zorder=5)
            if show_glm:
                glm_lons, glm_lats = fetch_glm_data(fs, sat, date_obj, hour_utc, minute_utc, accumulation_minutes=15)
                if len(glm_lons) > 0:
                    mask = (glm_lons >= bbox['min_lon']) & (glm_lons <= bbox['max_lon']) & (glm_lats >= bbox['min_lat']) & (glm_lats <= bbox['max_lat'])
                    valid_lons = glm_lons[mask]; valid_lats = glm_lats[mask]
                    if len(valid_lons) > 0:
                        span_lon = bbox['max_lon'] - bbox['min_lon']; dynamic_size = max(12, min(80, int(180 / max(span_lon, 0.2))))
                        ax.scatter(valid_lons, valid_lats, transform=pc, color='white', s=dynamic_size, marker='.', linewidths=0, alpha=0.9, zorder=10)
            if info_title:
                ax.set_title(info_title, color='black', fontsize=9, fontweight='bold', pad=6, backgroundcolor='#ffffffCC')
            ax.text(0.96, 0.03, 'Muryylo plots', transform=ax.transAxes, fontsize=10, fontweight='bold', color='#111111', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#222222', alpha=0.9, linewidth=1.0))
            ax.axis('off'); plt.subplots_adjust(left=0, right=1, top=0.92, bottom=0)
            out = os.path.join(OUTPUT_DIR, os.path.basename(output_filename))
            buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02, facecolor='#ffffff'); buf.seek(0)
            with open(out, 'wb') as f_out: f_out.write(buf.read())
            plt.close(fig)
            return out
    finally:
        if os.path.exists(local_nc): os.remove(local_nc)


def make_fs():
    return s3fs.S3FileSystem(anon=True)

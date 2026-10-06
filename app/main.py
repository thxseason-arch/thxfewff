from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from datetime import date, datetime, timedelta
import os, uuid
import imageio.v2 as imageio
from .core import make_fs, fetch_goes_data, process_and_render_scene

app = FastAPI(title='GOES Explorer 98 API', version='2.0.1')

# O frontend pode ser hospedado separadamente (por exemplo, Render Static Site).
# Em produção, o domínio do frontend pode ser restringido aqui.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SATELLITES = {'noaa-goes16':'GOES-16 (East)', 'noaa-goes19':'GOES-19 (East Operacional)'}
BANDS = [f'C{i:02d}' for i in range(1,17)]
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

class ImageRequest(BaseModel):
    satellite: str = 'noaa-goes16'
    band: str = 'C13'
    date: date
    hour: int = Field(18, ge=0, le=23)
    minute: int = Field(0, ge=0, le=59)
    min_lon: float = -60
    max_lon: float = -40
    min_lat: float = -25
    max_lat: float = -5
    glm: bool = True
    interpolation: str = 'nearest'
    decim: int = Field(1, ge=1, le=8)

class GifRequest(ImageRequest):
    duration_hours: int = Field(3, ge=1, le=12)
    interval_minutes: int = Field(15, ge=5, le=60)
    fps: int = Field(2, ge=1, le=10)
    # Compatibilidade com o frontend Muryllo plotts.
    duration_minutes: int | None = Field(None, ge=5, le=1440)

@app.get('/')
def root(): return {'app':'GOES Explorer 98','status':'online','version':'2.0.0'}
@app.get('/status')
def status(): return {'status':'OK','message':'Backend real do núcleo GOES carregado'}
@app.get('/satellites')
def satellites(): return SATELLITES
@app.get('/bands')
def bands(): return BANDS

@app.post('/generate')
def generate(req: ImageRequest):
    if req.satellite not in SATELLITES: raise HTTPException(400,'Satélite inválido')
    if req.band not in BANDS: raise HTTPException(400,'Banda inválida')
    if req.min_lon >= req.max_lon or req.min_lat >= req.max_lat: raise HTTPException(400,'Bounding box inválido')
    try:
        fs=make_fs(); s3=fetch_goes_data(fs, req.satellite, req.band, req.date, req.hour, req.minute)
        name=f'goes_{uuid.uuid4().hex}.png'
        path=process_and_render_scene(fs,s3,{'min_lon':req.min_lon,'max_lon':req.max_lon,'min_lat':req.min_lat,'max_lat':req.max_lat},req.date,req.hour,req.minute,req.satellite,req.glm,req.interpolation,req.band,name, f'{SATELLITES[req.satellite]} | {req.band} | {req.date} {req.hour:02d}:{req.minute:02d} UTC', req.decim)
        return FileResponse(path, media_type='image/png', filename=name)
    except Exception as e:
        raise HTTPException(500, f'{type(e).__name__}: {e}')

@app.post('/generate-gif')
def generate_gif(req: GifRequest):
    try:
        fs=make_fs(); end=datetime.combine(req.date, datetime.min.time()).replace(hour=req.hour,minute=req.minute)
        total_minutes = req.duration_minutes if req.duration_minutes is not None else req.duration_hours * 60
        start=end-timedelta(minutes=total_minutes)
        frames=[]; temp=[]; cur=start
        while cur <= end:
            try:
                s3=fetch_goes_data(fs,req.satellite,req.band,cur.date(),cur.hour,cur.minute)
                name=f'frame_{uuid.uuid4().hex}.png'
                path=process_and_render_scene(fs,s3,{'min_lon':req.min_lon,'max_lon':req.max_lon,'min_lat':req.min_lat,'max_lat':req.max_lat},cur.date(),cur.hour,cur.minute,req.satellite,req.glm,req.interpolation,req.band,name,'',req.decim)
                frames.append(imageio.imread(path)); temp.append(path)
            except Exception:
                pass
            cur += timedelta(minutes=req.interval_minutes)
        if not frames: raise RuntimeError('Nenhum quadro pôde ser gerado')
        name=f'goes_{uuid.uuid4().hex}.gif'; path=os.path.join(OUT,name)
        imageio.mimsave(path,frames,duration=1/req.fps,loop=0)
        for p in temp:
            try: os.remove(p)
            except OSError: pass
        return FileResponse(path,media_type='image/gif',filename=name)
    except Exception as e: raise HTTPException(500,f'{type(e).__name__}: {e}')

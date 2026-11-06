from pathlib import Path, PureWindowsPath
import json, tarfile, hashlib, sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from maichart.training.dataset_catalog import _source_asset
out=root/'dist/colab_v2_full'
archive=out/'composer_v2_full.tar.gz'
if archive.exists(): raise FileExistsError(archive)
files={}
for name in ('train','dev','final'):
 p=out/'data_split'/f'{name}_manifest.json'
 data=json.loads(p.read_text(encoding='utf-8'))
 files['splits/'+p.name]=p
 for entry in data['songs']:
  for key in ('audio_path','maidata_path'):
   source=_source_asset(entry[key],p,data)
   if not source.is_file(): raise FileNotFoundError(source)
   win=PureWindowsPath(str(source))
   files['materials/'+win.drive.rstrip(':')+'/'+('/'.join(win.parts[1:]))]=source
   structure=source.with_name('music_structure.json')
   if structure.is_file():
    w=PureWindowsPath(str(structure));files['materials/'+w.drive.rstrip(':')+'/'+('/'.join(w.parts[1:]))]=structure
annotation=Path('E:/全量官谱/quality_annotations.json')
if annotation.is_file(): files['materials/E/全量官谱/quality_annotations.json']=annotation
for p in (root/'src').rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):
  files['project/'+p.relative_to(root).as_posix()]=p
for name in ('README.md','pyproject.toml','THIRD_PARTY_NOTICES.md'):
 files['project/'+name]=root/name
print('files',len(files),'raw GiB',round(sum(p.stat().st_size for p in files.values())/2**30,2),flush=True)
with tarfile.open(str(archive)+'.partial','w:gz',compresslevel=1) as t:
 for i,(name,p) in enumerate(sorted(files.items()),1):
  t.add(p,arcname=name,recursive=False)
  if i%500==0: print('packed',i,flush=True)
Path(str(archive)+'.partial').replace(archive)
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
Path(str(archive)+'.sha256').write_text(h.hexdigest()+'  '+archive.name+'\n')
print('ready',archive,archive.stat().st_size,flush=True)

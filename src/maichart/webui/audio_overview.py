"""Cached CPU-only waveform/spectrum data for the v2 curve editor."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np

SCHEMA = 'composer-v2-editor-media-v1'


def overview(wave, rate):
    if not len(wave) or rate<=0:raise ValueError('音频没有可预览的采样')
    duration=len(wave)/rate
    hop=max(1,round(rate*.02),int(np.ceil(len(wave)/120000)))
    low=[];high=[]
    for start in range(0,len(wave),hop*2048):
        block=np.asarray(wave[start:start+hop*2048])
        full=len(block)//hop
        if full:
            frames=block[:full*hop].reshape(full,hop)
            low.extend(frames.min(1).round(5).tolist());high.extend(frames.max(1).round(5).tolist())
        if len(block)%hop:
            tail=block[full*hop:];low.append(float(tail.min()));high.append(float(tail.max()))
    fft=max(512,min(8192,2**int(np.ceil(np.log2(rate*.04)))))
    stride=max(1,round(rate*.04),int(np.ceil(len(wave)/16000)))
    centers=np.arange(0,len(wave),stride)
    frequencies=np.geomspace(32.,max(33.,min(16000.,rate/2)),49)
    bins=np.clip(np.floor(frequencies*fft/rate).astype(int),0,fft//2)
    window=np.hanning(fft).astype(np.float32);offset=np.arange(fft)-fft//2
    values=[]
    for start in range(0,len(centers),128):
        indices=centers[start:start+128,None]+offset
        valid=(indices>=0)&(indices<len(wave))
        frames=np.asarray(wave[np.clip(indices,0,len(wave)-1)])*valid*window
        spectrum=np.abs(np.fft.rfft(frames,axis=1))/max(1.,window.sum())
        bands=np.stack([spectrum[:,a:max(a+1,b)].max(1) for a,b in zip(bins[:-1],bins[1:])],1)
        values.extend(np.clip((20*np.log10(np.maximum(bands,1e-8))+80)/80*255,0,255).astype(np.uint8).tolist())
    return dict(schema=SCHEMA,duration_sec=duration,
        waveform=dict(step_sec=hop/rate,minimum=low,maximum=high),
        spectrogram=dict(step_sec=stride/rate,frequencies_hz=frequencies.tolist(),values=values,range_db=[-80,0]))


def prepare(audio,cache_dir):
    from maichart.form.rational_audio import _source_digest
    from maichart.form.waveform_cache import mapped_waveform
    audio=Path(audio).resolve();cache=Path(cache_dir).resolve()
    digest=_source_digest(str(audio),str(cache))
    directory=cache/'editor_media_v1';directory.mkdir(parents=True,exist_ok=True)
    cached=directory/(digest+'.json')
    if cached.is_file():return cached
    wave,rate=mapped_waveform(audio,sha256=digest,cache_dir=cache)
    result=overview(wave,rate);result['audio_sha256']=digest
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',suffix='.tmp',dir=directory,delete=False) as stream:
        temporary=Path(stream.name);json.dump(result,stream,ensure_ascii=False,separators=(',',':'))
    temporary.replace(cached)
    return cached


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cache-dir',default='cache/chart_composer_v2')
    args=parser.parse_args(argv)
    cached=prepare(args.audio,args.cache_dir)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(cached,output)
    print('composer_v2_media '+json.dumps(dict(output=str(output))),flush=True)


if __name__=='__main__':main()

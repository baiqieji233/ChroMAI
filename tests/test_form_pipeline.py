"""Small real CPU updates, independent checkpoint, exact arrangement and old path."""
from fractions import Fraction as F
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint,EventTempo
from maichart.generation.event_audio import EventAudio,ACOUSTIC_DIM
from maichart.form.codec import RationalCodec
from maichart.form.timing import build_spans,MeterChange
from maichart.form.data import organize_song,input_example,collate_form,form_audio,FormDataset


def fixture_audio(seconds=7):
    times=np.arange(int(seconds*50),dtype=np.float32)*.02
    features=np.zeros((len(times),ACOUSTIC_DIM),np.float32)
    features[:,80]=.7+.3*np.sin(times*3)
    features[:,92]=(.5+.5*np.cos(times*12))
    features[:,93]=.5
    features[:,95]=1
    return EventAudio(features,times,seconds,120.)


def test_no_fixed_four_beat_phase_or_future_chart_in_draft_inputs():
    audio=fixture_audio()
    meters=[MeterChange(F(0),3,4)]
    spans=build_spans(12,meters,bars_per_span=2)
    song=organize_song(audio,[EventTempo(F(0),120)],0,meters=meters,spans=spans)
    features=form_audio(song,F(0),F(6))
    assert features.shape[1]==120
    assert not features[:,108:112].any()
    # The actual bar phase repeats at 3 beats, not 4.
    torch.testing.assert_close(features[0,114:116],features[75,114:116],atol=1e-5,rtol=1e-5)
    with pytest.raises(ValueError,match='future'):
        input_example(song,1,[EventPoint(F(6),'1')])
    with pytest.raises(ValueError,match='draft/right'):
        input_example(song,0,[],right=[EventPoint(F(7),'1')])


def test_real_cpu_training_finetune_resume_and_blueprint_layout(tmp_path,monkeypatch):
    from maichart.form import data,train,checkpoint
    audio=fixture_audio(3)
    song=tmp_path/'song';song.mkdir()
    (song/'maidata.txt').write_text('&title=fixture\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){12}1,2,3,1,2,3,1h[4:4],2,3,1,2,3,E\n',encoding='utf-8')
    (song/'track.wav').write_bytes(b'fixture')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[{'song_id':'fixture','maidata_path':'song/maidata.txt','audio_path':'song/track.wav','difficulties':[{'difficulty_index':5,'level_raw':'13'}]}]}))
    real_dataset=data.FormDataset
    monkeypatch.setattr(train,'FormDataset',lambda *args,**kw:real_dataset(*args,**kw,audio_loader=lambda *a,**k:audio))
    common=['--manifest',str(manifest),'--cache-dir',str(tmp_path/'cache'),'--epochs','1',
            '--batch-size','1','--max-steps','2','--d-model','16','--nhead','2','--layers','1',
            '--max-tokens','192','--max-memory-tokens','64','--max-region-seconds','3']
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        assert train.main(common+['--checkpoint-dir',str(tmp_path/'model')])==0
        original=tmp_path/'model/form_last.pt'
        model,planner,codec,saved=checkpoint.load(original,inference=True)
        assert saved['optimizer_steps']==2 and saved['blueprint_steps']==1
        assert saved['task_training']=={'draft':1,'revision':1}
        untrained_revision=dict(saved,task_training={'draft':1,'revision':0})
        missing=tmp_path/'untrained_revision.pt';torch.save(untrained_revision,missing)
        with pytest.raises(ValueError,match='revision'):
            checkpoint.load(missing,inference=True,task='revision')
        assert train.main(common+['--checkpoint-dir',str(tmp_path/'tuned'),'--init-checkpoint',str(original)])==0
        assert train.main(common+['--checkpoint-dir',str(tmp_path/'resumed'),'--resume',str(original)])==0
        organized=organize_song(audio,[EventTempo(F(0),120)],0,max_seconds=3)
        from maichart.form.generate import plan_onsets,generate_chart
        plan=plan_onsets(planner,organized,condition=__import__('maichart.generation.conditions',fromlist=['GenerationCondition']).GenerationCondition(),threshold=.01)
        assert plan['musical_units']=='whole_song'
        # Real decoder, known onset skeleton. This tests interface and grammar,
        # not learned chart quality from a single optimizer step.
        from maichart.generation.conditions import GenerationCondition
        desired={'onsets':[{'beat':'0'},{'beat':'1/3'},{'beat':'2/3'},{'beat':'7/5'}]}
        points,report=generate_chart(model,planner,codec,organized,GenerationCondition(),['1','2','3'],
            temperature=0.,blueprint=desired)
        assert [p.beat for p in points]==[F(0),F(1,3),F(2,3),F(7,5)]
        assert report['quality']['syntax_valid']
        model.config.max_tokens=28
        points,bounded=generate_chart(model,planner,codec,organized,GenerationCondition(),['1','2','3'],
            temperature=0.,blueprint=desired)
        assert [p.beat for p in points]==[F(0),F(1,3),F(2,3),F(7,5)]
        assert bounded['arrangement'][0]['resource_passes']>1
        fake=tmp_path/'old.pt';torch.save({'schema':'maichart-event-generator-checkpoint-v1'},fake)
        with pytest.raises(ValueError,match='old event checkpoints'):
            checkpoint.load(fake)
    finally:torch.set_num_threads(threads)

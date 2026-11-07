from fractions import Fraction
import torch
from maichart.event_timing import EventPoint
from maichart.composer_v2.codec import ComponentCodec
from maichart.composer_v2.model import ComposerV2,ComposerV2Config


def fixture():
    codec=ComponentCodec.fit(['1','2b','1h[4:1]/5h[4:2]','1h[4:2]/5h[4:1]',
        '3-7[8:1]','A1/B2/Cf','Ch[4:1]'])
    model=ComposerV2(ComposerV2Config(d_model=32,layers=2,nhead=4,dim_feedforward=64,
        audio_layers=1,song_layers=1,dropout=0.,audio_backend='native'),codec)
    tokens=[codec.bos_id,codec.ctx_id,codec.tgt_id]+codec.encode_points([EventPoint(Fraction(1),'1h[4:1]/5h[4:2]')],0)+[codec.eos_id]
    ids=torch.tensor([tokens[:-1]])
    batch=dict(input_ids=ids,token_mask=torch.ones_like(ids,dtype=torch.bool),
        token_beats=torch.tensor([codec.token_clocks(tokens[:-1])]),labels=torch.tensor([tokens[1:]]),
        token_weights=torch.ones_like(ids,dtype=torch.float32),condition=torch.zeros(1,16),sample_weight=torch.ones(1),
        local_audio=torch.randn(1,32,98),local_semantic=torch.zeros(1,32,1024),local_semantic_mask=torch.zeros(1,32,dtype=torch.bool),
        local_mask=torch.ones(1,32,dtype=torch.bool),local_beats=torch.linspace(-16,20,32)[None],
        song_audio=torch.randn(1,8,98),song_semantic=torch.zeros(1,8,1024),song_semantic_mask=torch.zeros(1,8,dtype=torch.bool),
        song_mask=torch.ones(1,8,dtype=torch.bool),song_beats=torch.linspace(0,80,8)[None],start_beat=[Fraction(0)],song_id=['fixture'])
    return codec,model,batch


def test_component_binding_and_exact_clock_roundtrip():
    codec,_,_=fixture()
    a=codec.encode_points([EventPoint(Fraction(1),'1h[4:1]/5h[4:2]')],0)
    b=codec.encode_points([EventPoint(Fraction(1),'1h[4:2]/5h[4:1]')],0)
    assert a!=b
    assert codec.decode_points(a)[0].note_text=='1h[4:1]/5h[4:2]'
    old=EventPoint(Fraction(-100,3),'3-7[8:1]')
    encoded=codec.encode_points([old],0,region='carry')
    assert codec.decode_points(encoded)==[old]
    assert codec.token_clocks(encoded)[-1]==float(old.beat)


def test_finetune_vocabulary_extension_preserves_checkpoint_rows():
    from maichart.composer_v2.train import _extend_vocabulary_state

    codec, parent, _ = fixture()
    expanded_codec = codec.extended(['1^7[8:1]'])
    expanded = ComposerV2(parent.config, expanded_codec)
    initial_rows = expanded.embedding.weight[codec.vocab_size:].detach().clone()
    state = parent.state_dict()
    expanded.load_state_dict(_extend_vocabulary_state(expanded, state, codec.vocab_size))
    for name, weights in expanded.state_dict().items():
        if name in ('embedding.weight', 'output_bias'):
            weights = weights[:codec.vocab_size]
        assert torch.equal(weights, state[name]), name
    assert torch.equal(expanded.embedding.weight[codec.vocab_size:], initial_rows)
    assert expanded.embedding.weight.data_ptr() != parent.embedding.weight.data_ptr()


def test_grammar_matches_every_authored_component_and_allows_empty_windows():
    codec,_,_=fixture()
    target=codec.encode_points([EventPoint(Fraction(0),'A1/B2/Cf'),EventPoint(Fraction(1),'3-7[8:1]')],0)+[codec.eos_id]
    for i,value in enumerate(target):assert value in codec.allowed_next(target[:i])
    assert codec.eos_id in codec.allowed_next([])
    assert codec.decode_points([codec.eos_id])==[]


def test_unseen_duration_values_share_numeric_components_and_grammar():
    codec=ComponentCodec.fit(['1h[4:1]','1-5[8:1]'])
    for atom in ('1h[32:11]','1h[6:19]','1-5[0.1622##1.7321]','1-5[#0.8057]'):
        encoded=codec.encode_atom(atom)
        for i,token in enumerate(encoded):assert token in codec.allowed_atom_next(encoded[:i]),(atom,i)
        points=codec.decode_points([codec.ids['T:0'],*encoded])
        from maichart.composer_v2.codec import atom_components
        assert atom_components(points[0].note_text)==atom_components(atom)


def test_causal_full_and_incremental_logits_agree():
    _,model,batch=fixture();model.eval()
    with torch.inference_mode():
        full=model(batch)['logits'];cache=model.encode_audio(batch);parts=[]
        for end in range(1,batch['input_ids'].size(1)+1):
            query={**batch,**{k:batch[k][:,:end] for k in ('input_ids','token_mask','token_beats')}}
            parts.append(model.decode(query,cache)[:,-1:])
        torch.testing.assert_close(torch.cat(parts,1),full,atol=2e-6,rtol=2e-5)
        changed={**batch,'input_ids':batch['input_ids'].clone()}
        changed['input_ids'][:,-2:]=1
        torch.testing.assert_close(model(changed)['logits'][:,:-2],full[:,:-2],atol=1e-6,rtol=1e-5)


def test_music_and_component_history_receive_real_gradients():
    _,model,batch=fixture();loss,metrics=model.loss(model(batch),batch);loss.backward()
    assert torch.isfinite(loss) and metrics['token_count']>0
    for module in (model.embedding,model.mel,model.decoder[0].local_attention,model.decoder[1].song_attention):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0


def test_rewound_kv_matches_clean_replay_of_repaired_prefix():
    _,model,batch=fixture();model.eval()
    with torch.inference_mode():
        cache=model.encode_audio(batch);model.decode(batch,cache)
        keep=batch['input_ids'].size(1)-3
        model.rewind_decode_cache(cache,keep)
        query={**batch,**{k:batch[k][:,:keep] for k in ('input_ids','token_mask','token_beats')}}
        repaired=model.decode(query,cache)[:,-1]
        clean=model(query)['logits'][:,-1]
        torch.testing.assert_close(repaired,clean,atol=2e-6,rtol=2e-5)


def test_level_checkpoint_migration_preserves_unspecified_condition_logits():
    codec,new,batch=fixture()
    old_config={**new.config.to_dict(),'condition_policy':'level_only_v1'}
    old=ComposerV2(ComposerV2Config.from_dict(old_config),codec).eval()
    loaded=new.load_state_dict(old.state_dict(),strict=False)
    assert loaded.missing_keys==['condition_missing.weight'] and not loaded.unexpected_keys
    new.eval()
    with torch.inference_mode():torch.testing.assert_close(new(batch)['logits'],old(batch)['logits'],atol=0,rtol=0)
    # A measured zero and an automatic/unknown value have separate representations.
    with torch.no_grad():new.condition_missing.weight[:,1].fill_(.5)
    known=torch.zeros_like(batch['condition'],dtype=torch.bool);known[:,0]=True
    unknown=new.condition_context({**batch,'condition_known':known})
    known[:,1]=True
    measured_zero=new.condition_context({**batch,'condition_known':known})
    torch.testing.assert_close(unknown-measured_zero,torch.full_like(unknown,.5))


def test_validation_uses_deployed_automatic_controls_not_answer_statistics():
    from maichart.composer_v2.train import _validate
    _,model,batch=fixture()
    batch['condition'].fill_(.6)
    batch['condition_known']=torch.ones_like(batch['condition'],dtype=torch.bool)
    seen=[];forward=model.forward
    def record(query):
        seen.append(query['condition_known'].clone())
        return forward(query)
    model.forward=record
    assert _validate(model,[batch],torch.device('cpu'),'fp32',0)>0
    assert seen[0][:,0].all() and not seen[0][:,1:].any()



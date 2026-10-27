from maichart.composer.evaluation import compare_chart_reports


def test_reference_deviation_keeps_zero_and_unknown_distinct():
    baseline={'rates':{'touch_heads':{'rate':0.,'denominator':100},
                       'touch_events':{'rate':None}}}
    output={'rates':{'touch_heads':{'rate':.1,'denominator':20},
                     'touch_events':{'rate':1.}}}
    comparison=compare_chart_reports(baseline,output)
    assert len(comparison['alerts'])==1
    assert comparison['alerts'][0]['multiple'] is None
    assert comparison['alerts'][0]['metric']=='touch_heads'
    assert 'review signals' in comparison['scope']

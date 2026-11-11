"""Optional native MiaCode TapOnSlide enforcement and exact-child cleanup."""
from dataclasses import replace
import math


def options(args):
    threshold=float(getattr(args,'tap_on_slide_threshold_ms',200.))
    if not math.isfinite(threshold) or not 0<=threshold<=250:
        raise ValueError('撞尾检测阈值必须为0–250 ms')
    return dict(enabled=bool(getattr(args,'intercept_tap_on_slide',False)),threshold_ms=threshold)


def playability_options(args):
    from maichart.generation.muri.touch_reachability import (
        DEFAULT_COHOLD_RADIUS,
        validate_touch_reachability,
    )
    base = options(args)
    touch_policy, touch_radius = validate_touch_reachability(
        getattr(args, 'touch_reachability_policy', 'palm_span_v1'),
        getattr(args, 'touch_cohold_radius', DEFAULT_COHOLD_RADIUS),
    )
    return {
        **base,
        'touch_reachability_policy': touch_policy,
        'touch_cohold_radius': touch_radius,
        'intercept_slide_head_tap': bool(getattr(args, 'intercept_slide_head_tap', False)),
    }


def configure(guard,settings):
    if settings is not None:
        guard.intercept_tap_on_slide=bool(settings.get('enabled',guard.intercept_tap_on_slide))
        guard.intercept_slide_head_tap=bool(settings.get('intercept_slide_head_tap',getattr(guard,'intercept_slide_head_tap',False)))
        modes={
            **guard.config.conflict_kind_modes,
            'TapOnSlide':'enforce' if guard.intercept_tap_on_slide else 'report',
            'SlideHeadTap':'enforce' if guard.intercept_slide_head_tap else 'report',
        }
        updates=dict(
            tap_on_slide_threshold_ms=settings.get('threshold_ms',guard.config.tap_on_slide_threshold_ms),
            conflict_kind_modes=modes,
        )
        if 'touch_reachability_policy' in settings:
            updates['touch_reachability_policy']=settings['touch_reachability_policy']
        if 'touch_cohold_radius' in settings:
            updates['touch_cohold_radius']=float(settings['touch_cohold_radius'])
        guard.config=replace(guard.config,**updates)
    return guard


def cleanup(points,tempo,first,guard):
    from maichart.generation.post_process import process_chart_points
    from maichart.generation.playability_policy import CONFLICT_KINDS
    settings=dict(enabled=guard.intercept_tap_on_slide,threshold_ms=guard.config.tap_on_slide_threshold_ms)
    report={**settings,'removed':[],'remaining_hard':0,'policy':'miacode_exact_affected_child_v1'}
    if not settings['enabled']:return points,report
    # Only TapOnSlide may be repaired. Native warnings and other kinds retain
    # their independent report/enforce contracts; no global proximity rule.
    config=replace(guard.config,conflict_repair_mode='local',note_repair_mode='delete',
        conflict_kind_modes={k:'enforce' if k=='TapOnSlide' else 'report'
                             for k in CONFLICT_KINDS})
    current=list(points)
    while True:
        result=process_chart_points(current,tempo,first,config)
        def other_hard(issues):
            return {(i.conflict_type,tuple(sorted((str(n.beat),n.raw) for n in i.notes)))
                for i in issues if i.severity=='hard' and i.conflict_type!='miacode_TapOnSlide'}
        if other_hard(result.remaining_conflicts)-other_hard(result.conflicts_detected):
            raise RuntimeError('撞尾清理引入了其他类别的硬无理；未导出谱面')
        report['removed'].extend(result.repairs_applied)
        current=result.points
        remaining=[i for i in result.remaining_conflicts if i.conflict_type=='miacode_TapOnSlide' and i.severity=='hard']
        if not remaining:break
        if not result.repairs_applied:
            report['remaining_hard']=len(remaining)
            raise RuntimeError(f'撞尾清理后仍有{len(remaining)}处无法安全定位的MiaCode硬冲突；未导出谱面')
    return current,report

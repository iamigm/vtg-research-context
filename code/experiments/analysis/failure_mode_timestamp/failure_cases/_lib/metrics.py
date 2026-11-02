"""Geometric failure descriptions; these labels do not establish semantic causes."""
import math

def tiou(gt,pred):
    if pred is None or len(pred)!=2 or not all(math.isfinite(v) for v in pred) or pred[1]<=pred[0]: return 0.0
    inter=max(0,min(gt[1],pred[1])-max(gt[0],pred[0]))
    return inter/(max(gt[1],pred[1])-min(gt[0],pred[0]))

def measure(row):
    gt=row['span'][0];pred=row['timestamps'][0] if row['timestamps'] else None
    raw=row['parsed_timestamps'][0] if row['parsed_timestamps'] else None
    dur=row['duration'];length=gt[1]-gt[0];score=tiou(gt,pred)
    valid=pred is not None and all(math.isfinite(v) for v in pred) and pred[1]>pred[0]
    d={'iou':score,'raw_iou':tiou(gt,raw),'success_03':score>=.3,'success_05':score>=.5,'success_07':score>=.7,'gt_length':length,'gt_fraction':length/dur,'gt_center_relative':sum(gt)/2/dur,'gt_beyond_duration':gt[1]>dur,'multiple_pairs':len(row['timestamps'])>1,'rounding_flips_05':(score>=.5)!=(tiou(gt,raw)>=.5)}
    if not valid:
        d.update(failure_type='parse_failure' if pred is None else 'invalid_interval',length_ratio=None,center_error_relative=None,start_error=None,end_error=None,gt_coverage=0,pred_precision=0,out_of_bounds=False,out_of_bounds_excess_seconds=0,full_video_prediction=False)
        return d
    s,e=pred;inter=max(0,min(gt[1],e)-max(gt[0],s));ratio=(e-s)/length
    if score>=.5:label='success'
    elif inter==0:label='no_overlap'
    elif ratio>=1.5:label='overextended'
    elif ratio<=2/3:label='underextended'
    else:label='shifted'
    d.update(failure_type=label,length_ratio=ratio,center_error_relative=((s+e)-(gt[0]+gt[1]))/2/dur,start_error=s-gt[0],end_error=e-gt[1],gt_coverage=inter/length,pred_precision=inter/(e-s),out_of_bounds=s<0 or e>dur,out_of_bounds_excess_seconds=max(0,-s,e-dur),full_video_prediction=s<=.05*dur and e>=.95*dur)
    return d

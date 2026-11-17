"""在真正留出记录上验证可调用推理入口，含篡改未来数据的不变性检查。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from common import read_json,write_json
from predict import Predictor,record_input

root=Path(__file__).resolve().parent
manifest=read_json(root/'prepared/manifest.json');windows=np.load(root/'prepared/windows.npz')
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
checks=[]
for fold in manifest['folds']:
    k=fold['fold'];rid=fold['test'][0];record=manifest['records'][rid]
    path=root.parents[1]/'source_data/neural_forecast_v2_2026-09-15/records'/record['key']
    signal,names,times=record_input(path)
    predictor=Predictor(root,k,'anatomical',17,'cuda')
    left=manifest['config']['calibration_frames'];middle=left+manifest['config']['history']
    result=predictor.forecast(signal,names,times,left)
    changed=signal.copy();changed[middle:]=np.nan
    changed_times=times.copy();changed_times[middle:]=-999999
    second=predictor.forecast(changed,names,changed_times,left)
    np.testing.assert_array_equal(result['neural_normalized'],second['neural_normalized'])
    np.testing.assert_array_equal(result['behavior'],second['behavior'])
    index=int(np.flatnonzero((windows['record']==rid)&(windows['left']==left))[0])
    saved=np.load(root/'results/predictions/neural_anatomical_s17.npy')[index]
    saved_behavior=np.load(root/'results/predictions/behavior_cascade_anatomical_s17.npy')[index]
    np.testing.assert_allclose(result['neural_normalized'],saved,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(result['behavior'],saved_behavior,rtol=2e-5,atol=2e-5)
    output=root/'results/demo'/f'fold{k}_heldout_prediction.npz';output.parent.mkdir(exist_ok=True)
    np.savez(output,**result)
    checks.append({'fold':k,'record_index':rid,'record_id':record['id'],'left':left,
        'future_signal_and_timestamps_ignored':True,'saved_neural_predictions_reproduced':True,
        'saved_cascade_behavior_reproduced':True,'neural_max_abs_difference':float(np.max(np.abs(result['neural_normalized']-saved))),
        'behavior_max_abs_difference':float(np.max(np.abs(result['behavior']-saved_behavior)))})
write_json(root/'results/inference_verification.json',{'passed':True,'cases':checks})
print(__import__('json').dumps(checks,indent=2))

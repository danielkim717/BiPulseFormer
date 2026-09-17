import importlib.util
from pathlib import Path
import sys
import numpy as np
import unittest
import tempfile
import torch

ROOT = Path(__file__).resolve().parent/'source'
sys.path.insert(0,str(ROOT/'scripts'))
from run_final_protocol import choose, window_starts, merge_windows, normalize_input


def test_overlap_reconstructs_full_common_timebase_without_touching_gt():
    signal=np.sin(np.arange(1120)/13.)
    starts=window_starts(len(signal))
    windows=np.stack([signal[s:s+160] for s in starts])
    labels=np.arange(1120,dtype=np.float32).reshape(7,160)
    before=labels.tobytes()
    np.testing.assert_allclose(merge_windows(windows,starts,1120),signal,rtol=0,atol=0)
    assert labels.tobytes()==before
    assert starts[0]==0 and starts[-1]+160==1120
    with unittest.TestCase().assertRaises(ValueError):merge_windows(windows[:-1],starts[:-1],1120)
    with unittest.TestCase().assertRaises(ValueError):window_starts(1170)


def test_source_only_twenty_epoch_selection_and_tie():
    rows=[dict(epoch=i,dataset_role='source_validation',recording_test_rmse=2.) for i in range(1,21)]
    rows[3]['recording_test_rmse']=1.;rows[10]['recording_test_rmse']=1.
    assert choose(rows)['epoch']==4
    with unittest.TestCase().assertRaises(ValueError):choose(rows[:-1])
    rows[3]['dataset_role']='target_test'
    with unittest.TestCase().assertRaises(ValueError):choose(rows)


def test_input_matches_frozen_v1_on_real_synthetic_images(tmp_path):
    import cv2
    spec=importlib.util.spec_from_file_location('legacy',ROOT/'legacy/rppg_dataset.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    class Empty(legacy.RPPGDataset):
        def _prepare_data(self):pass
    ds=Empty('PURE','',face_crop=False,data_type='diff_normalized')
    rng=np.random.default_rng(7);paths=[];frames=[]
    for i in range(161):
        raw=rng.integers(0,256,(20,18,3),dtype=np.uint8)
        p=tmp_path/f'{i}.png';assert cv2.imwrite(str(p),raw);paths.append(str(p))
        frames.append(ds.transform(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)))
    ds.samples=[dict(bvp=np.sin(np.arange(161)/7),img_paths=paths,video_id='test',first_frame_idx=0)]
    actual,_=ds[0]
    new=normalize_input(torch.stack(frames).permute(1,0,2,3))
    torch.testing.assert_close(new,actual,rtol=0,atol=0)


def test_native_reader_tail_preserves_exact_common_length(tmp_path):
    import cv2
    from run_final_protocol import make_reader
    spec=importlib.util.spec_from_file_location('legacy_tail',ROOT/'legacy/rppg_dataset.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    for i in range(160):assert cv2.imwrite(str(tmp_path/f'{i:04}.png'),np.full((16,16,3),i,dtype=np.uint8))
    r=dict(id='tail',video=str(tmp_path),coverage=dict(decoded_frames=160))
    read,_=make_reader(legacy,[r],'PURE')
    raw=read(r,0,True)
    assert raw.shape==(3,161,128,128)
    assert torch.equal(raw[:,-1],raw[:,-2])
    assert torch.count_nonzero(normalize_input(raw)[:,-1])==0
    with unittest.TestCase().assertRaises(ValueError):read(r,0,False)


class FinalProtocolTests(unittest.TestCase):
    def test_overlap(self):test_overlap_reconstructs_full_common_timebase_without_touching_gt()
    def test_selection(self):test_source_only_twenty_epoch_selection_and_tie()
    def test_input(self):
        with tempfile.TemporaryDirectory() as tmp:test_input_matches_frozen_v1_on_real_synthetic_images(Path(tmp))
    def test_tail(self):
        with tempfile.TemporaryDirectory() as tmp:test_native_reader_tail_preserves_exact_common_length(Path(tmp))


if __name__=='__main__':unittest.main()

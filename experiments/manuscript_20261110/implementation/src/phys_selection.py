"""Fixed UBFC-PHYS selection from rPPG-Toolbox (NeurIPS 2023), Appendix H."""
import copy

EXCLUDED = {
    'T1': (3,8,9,26,28,30,31,32,33,40,52,53,54,56),
    'T2': (1,4,6,8,9,11,12,13,14,19,21,22,25,26,27,28,31,32,33,35,38,39,41,42,45,47,48,52,53,55),
    'T3': (5,8,9,10,13,14,17,22,25,26,28,30,32,33,35,37,40,47,48,49,50,52,53),
}
EXCLUDED_IDS = {f's{s}_{t}' for t, people in EXCLUDED.items() for s in people}
FULL_IDS = {f's{s}_{t}' for s in range(1,57) for t in EXCLUDED}
SELECTED_IDS = FULL_IDS - EXCLUDED_IDS


def select_plan(full):
    ids = [r['video_id'] for r in full['records']]
    if len(ids) != 168 or set(ids) != FULL_IDS:
        raise ValueError('Verified full 168-recording inventory required before selection')
    plan = copy.deepcopy(full)
    plan['records'] = [r for r in plan['records'] if r['video_id'] in SELECTED_IDS]
    plan['subjects'] = sorted({r['video_id'].split('_')[0] for r in plan['records']})
    plan.update(expected_subjects=48, expected_recordings=101,
                scope='Toolbox Appendix H selection: 48 subjects / 101 recordings (T1=42,T2=26,T3=33); 67 recordings excluded; other protocol differences remain')
    plan['selection'] = dict(name='toolbox_appendix_h_2023', excluded_video_ids=sorted(EXCLUDED_IDS),
                             source='https://proceedings.neurips.cc/paper_files/paper/2023/file/d7d0d548a6317407e02230f15ce75817-Paper-Datasets_and_Benchmarks.pdf#page=25')
    plan['protocol']['dataset_selection'] = plan['selection']
    return plan


def validate_inventory(plan):
    ids = [r['video_id'] for r in plan['records']]
    expected = SELECTED_IDS if plan.get('selection', {}).get('name') == 'toolbox_appendix_h_2023' else FULL_IDS
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError('PHYS recording inventory does not match the declared selection')
    actual = {v.split('_')[0] for v in ids}
    if actual != set(plan['subjects']):
        raise ValueError('PHYS subject inventory does not match recordings')
    return sorted(actual)

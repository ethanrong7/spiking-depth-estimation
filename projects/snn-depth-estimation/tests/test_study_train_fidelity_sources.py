"""Tiny metadata/byte fixtures for source routing; no real arrays or network."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts'))
import prepare_train_fidelity_sources as sources


def fixture(root):
    roots = dict(zip(sources.ALIASES, (str(root/'historical'), str(root/'new'))))
    index = dict(test=[], test_values_opened=False, sequences=[])
    records = {key: {} for key in sources.ALIASES}
    for number, sid in enumerate(sources.SCENES):
        alias = sources.ALIASES[0 if number < 2 else 1]
        frames = []
        for position in range(10):
            stem = f'{position+6:04d}'
            frame = dict(index=position, source_frame=stem,
                rgb_l=f'frames_cleanpass/{sid}/left/{stem}.png',
                rgb_r=f'frames_cleanpass/{sid}/right/{stem}.png',
                disparity_l=f'disparity/{sid}/left/{stem}.pfm' if position in range(1, 9) else None)
            frames.append(frame)
            for key in ('rgb_l', 'rgb_r', 'disparity_l'):
                name = frame[key]
                if name is None:
                    continue
                raw = name.encode()  # Synthetic bytes, deliberately not an image/PFM.
                path = Path(roots[alias])/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                records[alias][name] = dict(role='gt' if key == 'disparity_l' else 'rgb',
                    source_id=sid, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        index['sequences'].append(dict(source_id=sid, split='train', original_split='TRAIN',
            root_alias=alias, historical_exposure=number < 2, reference_centers=list(range(1, 9)), frames=frames))
    return index, roots, records


class FidelitySourceTests(unittest.TestCase):
    def test_fixed_cohort_routes_to_both_roots_and_only_declared_gt(self):
        with tempfile.TemporaryDirectory() as folder:
            index, roots, records = fixture(Path(folder))
            panel = sources.select_panel(index, roots, records)
            self.assertEqual([row['source_id'] for row in panel['sequences']], list(sources.SCENES))
            for row in panel['sequences']:
                self.assertEqual([f['frame_index'] for f in row['files'] if f['role'] == 'gt'], [1, 4, 8])
                self.assertEqual(len(row['files']), 23)
                self.assertTrue(sources.source_file(panel, row, row['files'][0]).is_relative_to(Path(roots[row['root_alias']]).resolve()))
            self.assertEqual(sources.verify_source_bytes(panel)['files'], 92)
            self.assertFalse(panel['registered_for_generation'])

    def test_test_view_and_missing_acquisition_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            index, roots, records = fixture(Path(folder))
            changed = copy.deepcopy(index)
            changed['sequences'][0]['frames'][0]['rgb_l'] = 'frames_cleanpass/TEST/C/0087/left/0006.png'
            with self.assertRaisesRegex(ValueError, 'Train source'):
                sources.select_panel(changed, roots, records)
            changed = copy.deepcopy(index)
            changed['sequences'][0]['frames'][0]['rgb_l'] = changed['sequences'][0]['frames'][0]['rgb_r']
            with self.assertRaisesRegex(ValueError, 'frame/view/path'):
                sources.select_panel(changed, roots, records)
            del records[sources.ALIASES[1]][index['sequences'][2]['frames'][4]['disparity_l']]
            with self.assertRaisesRegex(ValueError, 'member missing'):
                sources.select_panel(index, roots, records)

    def test_changed_bytes_and_root_escape_are_rejected_before_decoding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index, roots, records = fixture(root)
            panel = sources.select_panel(index, roots, records)
            row = panel['sequences'][0]
            path = sources.source_file(panel, row, row['files'][0])
            raw = path.read_bytes()
            path.write_bytes(b'x'*len(raw))
            with self.assertRaisesRegex(ValueError, 'Changed source bytes'):
                sources.verify_source_bytes(panel)
            path.unlink()
            outside = root/'outside'
            outside.write_bytes(raw)
            path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                sources.source_file(panel, row, row['files'][0])

    def test_root_alias_and_scene_identity_cannot_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            index, roots, records = fixture(Path(folder))
            changed = copy.deepcopy(index)
            changed['sequences'][2]['root_alias'] = sources.ALIASES[0]
            with self.assertRaisesRegex(ValueError, 'root/exposure'):
                sources.select_panel(changed, roots, records)
            changed = copy.deepcopy(index)
            changed['sequences'].append(copy.deepcopy(changed['sequences'][0]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                sources.select_panel(changed, roots, records)


if __name__ == '__main__':
    unittest.main()

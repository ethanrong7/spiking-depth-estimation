"""Two tiny source/aggregation checks; no model/data imports or execution."""
import ast
import json
from pathlib import Path
import unittest

CAMP=Path(__file__).resolve().parents[1]


class TrainingSoftwareTests(unittest.TestCase):
    def test_paired_configs_only_change_arm_and_prefix_recipe(self):
        configs=[json.loads((CAMP/f"configs/stage9-{arm}-seed1234-5k.json").read_text()) for arm in ("V","F")]
        self.assertEqual({k for k in configs[0] if configs[0][k]!=configs[1][k]},{"arm","prefix_recipe"})
        self.assertEqual(configs[0]["updates"],5000);self.assertEqual(configs[0]["schedule_steps"],5100)

    def test_scene_macro_and_pooled_bad3_are_distinct(self):
        source=ast.parse((CAMP/"scripts/run_stage9.py").read_text())
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="summarize_frames")
        def require(ok,message):
            if not ok:raise RuntimeError(message)
        scope={"require":require};exec(compile(ast.Module(body=[node],type_ignores=[]),"aggregation_only","exec"),scope)
        frames=[]
        for scene in range(30):
            for frame in range(2):
                count=1 if frame==0 else 3;bad=1 if frame==0 else 0
                frames.append({"source_id":str(scene),"actions":{"p49-r16":{"epe":float(scene+frame),"bad1_percent":0.,"bad3_percent":100*bad/count,"bad5_percent":0.,"bad3_count":bad,"valid_pixels":count}}})
        result=scope["summarize_frames"](frames,["p49-r16"])["p49-r16"]
        self.assertEqual(result["mean_scene_epe"],15.)
        self.assertEqual(result["mean_scene_bad3_percent"],50.)
        self.assertEqual(result["pooled_bad3_percent"],25.)
        self.assertEqual(result["valid_pixels"],120)
        self.assertEqual(result["worst5_scene_epe"],27.5)


if __name__=="__main__":unittest.main()

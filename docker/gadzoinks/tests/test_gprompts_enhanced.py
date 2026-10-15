# Usage (from docker/gadzoinks):  python -m unittest discover -s tests -p 'test_*.py'
"""GPromptsEnhanced: the prompt comes from computed_prompt, and a seed linked
from its seed output (["40", 3]) is followed back to the node."""
import asyncio
import contextlib
import copy
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import WorkflowParser4 as wp  # noqa: E402

TEMPLATE = "References: Use the provided image from <Picture 1> as the exact starting frame. Audio: chamber music in the background, no voice.\n"
ENHANCED = "For the target video, at 0.00 seconds the camera holds on the starting frame while chamber music plays."
DYNAMIC = "Use the provided image as the exact starting frame. Audio: chamber music, no voice.\n"
SEED = 787486911856994

# Shape taken from a real LTX-style video run (nodes 40/43/47/48/55/11 as logged).
VIDEO_PF = {
    "40": {"class_type": "GPromptsEnhanced",
           "inputs": {"text": TEMPLATE, "trigger_words": "", "delimiter_style": "curly { }",
                      "seed": SEED, "preserve_dynamic_words": True, "target_megapixels": "0.5",
                      "enhance": "every run", "prefetch_next": "off", "enhancer": ["43", 0],
                      "images.image_1": ["47", 0], "computed_prompt": ENHANCED},
           "_meta": {"title": "Dynamic Prompts with Enhancer", "computed_prompt": ENHANCED,
                     "dynamic_prompt": DYNAMIC}},
    "43": {"class_type": "GPromptEnhancerLoaderGGUFEdit",
           "inputs": {"model": "huihui-qwen3.5-9b-abliterated-q4_k_m.gguf", "mmproj": "mmproj-BF16.gguf",
                      "system_prompt": "h3_frames_system_prompt.txt", "seed_hint": 1}},
    "47": {"class_type": "GLoadImageFromImmich",
           "inputs": {"album": "", "tag": "c5", "favorites_only": False, "min_rating": "any", "index": 14}},
    "48": {"class_type": "PreviewImage", "inputs": {"images": ["47", 0]}},
    "55": {"class_type": "SomethingToString", "inputs": {"prefix": ["40", 0], "suffix": ""}},
    "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": ["40", 3]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["40", 0], "clip": ["38", 0]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality, blurry", "clip": ["38", 0]}},
    "22": {"class_type": "CFGGuider", "inputs": {"cfg": 3.0, "model": ["12", 0],
                                                 "positive": ["6", 0], "negative": ["7", 0]}},
    "12": {"class_type": "UNETLoader", "inputs": {"unet_name": "video/ltx-2-19b.safetensors"}},
    "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "17": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": 20, "model": ["12", 0]}},
    "13": {"class_type": "SamplerCustomAdvanced",
           "inputs": {"noise": ["11", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["17", 0]}},
}


def parse(pf):
    with contextlib.redirect_stdout(io.StringIO()):      # the parser prints a lot
        return wp.parse_promptflow(copy.deepcopy(pf))


class GPromptsEnhancedTests(unittest.TestCase):
    def test_family(self):
        self.assertTrue(wp.is_gprompts('GPrompts'))
        self.assertTrue(wp.is_gprompts('GPromptsEnhanced'))
        self.assertFalse(wp.is_gprompts('GPromptEnhancerLoaderGGUFEdit'), "loaders aren't prompt nodes")
        self.assertFalse(wp.is_gprompts(None))

    def test_video_graph_prompt_and_linked_seed(self):
        r = parse(VIDEO_PF)
        self.assertEqual(r['prompt'], ENHANCED, "computed_prompt, not the typed template")
        self.assertEqual(r['seed'], str(SEED), "RandomNoise.noise_seed = ['40', 3] followed to the node")
        self.assertEqual(r['negative_prompt'], "worst quality, blurry")
        self.assertEqual(r['steps'], 20)
        self.assertEqual(r['model'], 'ltx-2-19b')

    def test_trace_text_uses_computed_prompt(self):
        self.assertEqual(wp.trace_text(["40", 0], VIDEO_PF), ENHANCED)

    def test_meta_only_computed_prompt(self):
        pf = copy.deepcopy(VIDEO_PF)
        del pf["40"]["inputs"]["computed_prompt"]
        self.assertEqual(parse(pf)['prompt'], ENHANCED)

    def test_never_ran_falls_back_to_text(self):
        pf = copy.deepcopy(VIDEO_PF)
        del pf["40"]["inputs"]["computed_prompt"]
        del pf["40"]["_meta"]["computed_prompt"]
        self.assertEqual(parse(pf)['prompt'], TEMPLATE)

    def test_ksampler_with_linked_seed(self):
        pf = {k: v for k, v in VIDEO_PF.items() if k not in ('11', '13', '16', '17', '22')}
        pf["3"] = {"class_type": "KSampler",
                   "inputs": {"seed": ["40", 3], "steps": 25, "cfg": 4.0, "sampler_name": "euler",
                              "scheduler": "simple", "model": ["12", 0],
                              "positive": ["6", 0], "negative": ["7", 0]}}
        r = parse(pf)
        self.assertEqual(r['seed'], str(SEED))
        self.assertEqual(r['prompt'], ENHANCED)

    def test_seed_through_primitive(self):
        pf = copy.deepcopy(VIDEO_PF)
        pf["90"] = {"class_type": "PrimitiveInt", "inputs": {"value": 42424242}}
        pf["40"]["inputs"]["seed"] = ["90", 0]
        self.assertEqual(parse(pf)['seed'], "42424242")

    def test_plain_gprompts_still_works(self):
        pf = copy.deepcopy(VIDEO_PF)
        pf["40"]["class_type"] = "GPrompts"
        r = parse(pf)
        self.assertEqual(r['prompt'], ENHANCED)
        self.assertEqual(r['seed'], str(SEED))

    def test_ui_workflow_conversion(self):
        workflow = {
            "nodes": [
                {"id": 40, "type": "GPromptsEnhanced",
                 "inputs": [{"name": "enhancer", "link": 1}],
                 "widgets_values": [TEMPLATE, "", "curly { }", SEED, "fixed", True, "0.5", "every run", "off"],
                 "properties": {"_meta": {"computed_prompt": ENHANCED}}},
                {"id": 11, "type": "RandomNoise",
                 "inputs": [{"name": "noise_seed", "link": 2}], "widgets_values": [0, "fixed"]},
                {"id": 6, "type": "CLIPTextEncode",
                 "inputs": [{"name": "clip", "link": None}, {"name": "text", "link": 3}], "widgets_values": [""]},
                {"id": 22, "type": "CFGGuider",
                 "inputs": [{"name": "model", "link": None}, {"name": "positive", "link": 4}],
                 "widgets_values": [3.0]},
                {"id": 13, "type": "SamplerCustomAdvanced",
                 "inputs": [{"name": "noise", "link": 5}, {"name": "guider", "link": 6}], "widgets_values": []},
            ],
            "links": [[2, 40, 3, 11, 0, "INT"], [3, 40, 0, 6, 1, "STRING"], [4, 6, 0, 22, 1, "CONDITIONING"],
                      [5, 11, 0, 13, 0, "NOISE"], [6, 22, 0, 13, 1, "GUIDER"]],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            r = asyncio.run(wp.parse_comfyui_workflow_async(workflow=workflow))
        self.assertEqual(r['prompt'], ENHANCED)
        self.assertEqual(r['seed'], str(SEED))


if __name__ == '__main__':
    unittest.main()

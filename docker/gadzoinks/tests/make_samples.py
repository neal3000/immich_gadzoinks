# Usage: python make_samples.py /tmp/gz_samples   (one file per supported format)
"""Write sample images carrying metadata the way each generator really stores it."""
import json
import os
import sys

from PIL import Image, PngImagePlugin

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---- ComfyUI graphs ---------------------------------------------------------
promptflow_sdxl = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl/juggernautXL_v9.safetensors"}},
    "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "hair/extr3m3hair_V3.safetensors",
                                                  "strength_model": 0.8, "strength_clip": 0.8,
                                                  "model": ["4", 0], "clip": ["4", 1]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "tiki bar at sunset, Rawai beach, neon torches", "clip": ["10", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, lowres", "clip": ["10", 1]}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 832, "batch_size": 1}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 123456789, "steps": 30, "cfg": 6.5,
                                               "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                                               "model": ["10", 0], "positive": ["6", 0], "negative": ["7", 0],
                                               "latent_image": ["5", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI"}},
}
ui_workflow_stub = {"last_node_id": 10, "nodes": [{"id": 6, "type": "CLIPTextEncode",
                                                   "widgets_values": ["tiki bar at sunset"]}], "links": []}

promptflow_flux = {
    "12": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2-klein-9b.safetensors", "weight_dtype": "default"}},
    "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": 987654321}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "underwater photo of a manta ray, Phuket", "clip": ["11", 0]}},
    "26": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
    "22": {"class_type": "BasicGuider", "inputs": {"model": ["12", 0], "conditioning": ["26", 0]}},
    "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "17": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": 20, "denoise": 1, "model": ["12", 0]}},
    "27": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["25", 0], "guider": ["22", 0],
                                                             "sampler": ["16", 0], "sigmas": ["17", 0],
                                                             "latent_image": ["27", 0]}},
}

img = lambda: Image.new("RGB", (64, 64), (200, 120, 40))

# 1. ComfyUI PNG (core SaveImage node)
info = PngImagePlugin.PngInfo()
info.add_text("prompt", json.dumps(promptflow_sdxl))
info.add_text("workflow", json.dumps(ui_workflow_stub))
img().save(f"{OUT}/comfy_sdxl.png", pnginfo=info)

# 2. ComfyUI Flux PNG with SamplerCustomAdvanced (API graph only)
info = PngImagePlugin.PngInfo()
info.add_text("prompt", json.dumps(promptflow_flux))
img().save(f"{OUT}/comfy_flux.png", pnginfo=info)

# 3. ComfyUI WebP (SaveAnimatedWEBP style: graphs stuffed into EXIF Make/Model)
exif = Image.Exif()
exif[0x010F] = "workflow:" + json.dumps(ui_workflow_stub)
exif[0x0110] = "prompt:" + json.dumps(promptflow_sdxl)
img().save(f"{OUT}/comfy.webp", exif=exif.tobytes())

# 4. A1111 / Forge PNG
a1111 = ("masterpiece, tiki mug on bar, <lora:extr3m3hair_V3:0.7>, warm light\n"
         "Negative prompt: ugly, watermark\n"
         'Steps: 28, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 5.5, Seed: 42424242, '
         'Size: 832x1216, Model hash: abc123, Model: ponyDiffusionV6XL, '
         'Lora hashes: "extr3m3hair_V3: 9f8e7d", Version: f2.0.1v1.10.1')
info = PngImagePlugin.PngInfo()
info.add_text("parameters", a1111)
img().save(f"{OUT}/forge.png", pnginfo=info)

# 5. A1111 JPEG: same text in EXIF UserComment, 'UNICODE\0' + UTF-16BE (piexif helper)
exif = Image.Exif()
exif.get_ifd(0x8769)[0x9286] = b"UNICODE\x00" + a1111.replace("f2.0.1v1.10.1", "v1.10.1").encode("utf-16-be")
img().save(f"{OUT}/a1111.jpg", exif=exif.tobytes())

# 6. Fooocus PNG (JSON parameters)
fooocus = {"prompt": "retro surf poster, palm trees", "negative_prompt": "text, logo",
           "steps": 30, "guidance_scale": 4, "seed": "5550123", "sampler": "dpmpp_2m_sde_gpu",
           "scheduler": "karras", "base_model": "juggernautXL_v8Rundiffusion.safetensors",
           "resolution": "(896, 1152)", "loras": [["sd_xl_offset_example-lora_1.0.safetensors", 0.1]],
           "metadata_scheme": "fooocus"}
info = PngImagePlugin.PngInfo()
info.add_text("parameters", json.dumps(fooocus))
img().save(f"{OUT}/fooocus.png", pnginfo=info)

# 7. NovelAI PNG
info = PngImagePlugin.PngInfo()
info.add_text("Software", "NovelAI")
info.add_text("Source", "NovelAI Diffusion V4.5 4BDE2A90")
info.add_text("Description", "1girl, hula dancer, tiki torches")
info.add_text("Comment", json.dumps({"prompt": "1girl, hula dancer, tiki torches", "steps": 28,
                                     "scale": 5.0, "seed": 31337, "sampler": "k_euler_ancestral",
                                     "width": 832, "height": 1216, "uc": "lowres, bad anatomy"}))
img().save(f"{OUT}/novelai.png", pnginfo=info)

# 8. InvokeAI PNG
info = PngImagePlugin.PngInfo()
info.add_text("invokeai_metadata", json.dumps({
    "positive_prompt": "volcanic island at dawn, long exposure", "negative_prompt": "people",
    "seed": 77, "steps": 35, "cfg_scale": 7.5, "scheduler": "dpmpp_2m_k", "width": 1024, "height": 768,
    "model": {"name": "SDXL Base 1.0"}, "loras": [{"model": {"name": "film-grain"}, "weight": 0.6}]}))
img().save(f"{OUT}/invokeai.png", pnginfo=info)

# 9. Draw Things PNG (XMP exif:UserComment)
dt = {"c": "koi pond in a Thai temple garden", "uc": "blurry", "seed": 2024, "steps": 8,
      "scale": 1.0, "sampler": "Euler A Trailing", "model": "flux_1_schnell_q8p.ckpt", "size": "768x768"}
xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       '<rdf:Description xmlns:exif="http://ns.adobe.com/exif/1.0/"><exif:UserComment><rdf:Alt>'
       f'<rdf:li xml:lang="x-default">{json.dumps(dt).replace("&", "&amp;").replace("<", "&lt;").replace(chr(34), "&quot;")}</rdf:li>'
       '</rdf:Alt></exif:UserComment></rdf:Description></rdf:RDF></x:xmpmeta>')
info = PngImagePlugin.PngInfo()
info.add_itxt("XML:com.adobe.xmp", xmp)
img().save(f"{OUT}/drawthings.png", pnginfo=info)

# 10. Ordinary phone photo: camera EXIF, no AI metadata
exif = Image.Exif()
exif[0x010F] = "Apple"
exif[0x0110] = "iPhone 15 Pro"
exif.get_ifd(0x8769)[0x9286] = b"ASCII\x00\x00\x00Snorkel trip"
img().save(f"{OUT}/phone.jpg", exif=exif.tobytes())

# 11. ComfyUI PNG with a huge UI workflow (exiftool size check)
big = {"nodes": [{"id": i, "type": "Note", "widgets_values": ["x" * 500]} for i in range(400)], "links": []}
info = PngImagePlugin.PngInfo()
info.add_text("prompt", json.dumps(promptflow_sdxl))
info.add_text("workflow", json.dumps(big))
img().save(f"{OUT}/comfy_bigworkflow.png", pnginfo=info)

print("ok", sorted(os.listdir(OUT)))

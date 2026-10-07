# gadzoinks/EmbeddedMetadata.py
"""
Extract AI-generation metadata that is embedded in an image file.

Every generator hides its metadata somewhere different, and ComfyUI hides it
differently depending on which save node ran. So nothing here assumes a
format: each strategy looks at the whole tag bag, claims what it recognises
and returns a candidate result. Candidates are scored with
WorkflowParser4.result_quality; the best one wins and the others only fill
fields it is missing.

Input is a flat dict of tag name -> value. In production that is the subset of
exiftool tags the Immich server forwards (PNG:Prompt -> "Prompt", EXIF
UserComment -> "UserComment", ...). read_file_tags() builds the same bag from
a file with Pillow, for testing and offline backfill.

Output fields match WorkflowParser4.parse_promptflow so main.parse_and_store
can store either: prompt, negative_prompt, seed, steps, cfg, sampler_name,
scheduler, width, height, model, app, loraN / loraN_strength, loras.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from WorkflowParser4 import parse_comfyui_workflow_async, result_quality

logger = logging.getLogger(__file__)

MIN_QUALITY = 3          # a usable prompt on its own, or seed + steps + model
SCALAR_FIELDS = ('prompt', 'negative_prompt', 'seed', 'steps', 'cfg', 'sampler_name',
                 'scheduler', 'width', 'height', 'model')

# ============================================================================
# Tag bag normalisation
# ============================================================================

def norm_key(key: str) -> str:
    """'PNG:Prompt' -> 'prompt', 'Invokeai_metadata' -> 'invokeaimetadata'.
    exiftool, Pillow and the generators all spell keys differently; compare
    on lowercase alphanumerics only."""
    return re.sub(r'[^a-z0-9]', '', str(key).split(':')[-1].lower())


def decode_user_comment(raw: Any) -> Optional[str]:
    """EXIF UserComment is 8 bytes of charset header followed by the text.
    A1111 writes 'UNICODE\\0' + UTF-16; byte order varies by writer."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.strip('\x00').strip() or None
    if not isinstance(raw, (bytes, bytearray)):
        return str(raw)
    raw = bytes(raw)
    header, body = raw[:8], raw[8:]
    if header.startswith(b'UNICODE'):
        # ASCII-range text: BE puts the zero byte first, LE second
        candidates = ['utf-16-le', 'utf-16-be'] if body[1:2] == b'\x00' else ['utf-16-be', 'utf-16-le']
        for enc in candidates:
            try:
                text = body.decode(enc).strip('\x00').strip()
            except UnicodeDecodeError:
                continue
            if text and sum(c.isprintable() or c in '\n\r\t' for c in text) / len(text) > 0.95:
                return text
        return None
    if header.startswith((b'ASCII', b'\x00\x00\x00\x00')):
        raw = body
    return raw.decode('utf-8', errors='ignore').strip('\x00').strip() or None


def normalize_tags(tags: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, val in (tags or {}).items():
        k = norm_key(key)
        if not k or val is None:
            continue
        if k == 'usercomment' or isinstance(val, (bytes, bytearray)):
            val = decode_user_comment(val)
        elif isinstance(val, (dict, list)):
            val = json.dumps(val)
        else:
            val = str(val)
        if val and k not in out:          # first spelling wins
            out[k] = val
    return out


XMP_USER_COMMENT = re.compile(
    r'<exif:UserComment>.*?<rdf:li[^>]*>(.*?)</rdf:li>', re.DOTALL)


def _xmp_user_comment(xmp: Any) -> Optional[str]:
    if isinstance(xmp, (bytes, bytearray)):
        xmp = bytes(xmp).decode('utf-8', errors='ignore')
    if not isinstance(xmp, str):
        return None
    m = XMP_USER_COMMENT.search(xmp)
    if not m:
        return None
    import html
    return html.unescape(m.group(1)).strip() or None


def read_file_tags(path: str) -> Dict[str, Any]:
    """Build the same tag bag the server sends, straight from a file."""
    from PIL import Image
    tags: Dict[str, Any] = {}
    with Image.open(path) as img:
        for k, v in img.info.items():
            if isinstance(v, str):
                tags[k] = v                       # PNG tEXt/iTXt chunks
        exif = img.getexif()
        for tag_id, name in ((0x010F, 'Make'), (0x0110, 'Model'),
                             (0x010E, 'ImageDescription'), (0x0131, 'Software')):
            if exif.get(tag_id):
                tags[name] = exif[tag_id]
        uc = exif.get_ifd(0x8769).get(0x9286)
        if uc:
            tags['UserComment'] = uc
        xmp_comment = _xmp_user_comment(img.info.get('xmp') or img.info.get('XML:com.adobe.xmp'))
        if xmp_comment and 'UserComment' not in tags:
            tags['UserComment'] = xmp_comment
    return tags

# ============================================================================
# Small coercion helpers
# ============================================================================

def _json(val: Any) -> Any:
    if isinstance(val, (dict, list)):
        return val
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s or s[0] not in '{[':
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def _int(val: Any) -> Optional[int]:
    try:
        return int(float(str(val).strip()))
    except (TypeError, ValueError):
        return None


def _float(val: Any) -> Optional[float]:
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return None


def _seed(val: Any) -> Optional[str]:
    v = _int(val)
    return str(v) if v is not None and v >= 0 else None


def _size(val: Any) -> Tuple[Optional[int], Optional[int]]:
    """'1024x768', '(1024, 768)', [1024, 768] -> (1024, 768)."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return _int(val[0]), _int(val[1])
    m = re.search(r'(\d+)\s*[x×,]\s*(\d+)', str(val or ''))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _pick(d: Dict, *keys: str) -> Any:
    for k in keys:
        v = d.get(k)
        if v not in (None, '', []):
            return v
    return None


MODEL_EXT = re.compile(r'\.(safetensors|ckpt|pt|pth|bin|gguf|sft|onnx)$', re.IGNORECASE)


def model_name(val: Any) -> str:
    """Basename without a weights extension. Unlike extract_last_part this
    keeps dots that are part of the name ('SDXL Base 1.0', 'V4.5')."""
    name = str(val or '').replace('\\', '/').rsplit('/', 1)[-1]
    return MODEL_EXT.sub('', name).strip()


def _set_loras(result: Dict[str, Any], loras: List[Tuple[str, float]]) -> None:
    """Write loras in the loraN / loraN_strength shape parse_and_store expects."""
    flat: Dict[str, Any] = {}
    seen = set()
    n = 1
    for name, strength in loras:
        name = model_name(name)
        if not name or name.lower() == 'none' or name in seen:
            continue
        seen.add(name)
        flat[f'lora{n}'] = name
        flat[f'lora{n}_strength'] = strength if strength is not None else 1.0
        n += 1
    if flat:
        result.update(flat)
        result['loras'] = flat


PROMPT_LORA = re.compile(r'<(?:lora|lyco):([^:>]+)(?::([-\d.]+))?[^>]*>', re.IGNORECASE)


def _loras_from_prompt(text: Optional[str]) -> List[Tuple[str, float]]:
    out = []
    for name, weight in PROMPT_LORA.findall(text or ''):
        out.append((name.strip(), _float(weight) if weight else 1.0))
    return out

# ============================================================================
# Strategies. Each takes the normalised tag bag and returns a candidate dict
# (or None). Candidates may carry '_workflow' / '_promptflow' for storage.
# ============================================================================

# -- ComfyUI ------------------------------------------------------------------
# PNG save node:          tEXt 'prompt' (API graph) + 'workflow' (UI graph)
# WebP/animated save:     EXIF Make 'workflow:{...}', Model 'prompt:{...}'
# assorted JPEG savers:   UserComment / ImageDescription holding either graph
# So look for the graphs in every field instead of trusting the key names.

COMFY_PREFIX = re.compile(r'^\s*(prompt|workflow)\s*:\s*(?=\{)', re.IGNORECASE)
COMFY_KEY_ORDER = ('prompt', 'workflow', 'make', 'model', 'usercomment',
                   'imagedescription', 'comment', 'parameters')


def _is_ui_workflow(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get('nodes'), list)


def _is_promptflow(obj: Any) -> bool:
    return isinstance(obj, dict) and any(
        isinstance(v, dict) and 'class_type' in v for v in obj.values())


def find_comfy_graphs(tags: Dict[str, str]) -> Tuple[Optional[dict], Optional[dict]]:
    workflow = promptflow = None
    keys = [k for k in COMFY_KEY_ORDER if k in tags] + \
           [k for k in tags if k not in COMFY_KEY_ORDER]
    for k in keys:
        obj = _json(COMFY_PREFIX.sub('', tags[k]))
        if workflow is None and _is_ui_workflow(obj):
            workflow = obj
        elif promptflow is None and _is_promptflow(obj):
            promptflow = obj
        if workflow is not None and promptflow is not None:
            break
    return workflow, promptflow


async def strategy_comfyui(tags: Dict[str, str]) -> Optional[Dict[str, Any]]:
    workflow, promptflow = find_comfy_graphs(tags)
    if not workflow and not promptflow:
        return None
    result = await parse_comfyui_workflow_async(workflow, promptflow) or {}
    result = dict(result)
    result.setdefault('app', 'ComfyUI')
    result['_workflow'] = workflow
    result['_promptflow'] = promptflow
    return result

# -- Automatic1111 / Forge / SD.Next / Civitai / ComfyUI A1111-style savers -----
# "prompt\nNegative prompt: neg\nSteps: 20, Sampler: Euler a, CFG scale: 7, ..."

A1111_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
A1111_SOURCES = ('parameters', 'usercomment', 'comment', 'imagedescription', 'description')


def parse_a1111(text: str) -> Optional[Dict[str, Any]]:
    text = (text or '').replace('\r\n', '\n').strip()
    if 'Steps:' not in text:
        return None
    lines = text.split('\n')
    settings_idx = next((i for i in range(len(lines) - 1, -1, -1)
                         if lines[i].lstrip().startswith('Steps:')), None)
    if settings_idx is None:                 # single-line export: split at last "Steps:"
        cut = text.rfind('Steps:')
        body, settings = text[:cut].rstrip(' ,\n'), text[cut:]
    else:
        body, settings = '\n'.join(lines[:settings_idx]), '\n'.join(lines[settings_idx:])

    prompt, _, negative = body.partition('Negative prompt:')
    params: Dict[str, str] = {}
    for key, val in A1111_PARAM.findall(settings):
        val = val.strip()
        if val.startswith('"') and val.endswith('"'):
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                val = val[1:-1]
        params[key.strip()] = val

    result: Dict[str, Any] = {
        'prompt': prompt.strip() or None,
        'negative_prompt': negative.strip() or None,
        'steps': _int(params.get('Steps')),
        'sampler_name': params.get('Sampler'),
        'scheduler': params.get('Schedule type'),
        'cfg': _float(params.get('CFG scale')),
        'seed': _seed(params.get('Seed')),
        'model': params.get('Model'),
    }
    result['width'], result['height'] = _size(params.get('Size'))

    loras = _loras_from_prompt(result['prompt'])
    if not loras and params.get('Lora hashes'):
        loras = [(part.split(':')[0].strip(), 1.0)
                 for part in str(params['Lora hashes']).split(',') if part.strip()]
    _set_loras(result, loras)

    version = str(params.get('Version', ''))
    if 'Civitai resources' in params or 'Civitai metadata' in params:
        result['app'] = 'Civitai'
    elif version.startswith('f'):
        result['app'] = 'Forge'
    elif 'ComfyUI' in text:
        result['app'] = 'ComfyUI'
    else:
        result['app'] = 'Automatic1111'
    return result


def strategy_a1111(tags: Dict[str, str]) -> Optional[Dict[str, Any]]:
    for k in A1111_SOURCES:
        val = tags.get(k)
        if val and 'Steps:' in val and _json(val) is None:
            return parse_a1111(val)
    return None

# -- Fooocus / RuinedFooocus: 'parameters' is a JSON object --------------------

def strategy_fooocus(tags: Dict[str, str]) -> Optional[Dict[str, Any]]:
    data = _json(tags.get('parameters'))
    if not isinstance(data, dict) or _is_promptflow(data) or _is_ui_workflow(data):
        return None
    prompt = _pick(data, 'prompt', 'Prompt', 'full_prompt')
    if not prompt:
        return None
    if isinstance(prompt, list):
        prompt = ', '.join(map(str, prompt))
    negative = _pick(data, 'negative_prompt', 'Negative', 'full_negative_prompt')
    if isinstance(negative, list):
        negative = ', '.join(map(str, negative))

    result: Dict[str, Any] = {
        'prompt': str(prompt),
        'negative_prompt': negative,
        'seed': _seed(_pick(data, 'seed', 'Seed')),
        'steps': _int(_pick(data, 'steps', 'Steps')),
        'cfg': _float(_pick(data, 'guidance_scale', 'cfg', 'CFG')),
        'sampler_name': _pick(data, 'sampler', 'sampler_name'),
        'scheduler': _pick(data, 'scheduler'),
        'model': _pick(data, 'base_model', 'base_model_name', 'model'),
    }
    w, h = _size(_pick(data, 'resolution'))
    result['width'] = w or _int(data.get('width'))
    result['height'] = h or _int(data.get('height'))

    loras: List[Tuple[str, float]] = []
    for item in data.get('loras') or []:
        if isinstance(item, (list, tuple)) and item:
            loras.append((str(item[0]), _float(item[1]) if len(item) > 1 else 1.0))
        elif isinstance(item, str) and ':' in item:          # "name : 0.5"
            name, _, weight = item.rpartition(':')
            loras.append((name.strip(), _float(weight)))
    _set_loras(result, loras or _loras_from_prompt(result['prompt']))

    result['app'] = 'RuinedFooocus' if 'Prompt' in data else 'Fooocus'
    return result

# -- NovelAI: 'Description' = prompt, 'Comment' = JSON settings ---------------

def strategy_novelai(tags: Dict[str, str]) -> Optional[Dict[str, Any]]:
    data = _json(tags.get('comment'))
    if not isinstance(data, dict) or not ({'uc', 'scale'} & data.keys()):
        return None
    result: Dict[str, Any] = {
        'prompt': data.get('prompt') or tags.get('description'),
        'negative_prompt': data.get('uc'),
        'seed': _seed(data.get('seed')),
        'steps': _int(data.get('steps')),
        'cfg': _float(data.get('scale')),
        'sampler_name': data.get('sampler'),
        'width': _int(data.get('width')),
        'height': _int(data.get('height')),
        'model': tags.get('source'),
        'app': 'NovelAI',
    }
    return result

# -- InvokeAI: 'invokeai_metadata' (v3+) or 'sd-metadata' (v2) ----------------

def strategy_invokeai(tags: Dict[str, str]) -> Optional[Dict[str, Any]]:
    data = _json(tags.get('invokeaimetadata'))
    if isinstance(data, dict):
        model = data.get('model')
        result: Dict[str, Any] = {
            'prompt': data.get('positive_prompt'),
            'negative_prompt': data.get('negative_prompt'),
            'seed': _seed(data.get('seed')),
            'steps': _int(data.get('steps')),
            'cfg': _float(data.get('cfg_scale')),
            'scheduler': data.get('scheduler'),
            'width': _int(data.get('width')),
            'height': _int(data.get('height')),
            'model': (model.get('name') or model.get('model_name')) if isinstance(model, dict) else model,
            'app': 'InvokeAI',
        }
        loras = []
        for item in data.get('loras') or []:
            if isinstance(item, dict):
                ref = item.get('model') or item.get('lora') or {}
                name = ref.get('name') or ref.get('model_name') if isinstance(ref, dict) else ref
                loras.append((str(name or ''), _float(item.get('weight'))))
        _set_loras(result, loras)
        return result

    legacy = _json(tags.get('sdmetadata'))
    if isinstance(legacy, dict) and isinstance(legacy.get('image'), dict):
        img = legacy['image']
        prompt = img.get('prompt')
        if isinstance(prompt, list):
            prompt = ' '.join(p.get('prompt', '') for p in prompt if isinstance(p, dict))
        return {
            'prompt': prompt,
            'seed': _seed(img.get('seed')),
            'steps': _int(img.get('steps')),
            'cfg': _float(img.get('cfg_scale')),
            'sampler_name': img.get('sampler'),
            'width': _int(img.get('width')),
            'height': _int(img.get('height')),
            'model': legacy.get('model_weights'),
            'app': 'InvokeAI',
        }
    return None

# -- Draw Things: XMP exif:UserComment holds JSON with short keys -------------

def strategy_drawthings(tags: Dict[str, str]) -> Optional[Dict[str, Any]]:
    data = _json(tags.get('usercomment'))
    if not isinstance(data, dict) or 'c' not in data:
        return None
    result: Dict[str, Any] = {
        'prompt': data.get('c'),
        'negative_prompt': data.get('uc'),
        'seed': _seed(data.get('seed')),
        'steps': _int(data.get('steps')),
        'cfg': _float(data.get('scale')),
        'sampler_name': data.get('sampler'),
        'model': data.get('model'),
        'app': 'DrawThings',
    }
    result['width'], result['height'] = _size(data.get('size'))
    loras = [(str(l.get('file', '')), _float(l.get('weight')))
             for l in data.get('lora') or [] if isinstance(l, dict)]
    _set_loras(result, loras)
    return result


SYNC_STRATEGIES: List[Callable[[Dict[str, str]], Optional[Dict[str, Any]]]] = [
    strategy_a1111,
    strategy_fooocus,
    strategy_novelai,
    strategy_invokeai,
    strategy_drawthings,
]

# ============================================================================
# Orchestration
# ============================================================================

def _clean(result: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in result.items() if v not in (None, '', [])}
    for k in ('prompt', 'negative_prompt'):
        if isinstance(out.get(k), str):
            out[k] = out[k].strip()
    if 'model' in out:
        out['model'] = model_name(out['model'])
    return out


def merge_candidates(cands: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Best-scoring candidate wins; the rest only fill gaps. All candidates
    come from the same file, so gap-filling cannot mix two images."""
    cands = sorted(cands, key=lambda c: result_quality(c[1]), reverse=True)
    best_name, best = cands[0]
    merged = dict(best)
    merged['strategy'] = best_name
    for _, other in cands[1:]:
        for k in SCALAR_FIELDS:
            if merged.get(k) in (None, '') and other.get(k) not in (None, ''):
                merged[k] = other[k]
        if 'lora1' not in merged and 'lora1' in other:
            for k, v in other.items():
                if re.match(r'^lora\d+(_strength)?$', k) or k == 'loras':
                    merged[k] = v
    return merged


async def extract_from_tags(raw_tags: Dict[str, Any]) -> Dict[str, Any]:
    """Returns {'metadata': dict|None, 'workflow': dict|None,
    'promptflow': dict|None, 'candidates': [(strategy, score), ...]}."""
    tags = normalize_tags(raw_tags)
    cands: List[Tuple[str, Dict[str, Any]]] = []
    workflow = promptflow = None

    try:
        comfy = await strategy_comfyui(tags)
    except Exception:
        logger.exception('strategy_comfyui failed')
        comfy = None
    if comfy is not None:
        workflow, promptflow = comfy.pop('_workflow', None), comfy.pop('_promptflow', None)
        cands.append(('comfyui', _clean(comfy)))

    for strategy in SYNC_STRATEGIES:
        try:
            r = strategy(tags)
        except Exception:
            logger.exception('%s failed', strategy.__name__)
            continue
        if r:
            cands.append((strategy.__name__.removeprefix('strategy_'), _clean(r)))

    scored = [(name, result_quality(r)) for name, r in cands]
    if not cands:
        return {'metadata': None, 'workflow': workflow, 'promptflow': promptflow, 'candidates': []}

    merged = merge_candidates(cands)
    metadata = merged if result_quality(merged) >= MIN_QUALITY else None
    return {'metadata': metadata, 'workflow': workflow, 'promptflow': promptflow, 'candidates': scored}


async def extract_from_file(path: str) -> Dict[str, Any]:
    return await extract_from_tags(read_file_tags(path))


if __name__ == '__main__':                      # python EmbeddedMetadata.py image.png ...
    import asyncio
    import sys
    for p in sys.argv[1:]:
        out = asyncio.run(extract_from_file(p))
        print(f'== {p}  candidates={out["candidates"]}')
        print(json.dumps(out['metadata'], indent=2, default=str))

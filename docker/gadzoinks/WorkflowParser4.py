# WorkflowParser2.py
import json
import os
import re
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__file__)

# ── known valid values for sanity checking ──────────────────────────────────
KNOWN_SAMPLERS = {
    'euler', 'euler_cfg_pp', 'euler_ancestral', 'euler_ancestral_cfg_pp',
    'heun', 'heun_pp2', 'dpm_2', 'dpm_2_ancestral', 'lms',
    'dpm_fast', 'dpm_adaptive', 'dpmpp_2s_ancestral', 'dpmpp_sde',
    'dpmpp_sde_gpu', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
    'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ipndm',
    'ipndm_v', 'deis', 'ddim', 'uni_pc', 'uni_pc_bh2',
    'res_multistep', 'res_3s_ode', 'res_momentumized',
}
KNOWN_SCHEDULERS = {
    'normal', 'karras', 'exponential', 'sgm_uniform', 'simple',
    'ddim_uniform', 'beta', 'linear_quadratic', 'kl_optimal',
    'beta57', 'cosine',
}
TEXT_VALUE_NODES = {
    'PrimitiveStringMultiline',
    'PrimitiveString', 
    'String Literal',
    'Text Multiline',
    'Text to String',
    'CR Text',
    'ShowText|pysssss',
    'StringFunction|pysssss',
}
def is_bad_prompt(text: Any) -> bool:
    """Detect node references slipping through as prompts e.g. '59 0'."""
    if not text or not isinstance(text, str):
        return True
    t = text.strip()
    if not t:
        return True
    # looks like "NNN M" — a node ref
    if re.match(r'^\d+\s+\d+$', t):
        return True
    # too short to be a real prompt
    if len(t) < 5:
        return True
    return False

def is_bad_seed(val: Any) -> bool:
    if val is None or val == '' or val == 'None':
        return True
    try:
        v = int(str(val))
        return v <= 0
    except:
        return True

def is_bad_steps(val: Any) -> bool:
    if val is None:
        return True
    try:
        v = int(val)
        return not (1 <= v <= 200)
    except (TypeError, ValueError):
        return True          # catches stray ref strings like "238:243"

def is_bad_cfg(val: Any) -> bool:
    if val is None:
        return True
    try:
        v = float(val)
        return not (0.1 <= v <= 50)
    except (TypeError, ValueError):
        return True

def result_quality(r: Dict) -> int:
    """Score a parse result — higher is better."""
    score = 0
    if not is_bad_prompt(r.get('prompt')):  score += 3
    if not is_bad_seed(r.get('seed')):      score += 2
    if not is_bad_steps(r.get('steps')):    score += 1
    if not is_bad_cfg(r.get('cfg')):        score += 1
    if r.get('model'):                       score += 1
    return score

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_last_part(text: Any) -> str:
    if not text or not isinstance(text, str):
        return str(text) if text else ""
    name = os.path.splitext(text)[0]
    parts = name.replace('\\', '/').split('/')
    return parts[-1] if parts else ""

def is_ref(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)

def resolve_ref(ref: Any, pf: Dict) -> Optional[Dict]:
    if not is_ref(ref):
        return None
    return pf.get(str(ref[0]))

# ── text tracing ─────────────────────────────────────────────────────────────

ZERO_NODES = {'ConditioningZeroOut', 'ConditioningSetTimestepRange'}
PASS_NODES = {
    'ConditioningCombine', 'ConditioningAverage', 'ConditioningConcat',
    'ConditioningSetArea', 'ConditioningSetMask', 'FluxGuidance',
}
TEXT_PASSTHROUGH_NODES = {
    'Text to String', 'Text Multiline', 'StringFunction|pysssss',
    'CR Text', 'ShowText|pysssss',
}

PRIMITIVE_NODES = {
    'PrimitiveInt':     'value',
    'PrimitiveFloat':   'value',
    'PrimitiveBoolean': 'value',
    'PrimitiveString':  'value',
}
SYSTEM_PROMPT_PATTERNS = [
    r'You are an assistant designed to generate.*?<Prompt Start>\s*',
    r'<Prompt Start>\s*',
    r'<system>.*?</system>\s*',
]

def _extract_real_prompt(text: str) -> Optional[str]:
    """Strip system prompt prefixes from concatenated strings."""
    result = text
    for pattern in SYSTEM_PROMPT_PATTERNS:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.DOTALL)
    result = result.strip()
    return result if not is_bad_prompt(result) else None

def trace_primitive(ref: Any, pf: Dict, visited: set = None) -> Any:
    """Resolve a ref to a concrete primitive value, following switch nodes."""
    if visited is None:
        visited = set()
    node = resolve_ref(ref, pf)
    if not node:
        return None
    node_id = str(ref[0])
    if node_id in visited:
        return None
    visited.add(node_id)

    ct = node.get('class_type', '')
    inputs = node.get('inputs', {})

    # direct primitive
    if ct in PRIMITIVE_NODES:
        return inputs.get(PRIMITIVE_NODES[ct])

    # switch node — resolve which branch is active
    if ct == 'ComfySwitchNode':
        use_true = False  # default
        switch = inputs.get('switch')
        if isinstance(switch, bool):
            use_true = switch
        elif isinstance(switch, (int, float)):
            use_true = bool(switch)
        elif is_ref(switch):
            switch_node = resolve_ref(switch, pf)
            if switch_node:
                val = switch_node.get('inputs', {}).get('value')
                if isinstance(val, bool):
                    use_true = val

        preferred, fallback = ('on_true', 'on_false') if use_true else ('on_false', 'on_true')
        for key in (preferred, fallback):
            v = inputs.get(key)
            if is_ref(v):
                result = trace_primitive(v, pf, visited)
                if result is not None:
                    return result

    return None
def trace_text(ref: Any, pf: Dict, visited: set = None) -> Optional[str]:
    if visited is None:
        visited = set()
    node = resolve_ref(ref, pf)
    if not node:
        return None
    node_id = str(ref[0])
    if node_id in visited:
        return None
    visited.add(node_id)

    ct = node.get('class_type', '')
    inputs = node.get('inputs', {})
    print(f"trace_text ref:{trace_text} ct:{ct} inputs:{inputs}")

    if ct in ZERO_NODES:
        return None

    if ct == 'CLIPTextEncode':
        text = inputs.get('text')
        if isinstance(text, str) and not is_bad_prompt(text):
            return text
        if is_ref(text):
            return trace_text(text, pf, visited)

    if ct == 'GPrompts':
        computed = (node.get('_meta', {}).get('computed_prompt') or
                    inputs.get('computed_prompt'))
        if computed and isinstance(computed, str):
            return computed
        text = inputs.get('text')
        if isinstance(text, str) and not is_bad_prompt(text):
            return text
    if ct in ('StringConcatenate', 'CR Text Concatenate', 'StringJoin'):
        delimiter = inputs.get('delimiter', '')
        parts = []
        for key in ('string_a', 'string_b', 'text1', 'text2', 'string1', 'string2'):
            v = inputs.get(key)
            if isinstance(v, str) and not is_bad_prompt(v):
                parts.append(v)
            elif is_ref(v):
                result = trace_text(v, pf, visited)
                if result:
                    parts.append(result)
        if parts:
            combined = delimiter.join(parts)
            # filter out system prompt prefixes — keep only the actual user prompt
            return _extract_real_prompt(combined)




    if ct in TEXT_VALUE_NODES:
        for key in ('value', 'string', 'text', 'text_a'):
            v = inputs.get(key)
            if isinstance(v, str) and not is_bad_prompt(v):
                return v
            if is_ref(v):
                result = trace_text(v, pf, visited)
                if result:
                    return result


    if ct in TEXT_PASSTHROUGH_NODES:
        text = inputs.get('text') or inputs.get('text_a')
        if isinstance(text, str) and not is_bad_prompt(text):
            return text
        if is_ref(text):
            return trace_text(text, pf, visited)
    if ct in PASS_NODES:
        for key in ('conditioning', 'conditioning_1', 'positive'):
            v = inputs.get(key)
            if is_ref(v):
                return trace_text(v, pf, visited)

    if ct == 'PrimitiveStringMultiline':
        for key in ('value', 'text', 'string'):
            v = inputs.get(key)
            if isinstance(v, str) and not is_bad_prompt(v):
                return v
        return None
    if ct == 'ComfySwitchNode':
        # resolve switch value — may be a direct bool or a ref to PrimitiveBoolean
        use_true = True  # default
        switch = inputs.get('switch')
        if isinstance(switch, bool):
            use_true = switch
        elif isinstance(switch, (int, float)):
            use_true = bool(switch)
        elif is_ref(switch):
            switch_node = resolve_ref(switch, pf)
            if switch_node:
                val = switch_node.get('inputs', {}).get('value')
                if isinstance(val, bool):
                    use_true = val

        preferred, fallback = ('on_true', 'on_false') if use_true else ('on_false', 'on_true')
        for key in (preferred, fallback):
            v = inputs.get(key)
            if is_ref(v):
                result = trace_text(v, pf, visited)
            if result:
                return result
        return None

    if ct == 'TextGenerate':
        # LLM output is runtime-only; no stored text to trace
        return None
    # generic text fallback
    for key in ('text', 'string', 'prompt'):
        v = inputs.get(key)
        if isinstance(v, str) and not is_bad_prompt(v):
            return v
        if is_ref(v):
            result = trace_text(v, pf, visited)
            if result:
                return result

    return None

# ── model tracing ────────────────────────────────────────────────────────────

LORA_NODES = {
    'LoraLoader', 'LoraLoaderModelOnly',
    'Power Lora Loader (rgthree)', 'Lora Loader Stack (rgthree)',
    'CR Apply LoRA Stack', 'CR LoRA Stack',
}

def trace_model(ref: Any, pf: Dict, visited: set = None) -> Optional[str]:
    if visited is None:
        visited = set()
    node = resolve_ref(ref, pf)
    if not node:
        return None
    node_id = str(ref[0])
    if node_id in visited:
        return None
    visited.add(node_id)

    ct = node.get('class_type', '')
    inputs = node.get('inputs', {})

    if ct in ('CheckpointLoaderSimple', 'CheckpointLoader'):
        return inputs.get('ckpt_name')
    if ct in ('UNETLoader', 'UnetLoaderGGUF'):
        return inputs.get('unet_name')
    if ct in LORA_NODES:
        upstream = inputs.get('model')
        if is_ref(upstream):
            return trace_model(upstream, pf, visited)
    if ct in ('ModelSamplingFlux', 'ModelSamplingAuraFlow', 'ModelSamplingDiscrete',
               'LatentUpscaleBy', 'LatentUpscale'):
        upstream = inputs.get('model')
        if is_ref(upstream):
            return trace_model(upstream, pf, visited)

    upstream = inputs.get('model')
    if is_ref(upstream):
        return trace_model(upstream, pf, visited)
    return None

# ── dimension tracing ─────────────────────────────────────────────────────────

LATENT_NODES = {
    'EmptyLatentImage', 'EmptySD3LatentImage',
    'EmptyFlux2LatentImage', 'EmptyFluxLatentImage',
    'EmptyHunyuanLatentVideo',
}

def trace_dimensions(ref: Any, pf: Dict, visited: set = None) -> Tuple[Optional[int], Optional[int]]:
    if visited is None:
        visited = set()
    node = resolve_ref(ref, pf)
    if not node:
        return None, None
    node_id = str(ref[0])
    if node_id in visited:
        return None, None
    visited.add(node_id)

    ct = node.get('class_type', '')
    inputs = node.get('inputs', {})

    if ct in LATENT_NODES or 'EmptyLatent' in ct:
        return inputs.get('width'), inputs.get('height')

    for key in ('latent_image', 'samples', 'pixels'):
        v = inputs.get(key)
        if is_ref(v):
            w, h = trace_dimensions(v, pf, visited)
            if w:
                return w, h

    return None, None

# ── sampler-linked node tracing ───────────────────────────────────────────────

def trace_sampler_name(ref: Any, pf: Dict) -> Optional[str]:
    """Trace sampler ref → KSamplerSelect → sampler_name."""
    node = resolve_ref(ref, pf)
    if not node:
        return None
    ct = node.get('class_type', '')
    inputs = node.get('inputs', {})
    if ct == 'KSamplerSelect':
        name = inputs.get('sampler_name')
        if isinstance(name, str) and name in KNOWN_SAMPLERS:
            return name
    return inputs.get('sampler_name')

def trace_steps_scheduler(ref: Any, pf: Dict) -> Tuple[Optional[int], Optional[str]]:
    """Trace sigmas ref → BasicScheduler / KarrasScheduler → steps, scheduler."""
    node = resolve_ref(ref, pf)
    if not node:
        return None, None
    ct = node.get('class_type', '')
    inputs = node.get('inputs', {})

    if ct in ('BasicScheduler', 'KarrasScheduler', 'ExponentialScheduler',
               'PolyexponentialScheduler', 'VPScheduler',
               'SDTurboScheduler', 'Flux2Scheduler', 'AysScheduler',
               'GoldenNoiseScheduler'):
        steps = inputs.get('steps')
        sched = inputs.get('scheduler')
        return steps, sched

    # some schedulers pass through
    upstream = inputs.get('sigmas') or inputs.get('scheduler')
    if is_ref(upstream):
        return trace_steps_scheduler(upstream, pf)

    return None, None

def trace_cfg(ref: Any, pf: Dict) -> Optional[float]:
    """Trace guider ref → CFGGuider / FluxGuidance → cfg value."""
    node = resolve_ref(ref, pf)
    if not node:
        return None
    ct = node.get('class_type', '')
    inputs = node.get('inputs', {})

    if ct == 'CFGGuider':
        cfg = inputs.get('cfg')
        if cfg is not None and not is_bad_cfg(cfg):
            return cfg

    if ct == 'FluxGuidance':
        # guidance stored as 'guidance' widget
        guidance = inputs.get('guidance')
        if guidance is not None:
            try:
                return float(guidance)
            except:
                pass

    if ct == 'BasicGuider':
        # no cfg — return None
        return None

    # pass through
    upstream = inputs.get('guider')
    if is_ref(upstream):
        return trace_cfg(upstream, pf)
    return None

def trace_seed(ref: Any, pf: Dict) -> Optional[str]:
    """Trace noise ref → RandomNoise / GPrompts → seed."""
    node = resolve_ref(ref, pf)
    if not node:
        return None
    ct = node.get('class_type', '')
    inputs = node.get('inputs', {})

    if ct == 'RandomNoise':
        seed = inputs.get('noise_seed') or inputs.get('seed')
        if seed is not None:
            if is_ref(seed):
                print(f"trace_seed: RandomNoise seed is a ref {seed}, cannot resolve")
                return None
            try:
                return str(int(seed))
            except (TypeError, ValueError) as e:
                print(f"trace_seed: RandomNoise could not convert seed={seed!r}: {e}")
                return None

    if ct == 'GPrompts':
        seed = inputs.get('seed')
        if seed is not None:
            if is_ref(seed):
                print(f"trace_seed: GPrompts seed is a ref {seed}, cannot resolve")
                return None
            try:
                return str(int(seed))
            except (TypeError, ValueError) as e:
                print(f"trace_seed: GPrompts could not convert seed={seed!r}: {e}")
                return None
    print(f"trace_seed: could not parse ct:{ct}")

    return None


# ── LoRA extraction ───────────────────────────────────────────────────────────

def extract_loras(pf: Dict) -> Dict[str, Any]:
    result = {}
    count = 1

    for node_id, node in pf.items():
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type', '')
        inputs = node.get('inputs', {})

        if ct == 'LoraLoader':
            name = inputs.get('lora_name')
            if name and name != 'None':
                result[f'lora{count}'] = extract_last_part(name)
                result[f'lora{count}_strength'] = inputs.get('strength_model', 1.0)
                count += 1

        elif ct == 'LoraLoaderModelOnly':
            name = inputs.get('lora_name')
            if name and name != 'None':
                result[f'lora{count}'] = extract_last_part(name)
                result[f'lora{count}_strength'] = inputs.get('strength_model', 1.0)
                count += 1

        elif ct == 'Power Lora Loader (rgthree)':
            for key, val in inputs.items():
                if key.startswith('lora_') and isinstance(val, dict) and val.get('on'):
                    name = val.get('lora', '')
                    strength = val.get('strength', 1.0)
                    if name and strength > 0:
                        result[f'lora{count}'] = extract_last_part(name)
                        result[f'lora{count}_strength'] = strength
                        count += 1

        elif ct == 'Lora Loader Stack (rgthree)':
            for i in range(1, 9):
                name = inputs.get(f'lora_0{i}')
                if name and name != 'None':
                    result[f'lora{count}'] = extract_last_part(name)
                    result[f'lora{count}_strength'] = inputs.get(f'strength_0{i}', 1.0)
                    count += 1

        elif ct in ('CR Apply LoRA Stack', 'CR LoRA Stack'):
            # loras stored as stack ref — skip for now, hard to trace
            pass

    return result

# ── main promptflow parser ────────────────────────────────────────────────────

def parse_promptflow(pf: Dict) -> Optional[Dict[str, Any]]:
    """Parse ComfyUI promptflow (API/export format with named inputs)."""
    if not isinstance(pf, dict):
        return None

    result = {}

    # find sampler node
    sampler = None
    sampler_type = None
    for node_id, node in pf.items():
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type', '')
        if ct in ('KSampler', 'KSamplerAdvanced', 'SamplerCustomAdvanced'):
            sampler = node
            sampler_type = ct
            break

    if sampler:
        inputs = sampler.get('inputs', {})

        if sampler_type in ('KSampler', 'KSamplerAdvanced'):
            # seed — may be direct or ref
            seed_val = inputs.get('seed') or inputs.get('noise_seed')
            if is_ref(seed_val):
                result['seed'] = trace_seed(seed_val, pf)
            elif seed_val is not None:
                result['seed'] = str(int(seed_val))
            steps_val = inputs.get('steps')
            if is_ref(steps_val):
                result['steps'] = trace_primitive(steps_val, pf)
            else:
                result['steps'] = steps_val
            cfg_val = inputs.get('cfg')
            if is_ref(cfg_val):
                result['cfg'] = trace_primitive(cfg_val, pf)
            else:
                result['cfg'] = cfg_val
            result['sampler_name'] = inputs.get('sampler_name')
            result['scheduler']    = inputs.get('scheduler')

            pos = inputs.get('positive')
            if is_ref(pos):
                result['prompt'] = trace_text(pos, pf)

            neg = inputs.get('negative')
            if is_ref(neg):
                result['negative_prompt'] = trace_text(neg, pf)

            latent = inputs.get('latent_image')
            if is_ref(latent):
                w, h = trace_dimensions(latent, pf)
                if w: result['width'] = w
                if h: result['height'] = h

            model_ref = inputs.get('model')
            if is_ref(model_ref):
                m = trace_model(model_ref, pf)
                if m: result['model'] = extract_last_part(m)

        elif sampler_type == 'SamplerCustomAdvanced':
            # seed via noise input → RandomNoise
            noise_ref = inputs.get('noise')
            if is_ref(noise_ref):
                result['seed'] = trace_seed(noise_ref, pf)

            # steps + scheduler via sigmas input
            sigmas_ref = inputs.get('sigmas')
            if is_ref(sigmas_ref):
                steps, scheduler = trace_steps_scheduler(sigmas_ref, pf)
                if steps:    result['steps'] = steps
                if scheduler: result['scheduler'] = scheduler

            # cfg via guider input
            guider_ref = inputs.get('guider')
            if is_ref(guider_ref):
                cfg = trace_cfg(guider_ref, pf)
                if cfg is not None: result['cfg'] = cfg

                # positive/negative from CFGGuider
                guider_node = resolve_ref(guider_ref, pf)
                if guider_node:
                    g_inputs = guider_node.get('inputs', {})
                    pos = g_inputs.get('positive')
                    if is_ref(pos):
                        result['prompt'] = trace_text(pos, pf)
                    neg = g_inputs.get('negative')
                    if is_ref(neg):
                        result['negative_prompt'] = trace_text(neg, pf)

            # sampler_name via sampler input → KSamplerSelect
            sampler_ref = inputs.get('sampler')
            if is_ref(sampler_ref):
                name = trace_sampler_name(sampler_ref, pf)
                if name: result['sampler_name'] = name

            # dimensions via latent_image
            latent = inputs.get('latent_image')
            if is_ref(latent):
                w, h = trace_dimensions(latent, pf)
                if w: result['width'] = w
                if h: result['height'] = h

            # model via guider → CFGGuider → model
            if is_ref(guider_ref):
                guider_node = resolve_ref(guider_ref, pf)
                if guider_node:
                    model_ref = guider_node.get('inputs', {}).get('model')
                    if is_ref(model_ref):
                        m = trace_model(model_ref, pf)
                        if m: result['model'] = extract_last_part(m)

    # GPrompts override for prompt
    for node_id, node in pf.items():
        if isinstance(node, dict) and node.get('class_type') == 'GPrompts':
            computed = (node.get('_meta', {}).get('computed_prompt') or
                        node.get('inputs', {}).get('computed_prompt'))
            if computed:
                result['prompt'] = computed
            # seed from GPrompts if not found
            if is_bad_seed(result.get('seed')):
                seed = node.get('inputs', {}).get('seed')
                if seed is not None:
                    result['seed'] = str(int(seed))
            break

    # dimensions fallback — scan all latent nodes
    if not result.get('width'):
        for node_id, node in pf.items():
            if not isinstance(node, dict):
                continue
            ct = node.get('class_type', '')
            if ct in LATENT_NODES or 'EmptyLatent' in ct:
                inp = node.get('inputs', {})
                result.setdefault('width', inp.get('width'))
                result.setdefault('height', inp.get('height'))
                break

    # model fallback — scan loader nodes directly
    if not result.get('model'):
        for node_id, node in pf.items():
            if not isinstance(node, dict):
                continue
            ct = node.get('class_type', '')
            inp = node.get('inputs', {})
            if ct in ('CheckpointLoaderSimple', 'CheckpointLoader'):
                result['model'] = extract_last_part(inp.get('ckpt_name', ''))
                break
            if ct in ('UNETLoader', 'UnetLoaderGGUF'):
                result['model'] = extract_last_part(inp.get('unet_name', ''))
                break

    # if prompt still bad, scan all CLIPTextEncode nodes and pick longest
    if is_bad_prompt(result.get('prompt')):
        texts = []
        for node_id, node in pf.items():
            if not isinstance(node, dict):
                continue
            if node.get('class_type') == 'CLIPTextEncode':
                t = node.get('inputs', {}).get('text')
                if isinstance(t, str) and not is_bad_prompt(t):
                    texts.append(t)
                elif is_ref(t):                          # ← NEW: follow ref
                    resolved = trace_text(t, pf)
                    if resolved and not is_bad_prompt(resolved):
                        texts.append(resolved)
        if texts:
            result['prompt'] = max(texts, key=len)

    # LoRAs
    loras = extract_loras(pf)
    if loras:
        result['loras'] = loras
        result.update(loras)

    if result:
        result['app'] = 'ComfyUI'
        result['parser'] = 'promptflow_parser'

    return result if result else None

# ── UI workflow → promptflow converter ────────────────────────────────────────

def convert_workflow_to_promptflow(workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert UI workflow (nodes list + links) to promptflow (named inputs)."""
    if not isinstance(workflow, dict) or 'nodes' not in workflow:
        return None

    # build link map: target_node_id → {target_slot → [src_node_id, src_slot]}
    link_map = {}
    for link in workflow.get('links', []):
        if len(link) < 5:
            continue
        src_node = str(link[1])
        src_slot = link[2]
        tgt_node = str(link[3])
        tgt_slot = link[4]
        link_map.setdefault(tgt_node, {})[tgt_slot] = [src_node, src_slot]

    widget_idx_map = workflow.get('widget_idx_map', {})

    WIDGET_LAYOUTS = {
        'KSampler': ['seed', 'control_after_generate', 'steps', 'cfg',
                     'sampler_name', 'scheduler', 'denoise'],
        'KSamplerAdvanced': ['add_noise', 'noise_seed', 'control_after_generate',
                              'steps', 'cfg', 'sampler_name', 'scheduler',
                              'start_at_step', 'end_at_step', 'return_with_leftover_noise'],
        'CheckpointLoaderSimple': ['ckpt_name'],
        'CheckpointLoader':       ['config_name', 'ckpt_name'],
        'UNETLoader':             ['unet_name', 'weight_dtype'],
        'UnetLoaderGGUF':         ['unet_name'],
        'VAELoader':              ['vae_name'],
        'CLIPLoader':             ['clip_name', 'type', 'device'],
        'DualCLIPLoader':         ['clip_name1', 'clip_name2', 'type'],
        'DualCLIPLoaderGGUF':     ['clip_name1', 'clip_name2', 'type'],
        'CLIPTextEncode':         ['text'],
        'EmptyLatentImage':       ['width', 'height', 'batch_size'],
        'EmptySD3LatentImage':    ['width', 'height', 'batch_size'],
        'EmptyFlux2LatentImage':  ['width', 'height', 'batch_size'],
        'EmptyFluxLatentImage':   ['width', 'height', 'batch_size'],
        'UpscaleModelLoader':     ['model_name'],
        'KSamplerSelect':         ['sampler_name'],
        'BasicScheduler':         ['scheduler', 'steps', 'denoise'],
        'KarrasScheduler':        ['sigma_max', 'sigma_min', 'rho', 'steps'],
        'CFGGuider':              ['cfg'],
        'FluxGuidance':           ['guidance'],
        'RandomNoise':            ['noise_seed', 'noise_type'],
        'ModelSamplingFlux':      ['max_shift', 'base_shift'],
        'ModelSamplingAuraFlow':  ['shift'],
        'LatentUpscaleBy':        ['method', 'scale_by'],
        'Text Multiline':         ['text'],
        'Text to String':         ['text'],
    }

    promptflow = {}

    for node in workflow.get('nodes', []):
        node_id   = str(node.get('id', ''))
        node_type = node.get('type', '')
        if not node_id:
            continue

        pf_node = {
            'class_type': node_type,
            'inputs': {},
            '_meta': {'title': node.get('title', node_type)},
        }

        widgets      = node.get('widgets_values', [])
        node_inputs  = node.get('inputs', [])
        linked_slots = link_map.get(node_id, {})
        node_wmap    = widget_idx_map.get(node_id, {})
        idx_to_name  = {v: k for k, v in node_wmap.items()}
        layout       = WIDGET_LAYOUTS.get(node_type, [])

        # assign linked inputs by slot index
        for slot_idx, src in linked_slots.items():
            if slot_idx < len(node_inputs):
                name = node_inputs[slot_idx].get('name')
                if name:
                    pf_node['inputs'][name] = src

        # assign widget values for non-linked input slots
        widget_slot = 0
        for slot_idx, inp in enumerate(node_inputs):
            name = inp.get('name', '')
            if name in pf_node['inputs']:
                continue
            if slot_idx in linked_slots:
                continue
            has_widget = 'widget' in inp
            if has_widget and widget_slot < len(widgets):
                pf_node['inputs'][name] = widgets[widget_slot]
                widget_slot += 1

        # for pure-widget nodes (no inputs list) use layout or widget_idx_map
        if not node_inputs and widgets:
            if node_wmap:
                for name, idx in node_wmap.items():
                    if idx < len(widgets):
                        pf_node['inputs'][name] = widgets[idx]
            else:
                for i, val in enumerate(widgets):
                    if i < len(layout):
                        pf_node['inputs'][layout[i]] = val

        # KSampler / KSamplerAdvanced — prefer widget_idx_map, else layout
        if node_type in ('KSampler', 'KSamplerAdvanced'):
            if node_wmap:
                for name, idx in node_wmap.items():
                    if idx < len(widgets):
                        pf_node['inputs'].setdefault(name, widgets[idx])
            else:
                for i, name in enumerate(layout):
                    if i < len(widgets):
                        pf_node['inputs'].setdefault(name, widgets[i])

        # KSamplerSelect
        elif node_type == 'KSamplerSelect' and widgets:
            pf_node['inputs'].setdefault('sampler_name', widgets[0])

        # BasicScheduler — widget order: scheduler, steps, denoise
        elif node_type == 'BasicScheduler' and widgets:
            for i, name in enumerate(['scheduler', 'steps', 'denoise']):
                if i < len(widgets):
                    pf_node['inputs'].setdefault(name, widgets[i])

        # CFGGuider — widget: cfg
        elif node_type == 'CFGGuider' and widgets:
            pf_node['inputs'].setdefault('cfg', widgets[0])

        # FluxGuidance — widget: guidance
        elif node_type == 'FluxGuidance' and widgets:
            pf_node['inputs'].setdefault('guidance', widgets[0])

        # RandomNoise — widget: noise_seed
        elif node_type == 'RandomNoise' and widgets:
            pf_node['inputs'].setdefault('noise_seed', widgets[0])

        # CLIPTextEncode — text may be linked or widget
        elif node_type == 'CLIPTextEncode':
            text_slot = next(
                (i for i, inp in enumerate(node_inputs) if inp.get('name') == 'text'),
                None)
            if text_slot is not None and text_slot in linked_slots:
                pf_node['inputs']['text'] = linked_slots[text_slot]
            elif 'text' not in pf_node['inputs'] and widgets:
                pf_node['inputs']['text'] = widgets[0]

        # Lora Loader Stack — alternating name/strength
        elif node_type == 'Lora Loader Stack (rgthree)' and widgets:
            i, n = 0, 1
            while i + 1 < len(widgets):
                pf_node['inputs'][f'lora_0{n}']      = widgets[i]
                pf_node['inputs'][f'strength_0{n}']  = widgets[i + 1]
                n += 1; i += 2

        # GPrompts
        elif node_type == 'GPrompts':
            props = node.get('properties', {})
            meta  = props.get('_meta', {})
            computed = meta.get('computed_prompt') or meta.get('computed_result')
            if computed:
                pf_node['_meta']['computed_prompt'] = computed
                pf_node['inputs']['computed_prompt'] = computed
            if widgets:
                pf_node['inputs'].setdefault('text', widgets[0])
            if len(widgets) > 1:
                pf_node['inputs'].setdefault('seed', widgets[1])

        # Text pass-through nodes — last widget is computed value
        elif node_type in TEXT_PASSTHROUGH_NODES and widgets:
            pf_node['inputs']['text'] = widgets[-1]

        # generic loader nodes using layout
        elif layout and not node_inputs:
            for i, name in enumerate(layout):
                if i < len(widgets):
                    pf_node['inputs'].setdefault(name, widgets[i])

        promptflow[node_id] = pf_node

    return promptflow if promptflow else None

# ── async entry point ─────────────────────────────────────────────────────────

async def parse_comfyui_workflow_async(
    workflow:   Union[str, Dict, None] = None,
    promptflow: Union[str, Dict, None] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse ComfyUI metadata.
    Priority:
      1. provided promptflow (API export format — most reliable)
      2. convert UI workflow to promptflow and parse
    Validates result quality and logs warnings for partial results.
    """
    loop = asyncio.get_event_loop()
    best = None

    # 1. parse provided promptflow
    if promptflow:
        pf = promptflow if isinstance(promptflow, dict) else json.loads(promptflow)
        result = await loop.run_in_executor(None, parse_promptflow, pf)
        if result:
            best = result
            if result_quality(result) >= 5:
                return result

    # 2. convert UI workflow to promptflow and parse
    if workflow:
        wf = workflow if isinstance(workflow, dict) else json.loads(workflow)
        print(f"parse_comfyui_workflow_async *** try workflow.")
        converted = await loop.run_in_executor(None, convert_workflow_to_promptflow, wf)
        if converted:
            print(f"parse_comfyui_workflow_async *** try workflow.  converted:{converted}")
            result = await loop.run_in_executor(None, parse_promptflow, converted)
            if result:
                q = result_quality(result)
                if best is None or q > result_quality(best):
                    best = result
                if q >= 5:
                    return result

    if best:
        q = result_quality(best)
        if q < 3:
            print(f"WARNING: low quality parse result (score={q}): {best}")
        return best

    return None


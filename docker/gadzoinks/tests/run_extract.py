# Usage: python -I run_extract.py <dir with EmbeddedMetadata.py> /tmp/gz_samples/*
# Runs each file through exiftool (what Immich sends) and Pillow; needs exiftool on PATH.
import asyncio, json, subprocess, sys
sys.path.insert(0, sys.argv[1])
import contextlib, io
from EmbeddedMetadata import extract_from_tags, read_file_tags
FIELDS = ('app','strategy','prompt','negative_prompt','seed','steps','cfg','sampler_name','scheduler','width','height','model','lora1','lora1_strength')
for path in sys.argv[2:]:
    exif = json.loads(subprocess.check_output(['exiftool','-j',path]))[0]
    for label, tags in (('exiftool', exif), ('pillow', read_file_tags(path))):
        with contextlib.redirect_stdout(io.StringIO()):      # WorkflowParser4 prints a lot
            out = asyncio.run(extract_from_tags(tags))
        m = out['metadata']
        print(f"== {path.split('/')[-1]:24} [{label}] cands={out['candidates']} wf={'Y' if out['workflow'] else '-'} pf={'Y' if out['promptflow'] else '-'}")
        if m:
            print('   ' + ' | '.join(f"{k}={str(m[k])[:40]}" for k in FIELDS if k in m))
        else:
            print('   (no metadata)')

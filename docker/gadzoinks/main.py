# gadzoinks/main.py
import json
import re
import psycopg2
import psycopg2.extras
import os
import traceback
import requests as req
from fastapi import FastAPI, Request,status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from WorkflowParser4 import parse_comfyui_workflow_async

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this later
    allow_methods=["*"],
    allow_headers=["*"],
)

def dprint(s):
    print(s)
    pass
def apply_tags(asset_id: str, tags: list, cur):
    """Write tags to XMP sidecar via Immich API."""
    immich_url = os.environ.get('IMMICH_URL', 'http://localhost:2281')
    immich_key = os.environ.get('IMMICH_API_KEY', '')


    # get or create each tag, then assign to asset
    tag_ids = []
    for tag in tags:
        # upsert tag
        r = req.put(
            f"{immich_url}/api/tags",
            headers={"x-api-key": immich_key, "Content-Type": "application/json"},
            json={"value": tag}
        )
        if r.ok:
            tag_ids.append(r.json()['id'])
        else:
            print(f"Warning: could not create tag '{tag}': {r.status_code} {r.text}")

    if tag_ids:
        r = req.put(
            f"{immich_url}/api/assets/{asset_id}/tags",
            headers={"x-api-key": immich_key, "Content-Type": "application/json"},
            json={"tagIds": tag_ids}
        )
        if not r.ok:
            print(f"Warning: could not assign tags to {asset_id}: {r.status_code} {r.text}")


def apply_rating(asset_id: str, rating: int):
    """Set star rating on an asset via Immich API."""
    immich_url = os.environ.get('IMMICH_URL', 'http://localhost:2281')
    immich_key = os.environ.get('IMMICH_API_KEY', '')

    r = req.put(
        f"{immich_url}/api/assets",
        headers={"x-api-key": immich_key, "Content-Type": "application/json"},
        json={"ids": [asset_id], "rating": rating}
    )
    if not r.ok:
        print(f"Warning: could not set rating on {asset_id}: {r.status_code} {r.text}")

def coerce_bigint(val):
    """Convert value to int or None for bigint columns."""
    if val is None or val == '':
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

@app.on_event("startup")
async def startup():
    ensure_schema()

@app.post("/gz/test")
async def test(request: Request):
    body = await request.json()
    asset_id = body.get('asset_id', 'unknown')
    workflow = body.get('workflow', {})
    print(f"Got asset ID: {asset_id}")
    print(f"Got workflow: {workflow}")
    print(f"JSON: {body}")
    return {"status": "ok", "asset_id": asset_id}

@app.post("/gz/ingest")
async def ingest(request: Request):
    body = await request.json()
    asset_id = body.get('asset_id', 'unknown')
    info = body.get('info', {})
    print(f"Got info payload:")
    print( json.dumps(info, indent=4))
    workflow = info.get('workflow', {})  # UI format
    promptflow = info.get('promptflow', {}) # API format
    computed_prompt = info.get('computed_prompt', {})
    source_host = info.get('source_host')
    #TODO I should remove tags and ratings, and rely on base immich api
    tags        = info.get('tags', [])        # list of strings e.g. ["ai/lora/extr3m3hair", "beach"]
    rating      = info.get('rating')    
    metadata = await parse_comfyui_workflow_async(workflow,promptflow)
    if not metadata:
        # TODO what should i do for bad exit ?
        if workflow or promptflow:
            store_workflow('unknown',asset_id,workflow , promptflow)
        return JSONResponse( status_code=201,content={"status": "ok", "asset_id": asset_id})
    if source_host:
        metadata['source_host'] = source_host
    if computed_prompt:
        metadata['prompt'] = computed_prompt
    #print(f"Got asset ID: {asset_id}")
    #print(f"Got workflow: {workflow}")
    #print(f"JSON: {body}")
    print(f"metadata: \n{json.dumps(metadata,indent=4)}")

    conn = get_connection()
    try:
        cur = conn.cursor()
        print("a");
        parse_and_store( cur, asset_id, metadata,workflow,promptflow )
        print("b")
        conn.commit()
        """
        if tags:
            apply_tags(asset_id, tags, cur)
        if rating is not None:
            apply_rating(asset_id, rating)
        """
        cur.close()
    except Exception as e:
        conn.rollback()
        print(f"Error storing metadata: {e}")
        print(traceback.format_exc())
        return JSONResponse( status_code=500, content={"status": "error","asset_id": asset_id, "error": str(e)})
    finally:
        conn.close()
    
    return JSONResponse( status_code=201,content={"status": "ok", "asset_id": asset_id})

@app.get("/gz/assets/{asset_id}/meta")
async def get_meta(asset_id: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                m.*,
                a.name as architecture_name,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'name', l.name,
                            'filename', l.filename,
                            'strength', al.strength
                        )
                    ) FILTER (WHERE l.id IS NOT NULL),
                    '[]'
                ) as loras
            FROM gz_asset_metadata m
            LEFT JOIN gz_architecture a ON a.id = m.architecture_id
            LEFT JOIN gz_asset_lora al ON al.asset_id = m.asset_id
            LEFT JOIN gz_lora l ON l.id = al.lora_id
            WHERE m.asset_id = %s
            GROUP BY m.asset_id, a.name
        """, [asset_id])
        row = cur.fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return dict(row)
    finally:
        cur.close()
        conn.close()

@app.get("/gz/assets/{asset_id}/workflow")
async def get_workflow(asset_id: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT workflow FROM gz_workflow WHERE asset_id = %s", [asset_id])
        row = cur.fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return dict(row)
    finally:
        cur.close()
        conn.close()

@app.get("/gz/assets/{asset_id}/promptflow")
async def get_promptflow(asset_id: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT promptflow FROM gz_promptflow WHERE asset_id = %s", [asset_id])
        row = cur.fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return dict(row)
    finally:
        cur.close()
        conn.close()

@app.post("/gz/search")
async def search_assets(request: Request):
    body = await request.json()
    prompt       = body.get('prompt')
    model        = body.get('model')
    lora         = body.get('lora')
    architecture = body.get('architecture')
    asset_ids    = body.get('asset_ids')
    page         = body.get('page', 1)
    size         = body.get('size', 50)
    offset       = (page - 1) * size

    conn = get_connection()
    cur = conn.cursor()
    try:
        conditions = []
        params = []

        if asset_ids is not None:
            conditions.append("m.asset_id = ANY(%s::uuid[])")
            params.append(asset_ids)

        if prompt:
            conditions.append("m.prompt ILIKE %s")
            params.append(f'%{prompt}%')

        if model:
            conditions.append("m.model ILIKE %s")
            params.append(f'%{model}%')

        if architecture:
            conditions.append("a.name ILIKE %s")
            params.append(f'%{architecture}%')

        needs_lora_join = bool(lora)
        lora_join = """
            LEFT JOIN gz_asset_lora al ON al.asset_id = m.asset_id
            LEFT JOIN gz_lora l ON l.id = al.lora_id
        """ if needs_lora_join else ""

        # always join architecture since we may filter on it
        arch_join = """
            LEFT JOIN gz_architecture a ON a.id = m.architecture_id
        """ if architecture else ""

        if lora:
            conditions.append("l.name ILIKE %s")
            params.append(f'%{lora}%')

        where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''


        # get total count for pagination
        cur.execute(f"""
            SELECT COUNT(DISTINCT m.asset_id)
            FROM gz_asset_metadata m
            {arch_join}
            {lora_join}
            {where}
        """, params)
        total = cur.fetchone()['count']

        # get page of IDs
        cur.execute(f"""
            SELECT DISTINCT m.asset_id::text
            FROM gz_asset_metadata m
            {arch_join}
            {lora_join}
            {where}
            LIMIT %s OFFSET %s
        """, params + [size, offset])

        ids = [r['asset_id'] for r in cur.fetchall()]
        has_next = (offset + size) < total

        return {
            "asset_ids": ids,
            "total": total,
            "page": page,
            "size": size,
            "next_page": page + 1 if has_next else None,
        }
    finally:
        cur.close()
        conn.close()

@app.get("/gz/architectures")
async def get_architectures():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DISTINCT a.name 
            FROM gz_architecture a
            JOIN gz_asset_metadata m ON m.architecture_id = a.id
            WHERE a.name IS NOT NULL
            ORDER BY a.name
        """)
        return [r['name'] for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

@app.get("/gz/models")
async def get_models():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DISTINCT model
            FROM gz_asset_metadata
            WHERE model IS NOT NULL
            ORDER BY model
        """)
        return [r['model'] for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

@app.get("/gz/loras")
async def get_loras():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DISTINCT name
            FROM gz_lora
            WHERE name IS NOT NULL
            ORDER BY name
        """)
        return [r['name'] for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()
def get_connection():
    return psycopg2.connect(
        host     = "database",  # docker
        port     = os.environ.get('DB_PORT', 5432),
        dbname   = os.environ.get('DB_DATABASE_NAME', 'immich'),
        user     = os.environ.get('DB_USERNAME'),
        password = os.environ.get('DB_PASSWORD'),
        cursor_factory = psycopg2.extras.RealDictCursor
    )

def ensure_schema():
    """Run on startup. Creates tables if they don't exist, safe to run repeatedly."""
    global SCHEMA_SQL
    dprint("Checking gz schema...")
    try:
        conn = get_connection()
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute(SCHEMA_SQL)
        conn.commit()
        
        # report what we have
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'gz_%'
            ORDER BY table_name
        """)
        tables = [r['table_name'] for r in cur.fetchall()]
        dprint(f"gz tables present: {tables}")
        
        cur.close()
        conn.close()
        dprint("Schema OK")
    except Exception as e:
        print(f"Schema check failed: {e}")
        raise   # fail fast on startup if DB unreachable

def store_workflow(app,asset_id,workflow , promptflow):
    if workflow or promptflow:
        conn = get_connection()
        try:
            cur = conn.cursor()
            # minimal metadata row so foreign key for workflow table is satisfied
            cur.execute("""
                INSERT INTO gz_asset_metadata (asset_id, app)
                VALUES (%s, %s)
                ON CONFLICT (asset_id) DO NOTHING
            """, [asset_id, 'ComfyUI'])
            
            if workflow:
                cur.execute("""
                    INSERT INTO gz_workflow (asset_id, workflow)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (asset_id) DO UPDATE SET workflow=EXCLUDED.workflow
                """, [asset_id, json.dumps(workflow)])
            
            if promptflow:
                cur.execute("""
                    INSERT INTO gz_promptflow (asset_id, promptflow)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (asset_id) DO UPDATE SET promptflow=EXCLUDED.promptflow
                """, [asset_id, json.dumps(promptflow)])
            
            conn.commit()
            cur.close()
        except Exception as e:
            conn.rollback()
            print(f"Error storing workflow for {asset_id}: {e}")
            print(traceback.format_exc())
        finally:
            conn.close()

def parse_and_store(cur, asset_id: str, meta: dict, workflow: dict, promptflow: dict):
    # 1. insert core metadata
    cur.execute("""
        INSERT INTO gz_asset_metadata
            (asset_id, prompt, negative_prompt, model, seed, steps, cfg,
             sampler_name, width, height, app,source_host,scheduler)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (asset_id) DO UPDATE SET
            prompt=EXCLUDED.prompt, model=EXCLUDED.model
    """, [
        asset_id,
        meta.get('prompt'),
        meta.get('negative_prompt'),
        meta.get('model'),
        coerce_bigint(meta.get('seed')),
        coerce_bigint(meta.get('steps')),
        meta.get('cfg'),
        meta.get('sampler_name'),
        coerce_bigint(meta.get('width')),
        coerce_bigint(meta.get('height')),
        meta.get('app'),
        meta.get('source_host'),
        meta.get('scheduler'),
        ])

    # 2. parse loras from the messy lora1/lora2/lora3 structure
    loras = []
    i = 1
    while f'lora{i}' in meta:
        filename = meta[f'lora{i}']
        strength = meta.get(f'lora{i}_strength', 1.0)
        # derive display name: 'extr3m3hair_V3.safetensors' → 'extr3m3hair'
        # name = re.sub(r'[_\-]?(v\d+)?\.safetensors$', '', filename, flags=re.IGNORECASE)
        # extr3m3hair_V3.safetensors -> extr3m3hair_V3
        name = re.sub(r'\.safetensors$', '', filename, flags=re.IGNORECASE)
        loras.append((filename, name, strength))
        i += 1

    # 3. upsert loras into dictionary, then link to asset
    for filename, name, strength in loras:
        cur.execute("""
            INSERT INTO gz_lora (filename, name)
            VALUES (%s, %s)
            ON CONFLICT (filename) DO NOTHING
        """, [filename, name])
        cur.execute("SELECT id FROM gz_lora WHERE filename = %s", [filename])
        lora_id = cur.fetchone()['id']
        #lora_id = cur.fetchone(
        #    "SELECT id FROM gz_lora WHERE filename = %s", [filename]
        #)['id']

        cur.execute("""
            INSERT INTO gz_asset_lora (asset_id, lora_id, strength)
            VALUES (%s, %s, %s)
            ON CONFLICT (asset_id, lora_id) DO UPDATE SET strength=EXCLUDED.strength
        """, [asset_id, lora_id, strength])
    # store workflow
    if workflow:
        cur.execute("""
		    INSERT INTO gz_workflow (asset_id, workflow)
		    VALUES (%s, %s::jsonb)
		    ON CONFLICT (asset_id) DO UPDATE SET workflow=EXCLUDED.workflow
		""", [asset_id, json.dumps(workflow)])
    # store promptflow
    if promptflow:
        cur.execute("""
		    INSERT INTO gz_promptflow (asset_id, promptflow)
		    VALUES (%s, %s::jsonb)
		    ON CONFLICT (asset_id) DO UPDATE SET promptflow=EXCLUDED.promptflow
        """, [asset_id, json.dumps(promptflow)])

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gz_architecture (
    id          SERIAL PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    vendor_name TEXT
);

INSERT INTO gz_architecture (name, vendor_name) VALUES
    ('flux.1-dev',      'Black Forest Labs'),
    ('flux.1-schnell',  'Black Forest Labs'),
    ('flux.2-klein',    'Black Forest Labs'),
    ('illustrious',     'OnomaAI'),
    ('sdxl',            'Stability AI'),
    ('sd1.5',           'Stability AI'),
    ('sd3',             'Stability AI'),
    ('sd3.5',           'Stability AI'),
    ('kandinsky',       'Sber AI__AI Forever'),
    ('qwen-image',      'Alibaba__QwenLM'),
    ('z-image',         'Chroma Studio'),
    ('longcat',         'Meituan'),
    ('chroma',          NULL),              -- based on flux.1-schnell, independent release
    ('hidream',         'HiDream.ai__Vivago'),
    ('omnigen',         'BAAI__VectorSpaceLab'),
    ('ovis',            NULL),              -- Ovis-U1, vendor unclear
    ('lotus',           NULL),              -- uncertain which Lotus model you mean
    ('netayume',        NULL),              -- anime fine-tune, CivitAI community
    ('newbie',          NULL),              -- CivitAI community fine-tune
    ('capybara',        NULL)               
ON CONFLICT (name) DO NOTHING;

CREATE TABLE IF NOT EXISTS gz_asset_metadata (
    asset_id        UUID PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    prompt          TEXT,
    negative_prompt TEXT,
    model           TEXT,
    architecture_id INTEGER REFERENCES gz_architecture(id),
    seed            BIGINT,
    steps           INTEGER,
    cfg             FLOAT,
    sampler_name    TEXT,
    scheduler_name    TEXT,
    width           INTEGER,
    height          INTEGER,
    app             TEXT,
    scheduler       TEXT,
    source_host     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gz_lora (
    id              SERIAL PRIMARY KEY,
    filename        TEXT UNIQUE NOT NULL,
    name            TEXT,
    architecture_id INTEGER REFERENCES gz_architecture(id)
);

CREATE INDEX IF NOT EXISTS gz_asset_metadata_model           ON gz_asset_metadata(model);
CREATE INDEX IF NOT EXISTS gz_asset_metadata_architecture_id ON gz_asset_metadata(architecture_id);
CREATE INDEX IF NOT EXISTS gz_asset_metadata_prompt_trgm ON gz_asset_metadata USING gin(prompt gin_trgm_ops);
CREATE INDEX IF NOT EXISTS gz_asset_metadata_model_trgm ON gz_asset_metadata USING gin(model gin_trgm_ops);
CREATE INDEX IF NOT EXISTS gz_lora_name_trgm ON gz_lora USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS gz_architecture_name_trgm ON gz_architecture USING gin(name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS gz_lora_architecture_id ON gz_lora(architecture_id);

CREATE TABLE IF NOT EXISTS gz_asset_lora (
    asset_id    UUID    REFERENCES asset(id)  ON DELETE CASCADE,
    lora_id     INTEGER REFERENCES gz_lora(id),
    strength    FLOAT,
    PRIMARY KEY (asset_id, lora_id)
);

CREATE INDEX IF NOT EXISTS gz_asset_lora_lora_id ON gz_asset_lora(lora_id);
CREATE TABLE IF NOT EXISTS gz_workflow (
    asset_id    UUID PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    workflow    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gz_promptflow (
    asset_id    UUID PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    promptflow  JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""



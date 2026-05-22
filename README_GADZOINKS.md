# Gadzoinks — ComfyUI × Immich Metadata Bridge

A set of custom ComfyUI nodes and Immich patches. A ComfyUI node saves images directly into Immich. With the Immich patches applied, generation metadata is stored in the Immich DB and surfaces in the asset viewer — displayed and searchable.

---

## What it does

Images generated in ComfyUI are uploaded directly to Immich. On a patched Immich instance, the full generation metadata — prompt, model, seed, sampler, steps, CFG, LoRA — is stored in the database, shown in the asset viewer, and available as search filters.

**Pipeline:**

```
GPrompts → String Formatter → Sampler → Save to Immich
```

---

## Custom Nodes — comfyui_gprompts

Install via **ComfyUI Manager**, or manually from:

- Registry: https://registry.comfy.org/nodes/gprompts
- GitHub: https://github.com/GadzoinksOfficial/comfyui_gprompts

The `comfyui_gprompts` node pack contains all three nodes. Only this pack needs to be installed in `custom_nodes`.

---

### GPrompts Node

Dynamic prompt expansion with two variant syntaxes.

| Syntax | Behaviour |
|--------|-----------|
| `{{ a \| b \| c }}` | **Sequential** — cycles through options in order across generations |
| `{ a \| b \| c }` | **Random** — picks one variant per generation |

**Example:**

```
8k raw photo of a {{ pigeon | monkey | cat }}
sitting on top of a traffic light showing { red | green | yellow },
cinematic lighting
```

Run 1 → `8k raw photo of a pigeon sitting on top of a traffic light showing green, cinematic lighting`  
Run 2 → `8k raw photo of a monkey sitting on top of a traffic light showing red, cinematic lighting`  
Run 3 → `8k raw photo of a cat sitting on top of a traffic light showing green, cinematic lighting`

Outputs: `text`, `computed_prompt`, `dynprompt`, `seed`

---

### String Formatter Node

Multi-input string assembly with up to 8 named inputs (`a` through `h`). Supports built-in variables:

| Variable | Value |
|----------|-------|
| `$datetime` | Current date and time |
| `$hostname` | Machine hostname |
| `$os` | Operating system |

Output feeds into the Save to Immich node's notes field.

---

### Save to Immich Node

Uploads the generated image directly to your Immich instance via the normal Immich API. On patched Immich, generation metadata is stored in the Immich DB for display and searching.

- Configurable `filename_prefix`, `album`, and `tags`
- Standard Immich — image saved, metadata not stored
- Patched Immich — metadata posted to `/gz` endpoints, stored in DB, displayed and searchable
- Optionally saves a local copy alongside the upload

---

## Vanilla vs Patched Immich

| Feature | Standard Immich | Gadzoinks Immich |
|---------|:-:|:-:|
| Image saved to library | ✓ | ✓ |
| Album assignment | ✓ | ✓ |
| Tags applied | ✓ | ✓ |
| Prompt stored | ✗ | ✓ |
| Model / seed / sampler stored | ✗ | ✓ |
| AI Generation panel in viewer | ✗ | ✓ |
| Download Workflow button | ✗ | ✓ |
| XMP sidecar written | ✗ | ✓ |
| Search by prompt, model, LoRA | ✗ | ✓ |

---

## Asset Viewer — Metadata Panel

![Immich asset viewer showing Gadzoinks AI Generation metadata panel](images/image1.png)

---

## Search — AI Generation Filters

Immich search extended with AI Generation fields: prompt text, architecture, model, and LoRA.

![Immich search dialog showing AI Generation filter fields](images/image2.png)

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│  ComfyUI                                            │
│  ┌─────────────────────────────────────────────┐   │
│  │  comfyui_gprompts node pack                 │   │
│  │  GPrompts · String Formatter · Save to Immich│   │
│  └─────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP to Immich host
                        ▼
┌─────────────────────────────────────────────────────┐
│  Caddy (reverse proxy)                              │
│  /gz/*  ──────────────────► Gadzoinks server        │
│  everything else  ────────► Immich (unchanged)      │
└───────┬───────────────────────────────┬─────────────┘
        │ /gz REST calls                │ standard Immich API
        ▼                               ▼
┌───────────────────┐        ┌──────────────────────┐
│  Gadzoinks server │        │  Immich               │
│  (Docker)         │        │  (Docker, unmodified) │
│  FastAPI          │        └──────────────────────┘
│  metadata ingest  │
│  XMP sidecar      │
└───────┬───────────┘
        │ SQL write
        ▼
┌─────────────────────────────────────────────────────┐
│  Postgres                                           │
│  assets (Immich)  ·  gz_generation_metadata         │
└───────────────────────┬─────────────────────────────┘
                        │ read on panel open
                        ▼
┌─────────────────────────────────────────────────────┐
│  Immich Web (patched frontend)                      │
│  Asset Viewer  ·  Searching  ·  AI Generation panel │
│  Download Workflow button                           │
└─────────────────────────────────────────────────────┘
```

Install via ComfyUI Manager · [registry.comfy.org/nodes/gprompts](https://registry.comfy.org/nodes/gprompts) · [github.com/GadzoinksOfficial/comfyui_gprompts](https://github.com/GadzoinksOfficial/comfyui_gprompts)

---

## Components

### docker-compose.gz.yml
- Adds the Gadzoinks FastAPI server as a Docker service alongside the Immich stack
- Shares the Immich Postgres instance; mounts the Immich upload volume for sidecar writing
- Adds Caddy as the reverse proxy service

### Caddyfile
- `/gz/*` routed to the Gadzoinks FastAPI server container
- All other requests forwarded to the standard Immich server unchanged

### gz-server/main.py
- FastAPI app handling `POST /gz/metadata` — receives and stores generation metadata
- Writes metadata to a dedicated Postgres table keyed by Immich asset ID
- Writes XMP sidecar file into the Immich upload directory for the asset

### gz-server/schema.sql
- Postgres table `gz_generation_metadata` linked to Immich `assets` by asset ID
- Columns: `prompt`, `model`, `seed`, `steps`, `cfg`, `sampler`, `generated_on`, `workflow_json`

### immich-patch/AIGenerationDetail.svelte
- New Svelte component added to the Immich asset detail panel
- Fetches metadata from `GET /gz/metadata/:assetId` and renders it in the viewer
- "Download Workflow" button — retrieves and downloads the stored workflow JSON
- Only rendered when a metadata record exists for the asset

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE)


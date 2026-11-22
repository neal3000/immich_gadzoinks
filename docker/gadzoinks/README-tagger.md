# Automatic `ai/` tagging in the gadzoinks sidecar

Files to drop into `docker/gadzoinks/`:

| file | what it is |
|---|---|
| `gz_tagger.py` | Postgres state, the pass, the schema |
| `gz_vocabulary.py` | applies the vocabulary: drop / rename / fold / fallback |
| `gz_taxonomy.py` | morphology rules, stop list, policy blocklist |
| `gz_prompts.py` | prompt parsing and term matching |
| `gz_taggers.py` | WD / Danbooru ONNX inference |
| `gz_vocabulary.yaml` | the vocabulary — 1,430 terms |
| `Dockerfile` | adds onnxruntime, pillow, numpy, pyyaml, and the model volume |

Plus the four edits in `main_py_patch.md`.

Off by default. `GZ_TAG_ENABLED=1` turns it on.

## What changed from the standalone worker

**Postgres instead of SQLite**, in Immich's own database under `gz_tag_*`,
alongside `gz_asset_metadata`. Three consequences:

- Your existing backup covers it.
- `REFERENCES asset(id) ON DELETE CASCADE` on every table, matching your
  convention. The SQLite version leaked prediction rows for deleted assets
  forever; these clean themselves up.
- **Finding work is a query, not an API walk.** The old worker paged the
  whole library over HTTP every five minutes to find new assets. Now it is a
  `LEFT JOIN ... IS NULL` over an indexed column.

**Prompts are a `SELECT`.** `gz_asset_metadata.prompt` is already there, so
the per-asset HTTP call is gone — that was 50,000 requests on a backfill.
`negative_prompt` sits in the same row and is never read: it names what was
*excluded*, so mining it would tag an image with precisely what is absent.

**Module names are prefixed.** A plain `vocabulary.py` was shadowed by a
PyPI package of the same name once already; in a flat app directory that
collision is silent and permanent.

**Three asset classes are excluded in SQL** rather than discovered by a
failing fetch: trashed (`deletedAt`), non-images, and `visibility = 'locked'`
— Immich's locked folder. Tagging those would surface private content in the
tag tree, which is the one place it must not appear. `hidden` is the video
half of a live photo and has nothing to tag.

## Scheduling

A plain thread started from FastAPI's startup hook, ticking every
`GZ_TAG_INTERVAL` seconds. A thread rather than an async task because ONNX
inference is CPU-bound: in the event loop it would stall every `/gz/` request
for the length of a pass. ONNX Runtime releases the GIL during inference, so
the API stays responsive.

`run_pass` holds a non-blocking lock, so a pass that overruns the interval is
skipped rather than stacked.

**The model is loaded lazily and released when a pass finds nothing.** Most
ticks on a settled library have no work and never touch it; a permanently
resident gigabyte would be rent paid all day on a box also running Immich and
Postgres. `GZ_TAG_PER_PASS` (default 500) caps one tick, so a cold library is
tagged across several passes instead of one eight-hour block.

At roughly 1.7 images/second on CPU, 50,000 assets is about eight hours
spread over whatever number of ticks that takes.

## Editing the vocabulary

Each asset records which vocabulary version it was last mapped against —
a content hash of the file, so no one has to bump a number. Change
`gz_vocabulary.yaml`, restart the container, and the next tick re-maps the
affected assets from predictions already in the database. Seconds, not hours.

A **threshold** change is different: it leaves no trace in the hash, so
nothing notices. Clear `mapped_vocab` to force a pass:

```sql
UPDATE gz_tag_asset SET mapped_vocab = NULL;
```

## Prompt boosting

`GZ_TAG_PROMPT_BOOST=0.17` lets a term the generation prompt names clear a
threshold 0.17 lower — about 0.2 instead of 0.371 — but never below
`GZ_TAG_PROMPT_FLOOR` (0.15). A term the model produced nothing for is still
never tagged. The prompt is a prior, the model is the evidence, and a tag
needs both.

On "woman riding a flying dragon over a beach, buddha statues in background":
`dragon` clears on its own, `beach` at 0.24 and `statue` at 0.22 do not, and
the prompt is what promotes them. A `spacecraft` prediction at 0.26 still
fails, because nothing asked for one.

## It only ever adds

Tags already written stay if you later edit the vocabulary so a path is no
longer produced. A background job that strips tags from thousands of assets
because a config file changed is not something to ship. To undo a pass,
delete the `ai` tag subtree in Immich.

## Checking on it

```
curl localhost:2289/gz/tagger/status
curl -X POST localhost:2289/gz/tagger/run     # nudge, don't wait for the timer
```

`GZ_TAG_DRY_RUN=1` runs a full pass and logs what it would write, writing
nothing. Worth one tick before the first real one.

`gave_up` in the status payload counts assets past the retry limit — if that
is not zero, query `gz_tag_deferred.reason`.

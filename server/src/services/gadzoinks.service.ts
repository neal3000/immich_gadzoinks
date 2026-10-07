import { Injectable } from '@nestjs/common';
import { OnEvent } from 'src/decorators';
import { ArgOf } from 'src/repositories/event.repository';
import { BaseService } from 'src/services/base.service';

/**
 * Gadzoinks: forward AI-generation metadata embedded in imported files to the
 * gadzoinks sidecar (POST /gz/ingest_embedded), which parses and stores it.
 *
 * Hooks AssetMetadataExtracted, so it runs for every path into Immich: web and
 * mobile upload, CLI, external library scans, and admin re-runs of the
 * Extract Metadata job (which is how existing assets get backfilled).
 * Lives in its own file so rebases on upstream only touch services/index.ts.
 */

const GADZOINKS_URL = process.env.GADZOINKS_URL; // e.g. http://gadzoinks:2289; unset = disabled
const TIMEOUT_MS = 10_000;

// Tags that can carry generation metadata, compared lowercase-alphanumeric
// ('PNG:Prompt' -> 'prompt', 'Invokeai_metadata' -> 'invokeaimetadata').
const FORWARD_TAGS = new Set([
  'prompt', // ComfyUI API graph (PNG)
  'workflow', // ComfyUI UI graph (PNG)
  'parameters', // A1111 / Forge / Fooocus (PNG)
  'usercomment', // A1111 JPEG, Draw Things XMP
  'comment', // NovelAI settings JSON
  'description', // NovelAI prompt
  'imagedescription',
  'make', // ComfyUI WebP: 'workflow:{...}'
  'model', // ComfyUI WebP: 'prompt:{...}'
  'software',
  'source',
  'invokeaimetadata',
  'sdmetadata',
]);
const STRONG_TAGS = new Set(['prompt', 'workflow', 'parameters', 'invokeaimetadata', 'sdmetadata']);

const normKey = (key: string) =>
  key
    .split(':')
    .pop()!
    .toLowerCase()
    .replaceAll(/[^a-z0-9]/g, '');

/** Cheap check so ordinary camera photos never reach the sidecar. */
const looksGenerated = (tags: Record<string, string>) =>
  Object.entries(tags).some(
    ([key, value]) =>
      STRONG_TAGS.has(normKey(key)) ||
      /^\s*(prompt|workflow)\s*:\s*\{/i.test(value) || // ComfyUI WebP
      value.includes('Steps:') || // A1111 settings line
      /^\s*\{"(c|uc|prompt|steps)"/.test(value), // Draw Things / NovelAI JSON
  );

@Injectable()
export class GadzoinksService extends BaseService {
  @OnEvent({ name: 'AssetMetadataExtracted' })
  async onAssetMetadataExtracted({ assetId, source }: ArgOf<'AssetMetadataExtracted'>) {
    // sidecar-write re-extracts after Immich edits its own XMP; the file's
    // embedded generation data hasn't changed
    if (!GADZOINKS_URL || source === 'sidecar-write') {
      return;
    }

    const asset = await this.assetRepository.getById(assetId);
    if (!asset) {
      return;
    }

    // Second exiftool read of the same file. handleMetadataExtraction doesn't
    // pass its tags along, and re-reading keeps metadata.service.ts unpatched.
    const allTags = (await this.metadataRepository.readTags(asset.originalPath)) as Record<string, unknown>;
    const tags: Record<string, string> = {};
    for (const [key, value] of Object.entries(allTags)) {
      if (typeof value === 'string' && value.length > 0 && FORWARD_TAGS.has(normKey(key))) {
        tags[key] = value;
      }
    }
    if (!looksGenerated(tags)) {
      return;
    }

    try {
      const response = await fetch(`${GADZOINKS_URL}/gz/ingest_embedded`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ asset_id: assetId, tags }),
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
      if (!response.ok) {
        this.logger.warn(`gadzoinks ingest_embedded ${assetId}: HTTP ${response.status} ${await response.text()}`);
      }
    } catch (error) {
      // never fail Immich's own pipeline because the sidecar is down
      this.logger.warn(`gadzoinks ingest_embedded ${assetId} failed: ${error}`);
    }
  }
}

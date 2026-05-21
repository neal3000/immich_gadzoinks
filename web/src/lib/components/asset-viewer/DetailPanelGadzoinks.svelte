<script lang="ts">
  import { Text } from '@immich/ui';
  import type { AssetResponseDto } from '@immich/sdk';

  interface Props {
    asset: AssetResponseDto;
  }

  let { asset }: Props = $props();

  let meta: Record<string, unknown> | null = $state(null);
  let loading = $state(true);
  let error = $state(false);

  // fetch whenever asset changes
  $effect(() => {
    const id = asset.id;
    loading = true;
    error = false;
    meta = null;

    fetch(`/gz/assets/${id}/meta`)
      .then((r) => {
        if (!r.ok) throw new Error('not found');
        return r.json();
      })
      .then((data) => {
        meta = data;
        loading = false;
      })
      .catch(() => {
        error = true;
        loading = false;
      });
  });

const downloadWorkflow = async () => {
  const response = await fetch(`/gz/assets/${asset.id}/workflow`);
  if (!response.ok) return;
  const data = await response.json();
  const blob = new Blob([JSON.stringify(data.workflow, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${asset.originalFileName ?? asset.id}.workflow.json`;
  a.click();
  URL.revokeObjectURL(url);
};

</script>
<div>
 Gadzoinks Details Test
</div>
{#if loading}
  <!-- show nothing while loading, same as tags behaviour -->
{:else if meta}
  <section class="px-4 mt-4">
    <div class="flex h-10 w-full items-center justify-between text-sm">
      <Text color="muted">AI Generation</Text>
    </div>

    <div class="text-sm space-y-2 pb-4">
      {#if meta.prompt}
        <div>
          <span class="text-gray-900 text-xs">Prompt</span>
          <p class="text-xs mt-1 break-words">{meta.prompt}</p>
        </div>
      {/if}

      {#if meta.model}
        <div class="flex justify-between">
          <span class="text-gray-900 text-xs">Model</span>
          <span class="text-xs">{meta.model}</span>
        </div>
      {/if}

      {#if meta.seed}
        <div class="flex justify-between">
          <span class="text-gray-900 text-xs">Seed</span>
          <span class="text-xs font-mono">{meta.seed}</span>
        </div>
      {/if}

      {#if meta.steps}
        <div class="flex justify-between">
          <span class="text-gray-900 text-xs">Steps</span>
          <span class="text-xs">{meta.steps}</span>
        </div>
      {/if}

      {#if meta.cfg}
        <div class="flex justify-between">
          <span class="text-gray-900 text-xs">CFG</span>
          <span class="text-xs">{meta.cfg}</span>
        </div>
      {/if}

      {#if meta.sampler_name}
        <div class="flex justify-between">
          <span class="text-gray-900 text-xs">Sampler</span>
          <span class="text-xs">{meta.sampler_name}</span>
        </div>
      {/if}
	{#if meta.scheduler}
	  <div class="flex justify-between">
	    <span class="text-gray-900 dark:text-gray-100 text-xs">Scheduler</span>
	    <span class="text-xs">{meta.scheduler}</span>
	  </div>
	{/if}
	{#if meta.loras && (meta.loras as Array<{name: string, filename: string, strength: number}>).length > 0}
	  <div>
	    <span class="text-gray-900 text-xs">LoRAs</span>
	    <div class="mt-1 space-y-1 pl-3" >
	      {#each meta.loras as Array<{name: string, filename: string, strength: number}> as lora}
		<div class="flex justify-between items-center">
		  <span class="text-xs truncate max-w-[70%]">{lora.name}</span>
		  <span class="text-xs font-mono text-gray-900">{lora.strength}</span>
		</div>
	      {/each}
	    </div>
	  </div>
	{/if}
      {#if meta.source_host}
        <div class="flex justify-between">
          <span class="text-gray-900 text-xs">Generated On</span>
          <span class="text-xs">{meta.source_host}</span>
        </div>
      {/if}

    </div>
    <div class="flex justify-end pt-2">
	  <button
	    onclick={downloadWorkflow}
	    class="text-xs px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white"
	  >
	    Download workflow
	  </button>
    </div>

  </section>
{/if}


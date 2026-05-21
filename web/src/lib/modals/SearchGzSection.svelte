<script lang="ts">
  import { onMount } from 'svelte';
  import { t } from 'svelte-i18n';

  interface Props {
    gz: { prompt?: string; model?: string; lora?: string };
  }

  let { gz = $bindable() }: Props = $props();

  let models: string[] = $state([]);
  let loras: string[] = $state([]);

  onMount(async () => {
    const [m, l] = await Promise.all([
      fetch('/gz/models').then(r => r.json()),
      fetch('/gz/loras').then(r => r.json()),
    ]);
    models = m;
    loras = l;
  });
</script>

<div class="flex flex-col gap-3">
  <p class="text-sm font-medium dark:text-white">AI Generation</p>

  <label class="text-xs dark:text-gray-300">Prompt contains
    <input
      type="text"
      class="w-full mt-1 rounded-lg border px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white"
      placeholder="woman on beach..."
      bind:value={gz.prompt}
    />
  </label>

  <label class="text-xs dark:text-gray-300">Model
    <select
      class="w-full mt-1 rounded-lg border px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white"
      bind:value={gz.model}
    >
      <option value="">Any</option>
      {#each models as model}
        <option value={model}>{model}</option>
      {/each}
    </select>
  </label>

  <label class="text-xs dark:text-gray-300">LoRA
    <select
      class="w-full mt-1 rounded-lg border px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white"
      bind:value={gz.lora}
    >
      <option value="">Any</option>
      {#each loras as lora}
        <option value={lora}>{lora}</option>
      {/each}
    </select>
  </label>
</div>


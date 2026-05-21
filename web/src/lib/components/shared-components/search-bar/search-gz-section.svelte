<script lang="ts">
  import { onMount } from 'svelte';

  interface GzFilter {
    prompt?: string;
    model?: string;
    lora?: string;
    architecture?: string;
  }

  interface Props {
    gz: GzFilter;
  }

  let { gz = $bindable() }: Props = $props();

  let models: string[] = $state([]);
  let loras: string[] = $state([]);
  let architectures: string[] = $state([]);

  onMount(async () => {
    const [m, l, a] = await Promise.all([
      fetch('/gz/models').then(r => r.json()),
      fetch('/gz/loras').then(r => r.json()),
      fetch('/gz/architectures').then(r => r.json()),
    ]);
    models = m;
    loras = l;
    architectures = a;
  });
</script>

<section class="flex flex-col gap-4">
  <p class="text-sm font-medium dark:text-white">AI Generation</p>

  <!-- Prompt -->
  <div class="flex flex-col gap-1">
    <label class="text-xs text-gray-500 dark:text-gray-400" for="gz-prompt">Prompt contains</label>
    <input
      id="gz-prompt"
      type="text"
      class="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white"
      placeholder="woman on beach, voluminous hair..."
      bind:value={gz.prompt}
    />
  </div>

  <!-- Architecture -->
  <div class="flex flex-col gap-1">
    <label class="text-xs text-gray-500 dark:text-gray-400" for="gz-architecture">Architecture</label>
    <select
      id="gz-architecture"
      class="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white"
      bind:value={gz.architecture}
    >
      <option value="">Any</option>
      {#each architectures as arch}
        <option value={arch}>{arch}</option>
      {/each}
    </select>
  </div>

  <!-- Model -->
  <div class="flex flex-col gap-1">
    <label class="text-xs text-gray-500 dark:text-gray-400" for="gz-model">Model</label>
    <select
      id="gz-model"
      class="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white"
      bind:value={gz.model}
    >
      <option value="">Any</option>
      {#each models as model}
        <option value={model}>{model}</option>
      {/each}
    </select>
  </div>

  <!-- LoRA -->
  <div class="flex flex-col gap-1">
    <label class="text-xs text-gray-500 dark:text-gray-400" for="gz-lora">LoRA</label>
    <select
      id="gz-lora"
      class="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white"
      bind:value={gz.lora}
    >
      <option value="">Any</option>
      {#each loras as lora}
        <option value={lora}>{lora}</option>
      {/each}
    </select>
  </div>
</section>


<script lang="ts">
  import MenuOption from '$lib/components/shared-components/context-menu/menu-option.svelte';
  import { assetMultiSelectManager } from '$lib/managers/asset-multi-select-manager.svelte';
  import { handleError } from '$lib/utils/handle-error';
  import { updateAssets } from '@immich/sdk';
  import { toastManager } from '@immich/ui';
  import { mdiStar, mdiStarOutline } from '@mdi/js';
  import { t } from 'svelte-i18n';

  interface Props {
    rating: number;  // 0-5
  }

  let { rating }: Props = $props();

  const stars = ['☆☆☆☆☆', '★☆☆☆☆', '★★☆☆☆', '★★★☆☆', '★★★★☆', '★★★★★'];
  const labels = ['No stars', '1 star', '2 stars', '3 stars', '4 stars', '5 stars'];

  let text = $derived(labels[rating]);
  let icon = $derived(rating === 0 ? mdiStarOutline : mdiStar);

  const handleRate = async () => {
    try {
      const ids = assetMultiSelectManager.ownedAssets.map(({ id }) => id);
      if (ids.length === 0) return;
      await updateAssets({ assetBulkUpdateDto: { ids, rating } });
      for (const asset of assetMultiSelectManager.ownedAssets) {
        asset.rating = rating;
      }
      toastManager.primary(`Rated ${ids.length} asset${ids.length === 1 ? '' : 's'}: ${text}`);
      assetMultiSelectManager.clear();
    } catch (error) {
      handleError(error, `Unable to set rating`);
    }
  };
</script>

<MenuOption {text} {icon} onClick={handleRate} />

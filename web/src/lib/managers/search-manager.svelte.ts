import { AssetTypeEnum, AssetVisibility, type MetadataSearchDto, type SmartSearchDto } from '@immich/sdk';
import type { DateTime } from 'luxon';
import { SvelteSet } from 'svelte/reactivity';
import { goto } from '$app/navigation';
import { MediaType, QueryType, validQueryTypes } from '$lib/constants';
import { Route } from '$lib/route';
import type { SearchFilter } from '$lib/types';
import { asLocalTimeISO, parseUtcDate } from '$lib/utils/date-time';

class SearchManager {
  #filter = $state<SearchFilter>(this.#fromQuery({}));

  get filter() {
    return this.#filter;
  }

  reset() {
    this.#filter = this.#fromQuery({});
  }

  setQuery(query: MetadataSearchDto | SmartSearchDto) {
    this.#filter = this.#fromQuery(query);
  }

  async submit() {
    await goto(Route.search(this.#toQuery()));
  }

  #fromQuery(searchQuery: MetadataSearchDto | SmartSearchDto): SearchFilter {
    let query = 'query' in searchQuery && searchQuery.query ? searchQuery.query : '';
    let queryType = query ? QueryType.SMART : this.#defaultQueryType();

    if ('originalFileName' in searchQuery && searchQuery.originalFileName) {
      query = searchQuery.originalFileName;
      queryType = QueryType.METADATA;
    }

    if ('originalPath' in searchQuery && searchQuery.originalPath) {
      query = searchQuery.originalPath;
      queryType = QueryType.FULL_PATH;
    }

    if ('description' in searchQuery && searchQuery.description) {
      query = searchQuery.description;
      queryType = QueryType.DESCRIPTION;
    }

    if (searchQuery.ocr) {
      query = searchQuery.ocr;
      queryType = QueryType.OCR;
    }

    let gz: SearchFilter['gz'] = { prompt: '', model: '', lora: '', architecture: '' };
    const sq = searchQuery as any;
    if (sq.gzPrompt || sq.gzModel || sq.gzLora || sq.gzArchitecture) {
      // For AI searches the search box holds the prompt; the filter panel's prompt
      // field starts empty so the two aren't sent twice.
      query = sq.gzPrompt ?? '';
      gz = {
        prompt: '',
        model: sq.gzModel ?? '',
        lora: sq.gzLora ?? '',
        architecture: sq.gzArchitecture ?? '',
      };
      queryType = QueryType.GZ;
    }

    return {
      query,
      ocr: searchQuery.ocr,
      queryType,
      gz,
      queryAssetId: 'queryAssetId' in searchQuery ? searchQuery.queryAssetId : undefined,
      personIds: new SvelteSet('personIds' in searchQuery ? searchQuery.personIds : []),
      tagIds:
        'tagIds' in searchQuery
          ? searchQuery.tagIds === null
            ? null
            : new SvelteSet(searchQuery.tagIds)
          : new SvelteSet(),
      location: {
        country: this.#withNullAsEmptyString(searchQuery.country),
        state: this.#withNullAsEmptyString(searchQuery.state),
        city: this.#withNullAsEmptyString(searchQuery.city),
      },
      camera: {
        make: this.#withNullAsEmptyString(searchQuery.make),
        model: this.#withNullAsEmptyString(searchQuery.model),
        lensModel: this.#withNullAsEmptyString(searchQuery.lensModel),
      },
      date: {
        takenAfter: searchQuery.takenAfter ? this.#toStartOfDayDate(searchQuery.takenAfter) : undefined,
        takenBefore: searchQuery.takenBefore ? this.#toStartOfDayDate(searchQuery.takenBefore) : undefined,
      },
      display: {
        isArchive: searchQuery.visibility === AssetVisibility.Archive,
        isFavorite: searchQuery.isFavorite ?? false,
        isNotInAlbum: 'isNotInAlbum' in searchQuery ? (searchQuery.isNotInAlbum ?? false) : false,
      },
      mediaType:
        searchQuery.type === AssetTypeEnum.Image
          ? MediaType.Image
          : searchQuery.type === AssetTypeEnum.Video
            ? MediaType.Video
            : MediaType.All,
      rating: searchQuery.rating,
    };
  }

  #toQuery(): MetadataSearchDto &
    SmartSearchDto & { gzPrompt?: string; gzModel?: string; gzLora?: string; gzArchitecture?: string } {
    let type: AssetTypeEnum | undefined = undefined;
    if (this.filter.mediaType === MediaType.Image) {
      type = AssetTypeEnum.Image;
    } else if (this.filter.mediaType === MediaType.Video) {
      type = AssetTypeEnum.Video;
    }

    const query = this.filter.query || undefined;

    // AI search terms. The prompt can be typed in the search box (with the AI type
    // selected) and/or in the filter panel's "Prompt contains"; both are combined
    // (the sidecar ANDs comma-separated terms). The AI fields are sent whenever any
    // is set, so filling the panel works even if the type selector wasn't switched.
    const gz = this.filter.gz ?? { prompt: '', model: '', lora: '', architecture: '' };
    const promptParts = [this.filter.queryType === 'gz' ? query : undefined, gz.prompt]
      .map((part) => (part ?? '').trim())
      .filter((part, index, all) => part && all.indexOf(part) === index);
    const gzPrompt = promptParts.join(', ') || undefined;
    const hasGz = Boolean(gzPrompt || gz.model || gz.lora || gz.architecture);
    if (this.filter.queryType === 'gz' && !hasGz) {
      console.warn('[gz-search] AI search submitted with no prompt, model, LoRA or architecture');
    }

    return {
      query: this.filter.queryType === 'smart' ? query : undefined,
      queryAssetId: this.filter.queryAssetId || undefined,
      ocr: this.filter.queryType === 'ocr' ? query : undefined,
      originalFileName: this.filter.queryType === 'metadata' ? query : undefined,
      description: this.filter.queryType === 'description' ? query : undefined,
      originalPath: this.filter.queryType === 'fullPath' ? this.filter.query.trim() || undefined : undefined,
      country: this.#emptyStringToNull(this.filter.location.country),
      state: this.#emptyStringToNull(this.filter.location.state),
      city: this.#emptyStringToNull(this.filter.location.city),
      make: this.#emptyStringToNull(this.filter.camera.make),
      model: this.#emptyStringToNull(this.filter.camera.model),
      lensModel: this.#emptyStringToNull(this.filter.camera.lensModel),
      takenAfter: this.filter.date.takenAfter
        ? asLocalTimeISO(this.filter.date.takenAfter.startOf('day') as DateTime<true>)
        : undefined,
      takenBefore: this.filter.date.takenBefore
        ? asLocalTimeISO(this.filter.date.takenBefore.endOf('day') as DateTime<true>)
        : undefined,
      visibility: this.filter.display.isArchive ? AssetVisibility.Archive : undefined,
      isFavorite: this.filter.display.isFavorite || undefined,
      isNotInAlbum: this.filter.display.isNotInAlbum || undefined,
      personIds: this.filter.personIds.size > 0 ? [...this.filter.personIds] : undefined,
      tagIds: this.filter.tagIds === null ? null : this.filter.tagIds.size > 0 ? [...this.filter.tagIds] : undefined,
      type,
      rating: this.filter.rating,
      ...(hasGz
        ? {
            gzPrompt,
            gzModel: gz.model || undefined,
            gzLora: gz.lora || undefined,
            gzArchitecture: gz.architecture || undefined,
          }
        : {}),
    };
  }

  #withNullAsEmptyString<T>(value: T | null) {
    return value === null ? '' : value;
  }

  #emptyStringToNull(value: string | undefined) {
    return value === '' ? null : value;
  }

  #toStartOfDayDate(dateString: string) {
    return parseUtcDate(dateString)?.startOf('day') || undefined;
  }

  #defaultQueryType(): QueryType {
    const storedQueryType = localStorage.getItem('searchQueryType') as QueryType;
    return validQueryTypes.has(storedQueryType) ? storedQueryType : QueryType.SMART;
  }
}

export const searchManager = new SearchManager();

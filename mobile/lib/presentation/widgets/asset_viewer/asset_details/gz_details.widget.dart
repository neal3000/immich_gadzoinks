import 'package:flutter/material.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:immich_mobile/domain/models/asset/base_asset.model.dart';
import 'package:immich_mobile/providers/infrastructure/asset_viewer/gz_meta.provider.dart';

class GzDetails extends ConsumerWidget {
  final BaseAsset asset;
  const GzDetails({super.key, required this.asset});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final metaAsync = ref.watch(gzMetaProvider(asset.id));

    return metaAsync.when(
      loading: () => const SizedBox.shrink(),
      error: (_, __) => const SizedBox.shrink(),
      data: (meta) {
        if (meta == null) return const SizedBox.shrink();
        return _GzDetailsContent(meta: meta);
      },
    );
  }
}

class _GzDetailsContent extends StatelessWidget {
  final GzMeta meta;
  const _GzDetailsContent({required this.meta});

  Widget _row(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(label,
              style: const TextStyle(
                fontWeight: FontWeight.w500,
                fontSize: 12,
                color: Colors.grey,
              ),
            ),
          ),
          Expanded(
            child: Text(value,
              style: const TextStyle(fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Divider(),
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 8),
            child: Text('AI Generation',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
            ),
          ),
          if (meta.prompt != null)
            _row('Prompt', meta.prompt!),
          if (meta.model != null)
            _row('Model', meta.model!),
          if (meta.architectureName != null)
            _row('Architecture', meta.architectureName!),
          if (meta.seed != null)
            _row('Seed', meta.seed.toString()),
          if (meta.steps != null)
            _row('Steps', meta.steps.toString()),
          if (meta.cfg != null)
            _row('CFG', meta.cfg!.toStringAsFixed(1)),
          if (meta.samplerName != null)
            _row('Sampler', meta.samplerName!),
          if (meta.scheduler != null)
            _row('Scheduler', meta.scheduler!),
          if (meta.loras.isNotEmpty) ...[
            const SizedBox(height: 4),
            _row('LoRAs', ''),
            ...meta.loras.map((lora) => Padding(
              padding: const EdgeInsets.only(left: 100, bottom: 2),
              child: Text(
                '${lora['name']} (${(lora['strength'] as num).toStringAsFixed(2)})',
                style: const TextStyle(fontSize: 12),
              ),
            )),
          ],
        ],
      ),
    );
  }
}


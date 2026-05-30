import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'package:immich_mobile/domain/models/asset/base_asset.model.dart';
import 'package:immich_mobile/domain/models/store.model.dart';
import 'package:immich_mobile/entities/store.entity.dart';
class GzMeta {
  final String? prompt;
  final String? negativePrompt;
  final String? model;
  final String? architectureName;
  final String? samplerName;
  final String? scheduler;
  final int?    seed;
  final int?    steps;
  final double? cfg;
  final int?    width;
  final int?    height;
  final List<Map<String, dynamic>> loras;

  const GzMeta({
    this.prompt,
    this.negativePrompt,
    this.model,
    this.architectureName,
    this.samplerName,
    this.scheduler,
    this.seed,
    this.steps,
    this.cfg,
    this.width,
    this.height,
    this.loras = const [],
  });

  factory GzMeta.fromJson(Map<String, dynamic> json) {
    return GzMeta(
      prompt:           json['prompt'] as String?,
      negativePrompt:   json['negative_prompt'] as String?,
      model:            json['model'] as String?,
      architectureName: json['architecture_name'] as String?,
      samplerName:      json['sampler_name'] as String?,
      scheduler:        json['scheduler'] as String?,
      seed:             json['seed'] != null ? int.tryParse(json['seed'].toString()) : null,
      steps:            json['steps'] as int?,
      cfg:              json['cfg'] != null ? (json['cfg'] as num).toDouble() : null,
      width:            json['width'] as int?,
      height:           json['height'] as int?,
      loras:            (json['loras'] as List<dynamic>? ?? [])
                          .cast<Map<String, dynamic>>(),
    );
  }
}
final gzMetaProvider = FutureProvider.family<GzMeta?, String>((ref, assetId) async {
  try {
    final serverEndpoint = Store.get(StoreKey.serverEndpoint);
    final uri = Uri.parse('$serverEndpoint/gz/assets/$assetId/meta');
    final response = await http.get(uri).timeout(const Duration(seconds: 5));
    if (response.statusCode == 404) return null;
    if (response.statusCode != 200) return null;
    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return GzMeta.fromJson(json);
  } catch (_) {
    return null;
  }
});


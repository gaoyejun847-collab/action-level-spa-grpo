"""Merge an independently trained PRM, audit CPU INT8, and serve one frozen copy."""
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from spa_next.prm.data import JsonlDataset, sha256_file
from spa_next.prm.train import distribution


def memory_bytes():
    try:
        import psutil
        info = psutil.Process().memory_info()
        result = dict(rss_bytes=info.rss, vms_bytes=info.vms,
                      system_available_bytes=psutil.virtual_memory().available)
        if platform.system() == 'Linux':
            import resource
            result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return result
    except ImportError:
        return {}


def export(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    training, output, base = Path(args.training), Path(args.output), Path(args.model)
    report = json.loads((training / 'training_report.json').read_text())
    if not report.get('gate_passed') or report.get('smoke_only'):
        raise RuntimeError('Unqualified PRM checkpoint cannot be exported for policy scoring')
    if sha256_file(base / 'spa_snapshot.json') != report['config']['base_manifest_sha256']:
        raise RuntimeError('SFT base version differs from the PRM training base')
    from spa_next.training.sft_snapshot import verify_snapshot
    verify_snapshot(base)
    checkpoint = Path(report['best_checkpoint'])
    output.mkdir(parents=True, exist_ok=False)
    backbone = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                  torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True,
                  device_map={'': args.device})
    backbone = PeftModel.from_pretrained(backbone, checkpoint / 'adapter',
                                        device_map={'': args.device},
                                        torch_device=str(args.device)).merge_and_unload(safe_merge=True)
    backbone.eval()
    backbone.save_pretrained(output, safe_serialization=True, max_shard_size='2GB')
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(output)
    head = torch.load(checkpoint / 'head.pt', map_location='cpu', weights_only=True)
    if any(value.dtype != torch.float32 or not bool(torch.all(torch.isfinite(value))) for value in head.values()):
        raise ValueError('PRM head must contain finite FP32 parameters')
    torch.save(head, output / 'head.pt')
    shutil.copy2(training / 'training_report.json', output / 'training_report.json')
    files = {p.name: sha256_file(p) for p in sorted(output.iterdir())
             if p.is_file() and p.name not in ('prm_manifest.json', 'quantization_report.json')}
    content_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = dict(schema_version=1, kind='new-qwen-spa-prm', content_sha256=content_hash,
                    files=files, base=str(base.resolve()),
                    base_manifest_sha256=report['config']['base_manifest_sha256'],
                    best_checkpoint=str(checkpoint.resolve()), training_gate_passed=True,
                    merge_device=str(args.device),
                    training_reference=dict(kind='unmerged-bf16-sft-base-plus-prm-lora-fp32-head',
                        checkpoint=str(checkpoint.resolve()),
                        files={str(p.relative_to(checkpoint)): sha256_file(p)
                               for p in sorted(checkpoint.rglob('*')) if p.is_file() and
                               (p.suffix == '.safetensors' or p.name in ('adapter_config.json', 'head.pt'))}),
                    head='FP32 linear(vocabulary logits at complete assistant EOS -> signed scalar)',
                    supervision='trajectory sum MSE against environment terminal reward',
                    max_context=4096, frozen=True)
    (output / 'prm_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(event='PRM_EXPORTED', output=str(output.resolve()), content_sha256=content_hash)), flush=True)
    return manifest


def validate_artifact(model_path):
    path = Path(model_path)
    manifest = json.loads((path / 'prm_manifest.json').read_text())
    if manifest.get('kind') != 'new-qwen-spa-prm' or not manifest.get('training_gate_passed') or not manifest.get('frozen'):
        raise RuntimeError('PRM artifact lacks the required training qualification')
    for name, expected in manifest['files'].items():
        if sha256_file(path / name) != expected:
            raise RuntimeError('Frozen PRM artifact was modified: ' + name)
    calculated = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    if calculated != manifest['content_sha256']:
        raise RuntimeError('Invalid PRM content manifest')
    return manifest


def quantization_spec():
    import torch
    if 'fbgemm' not in torch.backends.quantized.supported_engines:
        raise RuntimeError('CPU dynamic INT8 requires the tested fbgemm backend')
    return dict(backbone='torch.ao.quantization.quantize_dynamic Linear qint8 inplace',
                head='FP32', engine='fbgemm', torch_version=str(torch.__version__),
                cpu_arch=platform.machine(), scoring='independent-exact-prefix-per-action-endpoint')


def model_version(manifest, spec):
    payload = dict(content_sha256=manifest['content_sha256'], quantization=spec)
    return 'qwen-spa-prm-int8-' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def load_frozen(model_path, *, quantized, device='cpu', cpu_threads=8):
    import torch
    from transformers import AutoModelForCausalLM
    from spa_next.prm.model import QwenSPAPRM
    if quantized:
        if device != 'cpu':
            raise ValueError('Frozen dynamic INT8 PRM scores on CPU only')
        torch.set_num_threads(cpu_threads)
        torch.backends.quantized.engine = quantization_spec()['engine']
    dtype = torch.float32 if quantized else torch.bfloat16
    backbone = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    torch_dtype=dtype, attn_implementation='sdpa', low_cpu_mem_usage=True,
                    device_map={'': device})
    backbone.config.use_cache = False
    backbone.eval()
    if quantized:
        # In-place replacement avoids cloning a full FP32 4B model during packing.
        # Embeddings and normalization stay FP32; all backbone Linear modules,
        # including lm_head, use dynamic INT8. The scalar head is added afterwards.
        backbone = torch.ao.quantization.quantize_dynamic(backbone, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    model = QwenSPAPRM(backbone)
    model.strict_prefix_scoring = bool(quantized)
    model.head.load_state_dict(torch.load(Path(model_path) / 'head.pt', map_location='cpu', weights_only=True))
    model.head.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    gc.collect()
    if quantized:
        from spa_next.training.cpu_memory import release_cpu_cache
        release_cpu_cache('prm-int8-packed')
    return model


def load_training_reference(manifest, device):
    """Use the exact qualified, unmerged checkpoint as the precision reference.

    Comparing INT8 only to an already-rounded BF16 merge could hide significant
    merge drift. This reference includes both merge and quantization error in the
    final deployment gate, relative to the model selected by validation MSE.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from spa_next.prm.model import QwenSPAPRM
    from spa_next.training.sft_snapshot import verify_snapshot
    base, reference = Path(manifest['base']), manifest['training_reference']
    checkpoint = Path(reference['checkpoint'])
    if reference.get('kind') != 'unmerged-bf16-sft-base-plus-prm-lora-fp32-head':
        raise RuntimeError('Missing qualified original-precision PRM reference')
    if sha256_file(base / 'spa_snapshot.json') != manifest['base_manifest_sha256']:
        raise RuntimeError('Original-precision PRM reference SFT snapshot changed')
    verify_snapshot(base)
    for name, expected in reference['files'].items():
        if sha256_file(checkpoint / name) != expected:
            raise RuntimeError('Original-precision PRM checkpoint changed: ' + name)
    backbone = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                   torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': device})
    backbone = PeftModel.from_pretrained(backbone, checkpoint / 'adapter', is_trainable=False,
                                        device_map={'': device}, torch_device=str(device))
    backbone.config.use_cache = False
    model = QwenSPAPRM(backbone)
    model.head.load_state_dict(torch.load(checkpoint / 'head.pt', map_location='cpu', weights_only=True))
    model.head.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model


def compare_scores(original, quantized, rewards):
    import numpy as np
    if len(original) != len(quantized) or len(rewards) != len(original) or not original:
        raise ValueError('Quantization comparison requires matching nonempty trajectories')
    errors, raw_mse, int8_mse, finite = [], 0., 0., True
    for raw, quant, reward in zip(original, quantized, rewards):
        if len(raw) != len(quant) or not raw:
            raise ValueError('PRM action counts differ after quantization')
        finite = finite and all(math.isfinite(x) for x in raw + quant)
        errors.extend(abs(a - b) for a, b in zip(raw, quant))
        raw_mse += (sum(raw) - reward) ** 2
        int8_mse += (sum(quant) - reward) ** 2
    raw_mse /= len(rewards)
    int8_mse /= len(rewards)
    mean = float(np.mean(errors))
    p95 = float(np.quantile(errors, .95))
    limit = 1.05 * raw_mse + 1e-4
    passed = finite and mean <= .02 and p95 <= .05 and int8_mse <= limit
    return dict(gate_passed=passed, finite=finite, trajectories=len(rewards), actions=len(errors),
                action_mean_absolute_difference=mean, action_absolute_difference_p95=p95,
                original_terminal_mse=raw_mse, quantized_terminal_mse=int8_mse,
                limits=dict(action_mean_absolute_difference=.02, action_absolute_difference_p95=.05,
                            quantized_terminal_mse=limit),
                action_absolute_difference_distribution=distribution(errors))


def verify(args):
    import random
    import torch
    from spa_next.prm.model import score_record
    model_path, data_path = Path(args.model), Path(args.data)
    manifest = validate_artifact(model_path)
    report = json.loads((data_path / 'data_report.json').read_text())
    if sha256_file(data_path / 'val.jsonl') != report['dataset_sha256']['val']:
        raise RuntimeError('PRM validation data changed before quantization audit')
    trained = json.loads((model_path / 'training_report.json').read_text())
    if str(data_path.resolve()) != trained['config']['data']:
        # Allow an exact immutable copy, but require both dataset hashes to match.
        original_report = Path(trained['config']['data']) / 'data_report.json'
        if not original_report.exists() or json.loads(original_report.read_text())['dataset_sha256'] != report['dataset_sha256']:
            raise RuntimeError('Quantization must use the independent PRM validation partition')
    dataset = JsonlDataset(data_path / 'val.jsonl')
    count = min(len(dataset), args.max_examples, 256)
    if count < 1:
        raise ValueError('Quantization audit needs at least one validation trajectory')
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    rows = [dataset[index] for index in indices[:count]]
    originals, quantized, timings = [], [], []
    started = time.monotonic()
    gpu = load_training_reference(manifest, device=args.device)
    with torch.inference_mode():
        for index, row in enumerate(rows):
            originals.append(score_record(gpu, row, args.device).float().cpu().tolist())
            print(json.dumps(dict(event='prm_original_precision_audit', completed=index + 1, total=count)), flush=True)
    del gpu
    gc.collect()
    torch.cuda.empty_cache()
    load_started = time.monotonic()
    cpu = load_frozen(str(model_path), quantized=True, cpu_threads=args.cpu_threads)
    load_seconds = time.monotonic() - load_started
    memory_loaded = memory_bytes()
    with (model_path / 'quantization_predictions.jsonl').open('w', buffering=1) as stream:
        with torch.inference_mode():
            for index, row in enumerate(rows):
                call_start = time.monotonic()
                scores = score_record(cpu, row, 'cpu').float().tolist()
                duration = time.monotonic() - call_start
                quantized.append(scores)
                timings.append(duration)
                entry = dict(event='prm_int8_audit', completed=index + 1, total=count,
                             trajectory_sha256=row['trajectory_sha256'], task_id=row['task_id'],
                             tokens=len(row['input_ids']), actions=len(scores), seconds=duration,
                             original_scores=originals[index], quantized_scores=scores,
                             reward=row['reward'], memory=memory_bytes())
                stream.write(json.dumps(entry) + '\n')
                print(json.dumps({key: value for key, value in entry.items()
                                  if key not in ('original_scores', 'quantized_scores')}), flush=True)
    spec = quantization_spec()
    result = compare_scores(originals, quantized, [row['reward'] for row in rows])
    result.update(model_version=model_version(manifest, spec), quantization=spec,
                  original_precision_reference=manifest['training_reference'],
                  artifact_manifest_sha256=sha256_file(model_path / 'prm_manifest.json'),
                  validation_sha256=report['dataset_sha256']['val'], cpu_threads=args.cpu_threads,
                  cpu_load_seconds=load_seconds, scoring_seconds=distribution(timings),
                  seconds_per_trajectory=sum(timings) / len(timings),
                  memory_after_load=memory_loaded, memory_after_scoring=memory_bytes(),
                  elapsed_seconds=time.monotonic() - started,
                  validation_selection_seed=42, audited_trajectory_sha256=[row['trajectory_sha256'] for row in rows])
    (model_path / 'quantization_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(event='PRM_INT8_GATE', **result), indent=2), flush=True)
    if not result['gate_passed']:
        raise RuntimeError('PRM dynamic INT8 audit failed; policy training and scoring service are blocked')
    return result


def serve(args):
    import torch
    from spa_next.prm.model import score_record, validate_inputs
    path = Path(args.model)
    manifest = validate_artifact(path)
    gpu_backend = getattr(args, 'backend', 'int8') == 'original-gpu'
    if gpu_backend:
        from spa_next.prm.gpu_backend import gpu_version
        report = json.loads((path / 'gpu_reference_report.json').read_text())
        version = gpu_version(manifest)
        device = args.device
    else:
        report = json.loads((path / 'quantization_report.json').read_text())
        version = model_version(manifest, quantization_spec())
        device = 'cpu'
    if (report.get('gate_passed') is not True or report.get('model_version') != version or
            report.get('artifact_manifest_sha256') != sha256_file(path / 'prm_manifest.json')):
        raise RuntimeError('PRM scoring refused: missing, failed or incompatible precision qualification')
    model = (load_training_reference(manifest, device) if gpu_backend else
             load_frozen(str(path), quantized=True, cpu_threads=args.cpu_threads))
    # HTTPServer, deliberately not ThreadingHTTPServer: one frozen model, one
    # scoring trajectory at a time; Ray actors never instantiate the PRM.
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.send_json(404, dict(error='unknown endpoint'))
            return self.send_json(200, dict(model_version=version, quantization_gate_passed=not gpu_backend,
                                           precision_gate_passed=True, backend='original-gpu' if gpu_backend else 'int8',
                                           frozen=True, single_process=True, pid=os.getpid(),
                                           memory=memory_bytes()))

        def do_POST(self):
            if self.path != '/score':
                return self.send_json(404, dict(error='unknown endpoint'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError('Invalid scoring request length')
                request = json.loads(self.rfile.read(size))
                if set(request) != {'model_version', 'trajectories'}:
                    raise ValueError('PRM accepts only model_version and trajectory token inputs')
                if request['model_version'] != version:
                    return self.send_json(409, dict(error='PRM model version mismatch', model_version=version))
                trajectories = request['trajectories']
                if not isinstance(trajectories, list) or not 1 <= len(trajectories) <= 256:
                    raise ValueError('Expected between 1 and 256 trajectory inputs')
            except (ValueError, TypeError, KeyError) as exc:
                return self.send_json(400, dict(error=str(exc)))
            results = []
            with torch.inference_mode():
                for row in trajectories:
                    try:
                        if not isinstance(row, dict) or set(row) != {'input_ids', 'action_end_positions'}:
                            raise ValueError('Scoring input may contain only original tokens and action endpoints')
                        validate_inputs(row['input_ids'], row['action_end_positions'],
                                        vocab_size=model.backbone.config.vocab_size)
                        started = time.monotonic()
                        scores = score_record(model, row, device).float().tolist()
                        finite = all(math.isfinite(value) for value in scores)
                        results.append(dict(prm_scores=scores if finite else [], valid=finite,
                                            error=None if finite else 'nonfinite_prm_scores'))
                        print(json.dumps(dict(event='prm_score', model_version=version,
                                              tokens=len(row['input_ids']), actions=len(scores),
                                              valid=finite, seconds=time.monotonic() - started)), flush=True)
                    except (ValueError, TypeError, KeyError) as exc:
                        results.append(dict(prm_scores=[], valid=False, error=str(exc)))
                    except Exception as exc:
                        # Inference/runtime failures are service failures, not bad
                        # predictions: trainer must stop, never silently use GRPO.
                        return self.send_json(500, dict(error='PRM scoring service failed: ' + repr(exc),
                                                        model_version=version))
            return self.send_json(200, dict(model_version=version, results=results))

    server = HTTPServer((args.host, args.port), Handler)
    print(json.dumps(dict(event='PRM_SCORING_READY', model_version=version,
                          host=args.host, port=args.port, memory=memory_bytes())), flush=True)
    server.serve_forever()

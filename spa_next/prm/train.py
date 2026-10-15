"""Independent LoRA PRM training; terminal-sum MSE with family-held-out selection."""
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

from spa_next.prm.data import JsonlDataset, sha256_file


def distribution(values):
    import numpy as np
    if not values:
        return dict(count=0)
    array = np.asarray(values, dtype=float)
    return dict(count=len(array), mean=float(array.mean()), std=float(array.std()),
                min=float(array.min()), p05=float(np.quantile(array, .05)),
                p50=float(np.quantile(array, .5)), p95=float(np.quantile(array, .95)), max=float(array.max()))


def evaluate(model, dataset, device, rank=0, world_size=1):
    import torch
    from spa_next.prm.model import score_record
    model.eval()
    totals, residuals, scores, rows = [], [], [], []
    with torch.inference_mode():
        for index in range(rank, len(dataset), world_size):
            row = dataset[index]
            predictions = score_record(model, row, device).float().cpu().tolist()
            if not all(math.isfinite(x) for x in predictions):
                raise FloatingPointError('Nonfinite PRM validation score')
            total = sum(predictions)
            delta = row['reward'] - total
            totals.append(total)
            residuals.append(delta)
            scores.extend(predictions)
            rows.append(dict(trajectory_sha256=row['trajectory_sha256'], task_id=row['task_id'],
                             reward=row['reward'], prm_scores=predictions, predicted_total=total,
                             calibration_residual=delta))
    payload = dict(totals=totals, residuals=residuals, scores=scores, rows=rows)
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(payload, gathered, dst=0)
        if rank != 0:
            return None, None
        payload = {key: [value for part in gathered for value in part[key]] for key in payload}
    residuals = payload['residuals']
    result = dict(examples=len(residuals), mse=sum(x*x for x in residuals) / len(residuals),
                  mae=sum(abs(x) for x in residuals) / len(residuals),
                  calibration_residual=distribution(residuals), scores=distribution(payload['scores']),
                  predicted_total=distribution(payload['totals']), finite=True)
    return result, payload['rows']


def train(args):
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from transformers import get_cosine_schedule_with_warmup, set_seed
    from spa_next.prm.model import load_trainable, score_record
    output, data = Path(args.output), Path(args.data)
    output.mkdir(parents=True, exist_ok=True)
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available():
        raise RuntimeError('PRM training requires CUDA BF16 support')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        torch.distributed.init_process_group('nccl')
    set_seed(args.seed)
    if not (Path(args.model) / 'spa_snapshot.json').is_file():
        raise ValueError('PRM must start from the separately merged, immutable SFT snapshot')
    if rank == 0:
        from spa_next.training.sft_snapshot import verify_snapshot
        verify_snapshot(args.model)
    if world > 1:
        torch.distributed.barrier()
    report = json.loads((data / 'data_report.json').read_text())
    if Path(report['model']).resolve() != Path(args.model).resolve() or report['max_length'] > 4096:
        raise ValueError('PRM model and context must match the frozen supervision preparation')
    for name in ('train', 'val'):
        if sha256_file(data / (name + '.jsonl')) != report['dataset_sha256'][name]:
            raise ValueError('PRM data changed after family split was frozen')
    training, validation = JsonlDataset(data / 'train.jsonl'), JsonlDataset(data / 'val.jsonl')
    if not len(training) or not len(validation):
        raise ValueError('PRM train and held-out sets must both be nonempty')
    if args.probe:
        import heapq
        longest = heapq.nlargest(min(16, len(training)), range(len(training)),
                                key=lambda i: len(training[i]['input_ids']))
        training = torch.utils.data.Subset(training, longest)
        validation = torch.utils.data.Subset(validation, range(min(16, len(validation))))
        args.max_updates = 2
        args.epochs = max(args.epochs, 2)
    microsteps = math.ceil(16 / world)
    steps_per_epoch = math.ceil(len(training) / 16)
    total_steps = steps_per_epoch * args.epochs
    if args.max_updates is not None:
        total_steps = min(total_steps, args.max_updates)
    if total_steps < 1 or not 1 <= args.epochs <= 3:
        raise ValueError('PRM uses one to three epochs with at least one update')
    config = dict(base=str(Path(args.model).resolve()), base_manifest_sha256=sha256_file(Path(args.model) / 'spa_snapshot.json'),
                  data=str(data.resolve()), seed=args.seed, epochs=args.epochs, global_batch_size=16,
                  per_device_microbatch=1, gradient_accumulation_steps=microsteps, world_size=world,
                  lora_rank=32, lora_alpha=64, lora_lr=1e-5, head_lr=1e-4, weight_decay=0.,
                  warmup_ratio=.03, scheduler='cosine', max_grad_norm=1., precision='backbone-bf16-head-loss-fp32',
                  max_context=4096, max_updates=args.max_updates, total_updates=total_steps,
                  save_every_updates=args.save_every,
                  supervision='mean((sum(action_scores)-environment_outcome)**2)',
                  provenance='newly-trained-qwen-spa-prm; not-original-author-checkpoint')
    if rank == 0:
        if (output / 'training_report.json').exists():
            raise FileExistsError('PRM run already finished; choose a fresh output directory')
        (output / 'training_config.json').write_text(json.dumps(config, indent=2))
    model = load_trainable(str(args.model), device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad], lr=1e-5),
        dict(params=list(model.head.parameters()), lr=1e-4)], weight_decay=0.)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps * .03), total_steps)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                      find_unused_parameters=False) if world > 1 else model
    started, update, best_mse, best_dir, history = time.monotonic(), 0, float('inf'), None, []
    start_epoch, start_offset = 0, 0
    if getattr(args, 'resume', None):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        recovery = Path(args.resume)
        state = torch.load(recovery / 'training_state.pt', map_location='cpu', weights_only=False)
        for key in ('base', 'base_manifest_sha256', 'data', 'seed', 'epochs', 'global_batch_size', 'total_updates'):
            if state['config'][key] != config[key]:
                raise ValueError('Resume configuration mismatch: ' + key)
        set_peft_model_state_dict(model.backbone, load_file(str(recovery / 'adapter/adapter_model.safetensors')))
        model.head.load_state_dict(torch.load(recovery / 'head.pt', map_location=device, weights_only=True))
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        update, start_epoch, start_offset = state['update'], state['epoch'], state['next_offset']
        for epoch_dir in sorted(output.glob('epoch-*')):
            metrics = json.loads((epoch_dir / 'validation_metrics.json').read_text())
            completed_epoch = int(epoch_dir.name.split('-')[-1])
            if completed_epoch > start_epoch:
                raise ValueError('Resume would overwrite a completed validation epoch')
            checkpoint = dict(epoch=completed_epoch, update=completed_epoch * steps_per_epoch,
                              path=str(epoch_dir.resolve()), validation=metrics)
            history.append(checkpoint)
            if metrics['mse'] < best_mse:
                best_mse, best_dir = metrics['mse'], epoch_dir
        if rank == 0:
            print(json.dumps(dict(event='prm_resumed', checkpoint=str(recovery), update=update,
                                  world_size=world, next_offset=start_offset)), flush=True)
    log_stream = (output / 'train_metrics.jsonl').open('a', buffering=1) if rank == 0 else None
    try:
        for epoch in range(start_epoch, args.epochs):
            order = list(range(len(training)))
            random.Random(args.seed + epoch).shuffle(order)
            wrapped.train()
            for offset in range(start_offset if epoch == start_epoch else 0, len(order), 16):
                if update >= total_steps:
                    break
                selected = order[offset:offset + 16]
                actual = len(selected)
                # Every rank executes the same number of backwards. Padded examples
                # get weight zero; DDP's rank average is corrected to an exact mean
                # over actual trajectories, including an incomplete final batch.
                selected += [selected[0]] * (microsteps * world - actual)
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=device)
                for micro in range(microsteps):
                    global_index = micro * world + rank
                    row = training[selected[global_index]]
                    ids = torch.tensor([row['input_ids']], device=device, dtype=torch.long)
                    valid = global_index < actual
                    context = wrapped.no_sync() if world > 1 and micro < microsteps - 1 else contextlib.nullcontext()
                    with context:
                        predicted = wrapped(input_ids=ids, action_end_positions=row['action_end_positions'],
                                            attention_mask=torch.ones_like(ids))
                        loss = (predicted.sum().float() - float(row['reward'])).square()
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite PRM loss')
                        (loss * (world / actual if valid else 0.)).backward()
                    if valid:
                        loss_sum += loss.detach()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                update += 1
                if world > 1:
                    torch.distributed.all_reduce(loss_sum)
                if rank == 0:
                    entry = dict(event='prm_update', update=update, total_updates=total_steps,
                                 epoch=epoch + 1, examples=actual, mse=float(loss_sum) / actual,
                                 grad_norm=float(grad_norm), lora_lr=optimizer.param_groups[0]['lr'],
                                 head_lr=optimizer.param_groups[1]['lr'], elapsed_seconds=time.monotonic() - started,
                                 cuda_peak_bytes=torch.cuda.max_memory_allocated(device))
                    log_stream.write(json.dumps(entry) + '\n')
                    if update == 1 or update % 10 == 0:
                        print(json.dumps(entry), flush=True)
                    (output / 'status.json').write_text(json.dumps(dict(status='training', **entry), indent=2))
                if not args.probe and args.save_every > 0 and update % args.save_every == 0:
                    if rank == 0:
                        recovery = output / 'recovery' / ('update-' + str(update))
                        recovery.mkdir(parents=True, exist_ok=False)
                        model.backbone.save_pretrained(recovery / 'adapter', safe_serialization=True)
                        torch.save(model.head.state_dict(), recovery / 'head.pt')
                        torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                        update=update, epoch=epoch, next_offset=offset + 16,
                                        config=config), recovery / 'training_state.pt')
                        (output / 'latest_recovery.json').write_text(json.dumps(dict(
                            checkpoint=str(recovery.resolve()), update=update, epoch=epoch + 1), indent=2))
                        print(json.dumps(dict(event='prm_recovery_checkpoint', update=update,
                                              checkpoint=str(recovery.resolve()))), flush=True)
                    if world > 1:
                        torch.distributed.barrier()
            metrics, predictions = evaluate(model, validation, device, rank, world)
            if rank == 0:
                epoch_dir = output / ('epoch-' + str(epoch + 1))
                epoch_dir.mkdir(exist_ok=False)
                model.backbone.save_pretrained(epoch_dir / 'adapter', safe_serialization=True)
                torch.save(model.head.state_dict(), epoch_dir / 'head.pt')
                (epoch_dir / 'validation_metrics.json').write_text(json.dumps(metrics, indent=2))
                with (epoch_dir / 'validation_predictions.jsonl').open('w') as stream:
                    for row in predictions:
                        stream.write(json.dumps(row) + '\n')
                checkpoint = dict(epoch=epoch + 1, update=update, path=str(epoch_dir.resolve()), validation=metrics)
                history.append(checkpoint)
                if metrics['mse'] < best_mse:
                    best_mse, best_dir = metrics['mse'], epoch_dir
                    (output / 'best_checkpoint.json').write_text(json.dumps(checkpoint, indent=2))
                print(json.dumps(dict(event='prm_validation', **checkpoint)), flush=True)
                (output / 'status.json').write_text(json.dumps(dict(status='validated_epoch', **checkpoint), indent=2))
            if world > 1:
                torch.distributed.barrier()
            if update >= total_steps:
                break
        passed = torch.zeros((), device=device, dtype=torch.int64)
        if rank == 0:
            gate = best_mse < report['constant_validation_mse']
            # A bounded smoke run can test the implementation but must never certify
            # deployment or automatically launch the policy experiments.
            qualified = gate and args.max_updates is None
            result = dict(status='smoke_completed' if args.max_updates is not None else ('completed' if qualified else 'gate_failed'), gate_passed=qualified,
                          validation_mse_gate_passed=gate, smoke_only=args.max_updates is not None,
                          best_validation_mse=best_mse, constant_validation_mse=report['constant_validation_mse'],
                          constant_train_mean=report['constant_train_mean'],
                          best_checkpoint=str(best_dir.resolve()), updates=update, history=history,
                          elapsed_seconds=time.monotonic() - started, config=config,
                          limitation='Terminal sum fit does not establish correct per-action credit attribution.')
            (output / 'training_report.json').write_text(json.dumps(result, indent=2))
            (output / 'status.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(dict(event='PRM_TRAINING_FINISHED', **result)), flush=True)
            passed.fill_(int(qualified or args.max_updates is not None))
        if world > 1:
            torch.distributed.broadcast(passed, src=0)
        if not int(passed):
            raise RuntimeError('PRM validation MSE did not beat the train-mean constant baseline; policy training is blocked')
    finally:
        if log_stream:
            log_stream.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

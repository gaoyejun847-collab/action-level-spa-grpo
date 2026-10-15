"""python -m spa_next.prm {prepare,train,export,verify,serve}."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--model', type=Path, required=True)
    prepare.add_argument('--split-dir', type=Path, required=True)
    prepare.add_argument('--trajectories', nargs='+', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--max-length', type=int, default=4096)
    prepare.add_argument('--seed', type=int, default=42)
    train = commands.add_parser('train')
    train.add_argument('--model', type=Path, required=True)
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--epochs', type=int, default=3)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--max-updates', type=int)
    train.add_argument('--save-every', type=int, default=200)
    train.add_argument('--resume', type=Path)
    train.add_argument('--probe', action='store_true', help='Two updates on the 16 longest trajectories; never deployment-qualified')
    export = commands.add_parser('export')
    export.add_argument('--model', type=Path, required=True, help='The immutable merged SFT base')
    export.add_argument('--training', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--device', default='cuda:0', help='Merge on a free GPU; CPU is supported explicitly')
    verify = commands.add_parser('verify')
    verify.add_argument('--model', type=Path, required=True, help='The independently merged PRM artifact')
    verify.add_argument('--data', type=Path, required=True)
    verify.add_argument('--max-examples', type=int, default=256)
    verify.add_argument('--cpu-threads', type=int, default=8)
    verify.add_argument('--device', default='cuda:0')
    serve = commands.add_parser('serve')
    serve.add_argument('--model', type=Path, required=True)
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8766)
    serve.add_argument('--cpu-threads', type=int, default=8)
    serve.add_argument('--backend', choices=('int8', 'original-gpu'), default='int8')
    serve.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.mode == 'prepare':
        from .data import prepare
        prepare(args)
    elif args.mode == 'train':
        from .train import train
        train(args)
    else:
        from . import deployment
        getattr(deployment, args.mode)(args)


if __name__ == '__main__':
    main()

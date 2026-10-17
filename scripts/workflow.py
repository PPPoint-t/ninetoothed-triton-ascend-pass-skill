#!/usr/bin/env python3
"""Command-line entry point for the personal migration skill."""

import argparse
import sys
from pathlib import Path

import engine
from mdio import WorkflowError, fields, lock


def parser():
    root = argparse.ArgumentParser(
        description="NineToothed 昇腾 pass 迁移与中文算子回归"
    )
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="建立中文批次、规则清单和源码快照")
    init.add_argument("--output", required=True)
    init.add_argument("--source", default="/home/swy/infini/ninetoothed-swy")
    init.add_argument("--target", default="/home/swy/infini/ninetoothed")
    init.add_argument("--policy", default="/home/swy/infini/ninetoothed.skill")
    for name in (
        "collect",
        "run",
        "compare",
        "begin",
        "finish",
        "recover",
        "recover-device",
        "status",
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--batch", required=True, type=Path)
        if name == "collect":
            sub.add_argument("--layer", choices=("源码", "设备"), required=True)
            sub.add_argument(
                "--group",
                choices=(
                    "核心",
                    "完整",
                    "逐元素索引",
                    "归约循环",
                    "矩阵布局",
                    "融合数学",
                ),
                default="核心",
            )
            sub.add_argument(
                "--backend",
                choices=("triton", "triton-ascend", "两者"),
                default="triton-ascend",
            )
            sub.add_argument(
                "--match", help="可选完整节点 glob；筛选后的计划不能替代固定核心回归"
            )
        if name == "run":
            sub.add_argument("--plan", required=True, type=Path)
            sub.add_argument("--phase", choices=("基线", "修改后"), required=True)
            sub.add_argument("--device", default="0")
            sub.add_argument("--timeout", type=float, default=120)
            sub.add_argument("--budget", type=float, default=1800)
        if name == "compare":
            sub.add_argument("--before", type=Path, required=True)
            sub.add_argument("--after", type=Path, required=True)
        if name in ("begin", "finish"):
            sub.add_argument("--rule", choices=tuple(engine.RULES), required=True)
        if name == "finish":
            sub.add_argument(
                "--decision",
                choices=("已有覆盖", "已实现", "已过时", "暂缓"),
                required=True,
            )
            sub.add_argument("--reason", required=True)
            sub.add_argument("--comparisons", nargs="*", type=Path, default=[])
        if name == "recover-device":
            sub.add_argument("--device", default="0")
            sub.add_argument(
                "--after-external-recovery",
                action="store_true",
                required=True,
                help="确认已完成外部恢复；只做健康检查，不执行设备 reset",
            )
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            print(engine.initialize(args.output, args.source, args.target, args.policy))
            return 0
        batch = args.batch.resolve()
        engine.config(batch)
        if args.command == "status":
            print(engine.summary_text(batch))
            return 0
        with lock(batch / "主控.lock"):
            if args.command == "collect":
                print(
                    engine.collect(
                        batch, args.layer, args.group, args.backend, args.match
                    )
                )
            elif args.command == "run":
                path = engine.execute(
                    batch, args.plan, args.phase, args.device, args.timeout, args.budget
                )
                print(path)
                return 0 if fields(path / "记录.md")["门禁"] == "通过" else 1
            elif args.command == "compare":
                path = engine.compare_runs(batch, args.before, args.after)
                print(path)
                return 0 if fields(path)["门禁"] == "通过" else 1
            elif args.command == "begin":
                engine.begin_rule(batch, args.rule)
            elif args.command == "finish":
                engine.finish_rule(
                    batch, args.rule, args.decision, args.reason, args.comparisons
                )
            elif args.command == "recover":
                for message in engine.recover(batch):
                    print(message)
            elif args.command == "recover-device":
                with lock(engine.device_paths(args.device)[0]) as device_lock:
                    print(
                        engine.health_probe(
                            batch,
                            args.device,
                            clear=True,
                            pass_fds=(device_lock.fileno(),),
                        )
                    )
        return 0
    except (WorkflowError, KeyError, ValueError, OSError) as exc:
        print(f"工作流停止：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

import os
import sys
import argparse
import subprocess
import multiprocessing


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)


BENCHMARK_CONFIGS = {
    "videomme": {
        "streaming": False,
        "anno_path": "data/videomme/videomme.json",
        "eval_cmds": [
            "{python} eval/eval_multiple_choice.py general --results_path {results_path}",
            "{python} eval/eval_multiple_choice.py videomme --results_path {results_path}",
        ],
    },
    "mvbench": {
        "streaming": False,
        "anno_path": "data/mvbench/mvbench.json",
        "eval_cmds": [
            "{python} eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
    "egoschema": {
        "streaming": False,
        "anno_path": "data/egoschema/egoschema.json",
        "eval_cmds": [
            "{python} eval/eval_multiple_choice.py egoschema --results_path {results_path}",
        ],
    },
    "rvs_ego": {
        "streaming": True,
        "anno_path": "data/rvs/ego/ego4d_oe.json",
        "eval_cmds": [
            "{python} eval/eval_open_ended.py --pred_path {results_path} --output_dir {save_dir}/tmp --output_json {save_dir}/results.json",
        ],
    },
    "rvs_movie": {
        "streaming": True,
        "anno_path": "data/rvs/movie/movienet_oe.json",
        "eval_cmds": [
            "{python} eval/eval_open_ended.py --pred_path {results_path} --output_dir {save_dir}/tmp --output_json {save_dir}/results.json",
        ],
    },
    "ovobench": {
        "streaming": True,
        "anno_path": "data/ovobench/ovobench_realtime_backeward.json",
        "eval_cmds": [
            "{python} eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
    "streamingbench": {
        "streaming": True,
        "anno_path": "data/streamingbench/streamingbench_realtime.json",
        "eval_cmds": [
            "{python} eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
}


def run_eval(args, config):
    num_chunks = args.num_chunks
    if args.rolekv_ragged and not args.kv_head_ragged_prefill:
        args.kv_head_ragged_prefill = True

    head_budget_suffix = ""
    if args.head_budget_scores:
        score_tag = os.path.splitext(os.path.basename(args.head_budget_scores))[0]
        if not score_tag:
            score_tag = args.head_budget_scores.replace("/", "_")
        head_budget_suffix = f"-headbudget-{score_tag}"
    layer_budget_suffix = ""
    if args.layer_budget_scores:
        score_tag = os.path.splitext(os.path.basename(args.layer_budget_scores))[0]
        if not score_tag:
            score_tag = args.layer_budget_scores.replace("/", "_")
        layer_budget_suffix = f"-layerbudget-{score_tag}"
        if args.layer_budget_variable:
            layer_budget_suffix += "-varlen"
    kv_head_budget_suffix = ""
    if args.kv_head_budget_scores:
        score_tag = os.path.splitext(os.path.basename(args.kv_head_budget_scores))[0]
        if not score_tag:
            score_tag = args.kv_head_budget_scores.replace("/", "_")
        kv_head_budget_suffix = (
            f"-kvheadbudget-{score_tag}"
            f"-{args.kv_head_budget_scheme}"
            f"-union{args.kv_head_budget_union_cap_ratio:g}"
        )
        if args.kv_head_budget_scheme.startswith("sparsemm"):
            kv_head_budget_suffix += (
                f"-r{args.kv_head_budget_sparsemm_ratio:g}"
                f"-w{args.kv_head_budget_sparsemm_window_size}"
            )
        if args.kv_head_ragged_prefill:
            kv_head_budget_suffix += "-raggedprefill"
        elif args.kv_head_ragged_decode:
            kv_head_budget_suffix += "-raggeddecode"
    rolekv_suffix = ""
    if args.rolekv_ragged:
        class_tag = os.path.splitext(os.path.basename(args.rolekv_head_classes))[0]
        if not class_tag:
            class_tag = args.rolekv_head_classes.replace("/", "_")
        rolekv_suffix = (
            f"-rolekvragged-{class_tag}"
            f"-{args.rolekv_ragged_mode}"
            f"-q{args.rolekv_quota_ratio:g}"
            f"-lm{args.rolekv_lambda_memory:g}"
            f"-lc{args.rolekv_lambda_current:g}"
        )
    save_dir = (
        f"results/{args.model}/{args.dataset}/"
        f"fps{args.sample_fps}-kv{args.kv_size}-{args.compress_mode}"
        f"{head_budget_suffix}{layer_budget_suffix}{kv_head_budget_suffix}{rolekv_suffix}"
    )
    streaming = config["streaming"]
    results_path = f"{save_dir}/results.csv"

    if not args.only_eval:
        devices = args.devices[:num_chunks] if args.devices else [str(idx) for idx in range(num_chunks)]
        if len(devices) < num_chunks:
            raise ValueError(
                f"num_chunks={num_chunks} requires {num_chunks} devices, "
                f"but got {len(devices)} from --devices."
            )
        processes = []
        for idx in range(num_chunks):
            cmd = [
                sys.executable, "video_qa/hermes_vqa.py",
                "--model", args.model,
                "--sample_fps", str(args.sample_fps),
                "--save_dir", save_dir,
                "--anno_path", config["anno_path"],
                "--debug", args.debug,
                "--num_chunks", str(num_chunks),
                "--chunk_idx", str(idx),
                "--kv_size", str(args.kv_size),
                "--streaming", str(streaming),
                "--compress_mode", args.compress_mode,
            ]
            if args.head_budget_scores:
                cmd.extend(["--head_budget_scores", args.head_budget_scores])
            if args.layer_budget_scores:
                cmd.extend(["--layer_budget_scores", args.layer_budget_scores])
                cmd.extend(["--layer_budget_variable", str(args.layer_budget_variable)])
            if args.kv_head_budget_scores:
                cmd.extend(["--kv_head_budget_scores", args.kv_head_budget_scores])
                cmd.extend([
                    "--kv_head_budget_union_cap_ratio",
                    str(args.kv_head_budget_union_cap_ratio),
                ])
                cmd.extend([
                    "--kv_head_budget_max_mask_q_len",
                    str(args.kv_head_budget_max_mask_q_len),
                ])
                cmd.extend(["--kv_head_budget_scheme", args.kv_head_budget_scheme])
                cmd.extend([
                    "--kv_head_budget_sparsemm_ratio",
                    str(args.kv_head_budget_sparsemm_ratio),
                ])
                cmd.extend([
                    "--kv_head_budget_sparsemm_window_size",
                    str(args.kv_head_budget_sparsemm_window_size),
                ])
            if args.kv_head_ragged_decode:
                cmd.extend(["--kv_head_ragged_decode", str(args.kv_head_ragged_decode)])
            if args.kv_head_ragged_prefill:
                cmd.extend(["--kv_head_ragged_prefill", str(args.kv_head_ragged_prefill)])
            if args.rolekv_ragged:
                cmd.extend(["--rolekv_ragged", str(args.rolekv_ragged)])
                cmd.extend(["--rolekv_head_classes", args.rolekv_head_classes])
                cmd.extend(["--rolekv_ragged_mode", args.rolekv_ragged_mode])
                cmd.extend(["--rolekv_quota_ratio", str(args.rolekv_quota_ratio)])
                cmd.extend(["--rolekv_lambda_memory", str(args.rolekv_lambda_memory)])
                cmd.extend(["--rolekv_lambda_current", str(args.rolekv_lambda_current)])
                cmd.extend(["--rolekv_seed", str(args.rolekv_seed)])
                cmd.extend(["--rolekv_role_min_votes", str(args.rolekv_role_min_votes)])
            p = multiprocessing.Process(target=exec, args=(cmd, True, devices[idx]))
            processes.append(p)
            p.start()

        failed_chunks = []
        for idx, p in enumerate(processes):
            p.join()
            chunk_file = f"{save_dir}/{num_chunks}_{idx}.csv"
            if not os.path.exists(chunk_file):
                failed_chunks.append(idx)
                print(f"WARNING: Chunk {idx} failed - file {chunk_file} not found!")

        if failed_chunks:
            raise RuntimeError(
                f"The following chunks failed: {failed_chunks}. Please rerun them manually."
            )

        exec(f"> {results_path}")
        for idx in range(num_chunks):
            chunk_file = f"{save_dir}/{num_chunks}_{idx}.csv"
            if idx == 0:
                exec(f"head -n 1 {chunk_file} > {results_path}")
            exec(f"tail -n +2 {chunk_file} >> {results_path}")
            exec(f"rm {chunk_file}")

    fmt = {
        "python": sys.executable,
        "results_path": results_path,
        "save_dir": save_dir,
        "anno_path": config["anno_path"],
    }
    for cmd_template in config["eval_cmds"]:
        exec(cmd_template.format(**fmt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'qwen2.5_vl_3b', 'qwen2.5_vl_7b', 'qwen2.5_vl_32b'])
    parser.add_argument("--dataset", type=str, default=None, choices=list(BENCHMARK_CONFIGS.keys()))
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--kv_size", type=int)
    parser.add_argument("--compress_mode", type=str, default="hermes", choices=["hermes", "streamingvlm"])
    parser.add_argument("--head_budget_scores", type=str, default=None,
                        help="Enable HERMES per-head voting budgets. Use 'pseudo', 'sparsemm', or a .npz/.json score file.")
    parser.add_argument("--layer_budget_scores", type=str, default=None,
                        help="Enable HERMES per-layer budgets. Use 'pseudo', 'sparsemm', or a .npz/.json score file.")
    parser.add_argument("--layer_budget_variable", type=str2bool, nargs='?', const=True, default=False,
                        help="Use real per-layer KV cache lengths with per-layer Qwen attention masks. Experimental.")
    parser.add_argument("--kv_head_budget_scores", type=str, default=None,
                        help="Enable per-KV-head logical eviction. Use 'pseudo', 'sparsemm_qwen25', or a .npz/.json score file.")
    parser.add_argument("--kv_head_budget_union_cap_ratio", type=float, default=1.0,
                        help="Dense union cap relative to kv_size for per-KV-head logical eviction.")
    parser.add_argument("--kv_head_budget_max_mask_q_len", type=int, default=128,
                        help="Only apply per-head masks when q_len is at most this value, to avoid OOM on large video chunks.")
    parser.add_argument(
        "--kv_head_budget_scheme",
        type=str,
        default="relative",
        choices=[
            "relative",
            "sparsemm",
            "sparsemm_layer_total",
            "sparsemm_layer",
            "sparsemm_total",
            "sparsemm_per_layer_total",
            "sparsemm_layer_exact",
            "sparsemm_per_layer",
        ],
        help=(
            "Per-KV-head budget allocation. 'relative': old per-layer min/max ratios; "
            "'sparsemm': SparseMM min-cache + global visual-score allocation with "
            "num_keep average per layer-KV-head; 'sparsemm_layer_total': SparseMM "
            "allocation with total budget num_keep*num_layers; "
            "'sparsemm_per_layer_total': SparseMM allocation with each layer summing "
            "to num_keep."
        ),
    )
    parser.add_argument("--kv_head_budget_sparsemm_ratio", type=float, default=0.1,
                        help="SparseMM floor ratio for each KV head when --kv_head_budget_scheme=sparsemm.")
    parser.add_argument("--kv_head_budget_sparsemm_window_size", type=int, default=32,
                        help="SparseMM recent/window budget added back after score allocation.")
    parser.add_argument("--kv_head_ragged_decode", type=str2bool, nargs='?', const=True, default=False,
                        help="Use experimental physical per-KV-head ragged cache for answer decode.")
    parser.add_argument("--kv_head_ragged_prefill", type=str2bool, nargs='?', const=True, default=False,
                        help="Use experimental physical per-KV-head ragged cache for video/text prefill and decode.")
    parser.add_argument("--rolekv_ragged", type=str2bool, nargs='?', const=True, default=False,
                        help="Use RoleKV-v2 role-aware physical per-KV-head ragged retention.")
    parser.add_argument("--rolekv_head_classes", type=str,
                        default="results/observations/head_classes_prev_current/head_classes.json",
                        help="Head-class JSON built from offline previous/current eager profiling.")
    parser.add_argument(
        "--rolekv_ragged_mode",
        type=str,
        default="rolekv_quota",
        choices=["baseline", "uniform", "norole", "rolekv", "rolekv_quota", "random", "random_quota", "inverted", "inverted_quota"],
        help="RoleKV-v2 role assignment/retention mode for ragged compression.",
    )
    parser.add_argument("--rolekv_quota_ratio", type=float, default=0.7,
                        help="Fraction of a role KV-head budget reserved for its target region.")
    parser.add_argument("--rolekv_lambda_memory", type=float, default=0.2,
                        help="Soft score bonus added to previous-memory tokens for memory-oriented KV heads.")
    parser.add_argument("--rolekv_lambda_current", type=float, default=0.2,
                        help="Soft score bonus added to latest-chunk tokens for current-sensitive KV heads.")
    parser.add_argument("--rolekv_seed", type=int, default=0,
                        help="Seed for random/inverted RoleKV control modes.")
    parser.add_argument("--rolekv_role_min_votes", type=int, default=1,
                        help="Minimum query-head votes needed to assign a KV head to a non-mixed role.")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated GPU IDs, e.g. '0,1,2'. Overrides auto-assignment.")
    args = parser.parse_args()

    budget_modes = [
        bool(args.head_budget_scores),
        bool(args.layer_budget_scores),
        bool(args.kv_head_budget_scores),
    ]
    if sum(budget_modes) > 1:
        raise ValueError(
            "Use only one of --head_budget_scores, --layer_budget_scores, "
            "or --kv_head_budget_scores."
        )

    if args.devices is not None:
        args.devices = args.devices.split(",")
    else:
        args.devices = []

    if args.dataset in BENCHMARK_CONFIGS:
        print(f'Execute {args.dataset} evaluation')
        run_eval(args, BENCHMARK_CONFIGS[args.dataset])

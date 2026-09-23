import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

STEPS = [
    (
        "校验评测数据",
        "scripts.validate_retrieval_dataset",
    ),
    (
        "BM25",
        "scripts.evaluate_bm25",
    ),
    (
        "Dense",
        "scripts.evaluate_dense",
    ),
    (
        "Hybrid + RRF",
        "scripts.evaluate_hybrid",
    ),
    (
        "Dense + Query Rewrite",
        "scripts.evaluate_dense_rewrite",
    ),
    (
        "Hybrid + Query Rewrite",
        "scripts.evaluate_hybrid_rewrite",
    ),
    (
        "Hybrid + Rewrite + Reranker",
        "scripts.evaluate_full_rag",
    ),
    (
        "生成消融实验报告",
        "scripts.compare_retrieval_results",
    ),
]


def run_step(
    index,
    total,
    name,
    module,
):
    print("\n" + "=" * 70)
    print(
        f"[{index}/{total}] {name}"
    )
    print("=" * 70)

    start = time.perf_counter()

    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    elapsed = (
        time.perf_counter() - start
    )

    print(
        f"{name} 完成，"
        f"耗时 {elapsed:.2f} 秒"
    )

    return elapsed


def main():
    total_start = time.perf_counter()
    timings = []

    for index, (
        name,
        module,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        try:
            elapsed = run_step(
                index=index,
                total=len(STEPS),
                name=name,
                module=module,
            )
        except subprocess.CalledProcessError:
            print(
                f"\n{name} 运行失败，"
                "已停止后续评测。"
            )
            raise SystemExit(1)

        timings.append(
            {
                "name": name,
                "seconds": elapsed,
            }
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    print("\n" + "=" * 70)
    print("全部评测完成")
    print("=" * 70)

    for item in timings:
        print(
            f"{item['name']}："
            f"{item['seconds']:.2f} 秒"
        )

    print(
        f"\n总耗时："
        f"{total_seconds:.2f} 秒"
    )

    print(
        "\n最终报告："
        "\nreports/retrieval/"
        "ablation_report.md"
    )


if __name__ == "__main__":
    main()
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


@dataclass(frozen=True)
class MoEParamModel:
    hidden: int = 1024
    intermediate: int = 2048
    num_mats: int = 3               # gate/up/down
    num_experts_total: int = 64 * 16
    base_params: int = 476_710_912  # 不随专家数/rank变化的参数

    @property
    def per_expert_full(self) -> int:
        return self.hidden * self.intermediate * self.num_mats

    @property
    def per_expert_svd_unit(self) -> int:
        # SVD 后 per-expert params = rank * (h+i) * num_mats
        return (self.hidden + self.intermediate) * self.num_mats

    @property
    def total_params(self) -> int:
        return self.base_params + self.num_experts_total * self.per_expert_full

    @property
    def rank_break_even(self) -> float:
        # 使得 SVD 参数量 == 原参数量 的 break-even rank
        return self.per_expert_full / self.per_expert_svd_unit


@dataclass
class State:
    n_experts: int
    per_expert_params: int
    svd_rank: Optional[int] = None

    def total_params(self, B: int) -> int:
        return B + self.n_experts * self.per_expert_params


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _choose_best_int_by_target(
    x_real: float,
    build_params_fn,
    target: float,
    lo: int,
    hi: int,
) -> int:
    """
    在 floor/ceil（以及 clamp 后）里选一个，使得 build_params_fn(x) 最接近 target
    """
    cand = set()
    cand.add(_clamp_int(int(math.floor(x_real)), lo, hi))
    cand.add(_clamp_int(int(math.ceil(x_real)), lo, hi))
    cand.add(_clamp_int(int(math.floor(x_real + 0.5)), lo, hi))
    best_x = None
    best_err = None
    for x in cand:
        p = build_params_fn(x)
        err = abs(p - target)
        if best_err is None or err < best_err:
            best_err = err
            best_x = x
    return best_x


def _normalize_order(order: str) -> str:
    order = order.upper().replace("-", "").replace("_", "").strip()
    if sorted(order) != sorted("PMD") or len(order) != 3:
        raise ValueError(f"order 必须是 'PMD' 的任意排列，比如 'DMP'，当前: {order}")
    return order


def solve_pmd_any_order(
    all_ratio: float,
    pruning_ratio: float,
    merging_ratio: float,
    *,
    order: str = "PMD",
    model: MoEParamModel = MoEParamModel(),
    # rank 上限：默认用 break-even 的 floor（保证 SVD 不会“增参”）
    force_rank_cap: Optional[int] = None,
    # 允许把“应为 0 的步骤”判定为 no-op 的容差（按参数量）
    noop_tol: float = 0.5,
) -> Dict[str, Any]:
    """
    通用求解：P/M/D 任意顺序。

    贡献率含义：按给定顺序逐步达到累计压缩率。
      - 总目标：最终 params = (1 - all_ratio) * T
      - 第 1 步目标：params = (1 - all_ratio * r_first) * T
      - 第 2 步目标：params = (1 - all_ratio * (r_first+r_second)) * T
      - 第 3 步目标：params = (1 - all_ratio) * T

    注意：因为顺序不同，后续步骤的“每减少一个 expert / 每降一个 rank”带来的减参量会不同，
    所以必须基于当前 state 解方程。
    """
    if not (0 <= all_ratio < 1):
        raise ValueError("all_ratio 必须在 [0,1) 内")
    if not (0 <= pruning_ratio <= 1 and 0 <= merging_ratio <= 1):
        raise ValueError("pruning_ratio / merging_ratio 必须在 [0,1]")
    if pruning_ratio + merging_ratio > 1 + 1e-12:
        raise ValueError("pruning_ratio + merging_ratio 不能超过 1（否则 svd_ratio 为负）")

    svd_ratio = 1.0 - pruning_ratio - merging_ratio
    order = _normalize_order(order)

    B = model.base_params
    T = model.total_params
    unit = model.per_expert_svd_unit
    full = model.per_expert_full

    # 初始 state：N 个 expert，每个 expert full 参数
    s = State(n_experts=model.num_experts_total, per_expert_params=full, svd_rank=None)

    # 给每个方法分配“贡献率”
    contrib_ratio = {"P": pruning_ratio, "M": merging_ratio, "D": svd_ratio}

    # rank cap
    if force_rank_cap is not None:
        rank_cap = int(force_rank_cap)
    else:
        rank_cap = int(math.floor(model.rank_break_even))  # 默认 682
    rank_cap = min(rank_cap, min(model.hidden, model.intermediate))
    rank_cap = max(rank_cap, 1)

    steps = []
    method_hparams = {"num_pruning": 0, "num_group": None, "svd_rank": None}

    cum = 0.0
    for method in order:
        r = contrib_ratio[method]
        cum += r
        target_params = (1.0 - all_ratio * cum) * T
        cur_params = s.total_params(B)

        # 如果这一步目标几乎等于当前，直接视为 no-op（尤其是某方法 ratio=0 的情况）
        if abs(target_params - cur_params) <= noop_tol:
            steps.append({
                "method": method,
                "target_params": target_params,
                "before_params": cur_params,
                "after_params": cur_params,
                "delta_removed": 0,
                "note": "no-op (target≈current)",
                "state_after": {"n_experts": s.n_experts, "per_expert": s.per_expert_params, "svd_rank": s.svd_rank},
            })
            continue

        if target_params > cur_params + noop_tol:
            # 目标比当前还大：意味着这一方法要“增参”，与压缩语义冲突
            # 这里直接做成 no-op，让后续方法去补；同时标记出来
            steps.append({
                "method": method,
                "target_params": target_params,
                "before_params": cur_params,
                "after_params": cur_params,
                "delta_removed": 0,
                "note": "infeasible (would increase params), treated as no-op",
                "state_after": {"n_experts": s.n_experts, "per_expert": s.per_expert_params, "svd_rank": s.svd_rank},
            })
            continue

        # 真正执行一步：根据 method 解方程
        if method == "P":
            # B + (n - x)*per = target  => x = n - (target-B)/per
            n0 = s.n_experts
            per = s.per_expert_params
            x_real = n0 - (target_params - B) / per

            def build_params(x: int) -> int:
                n1 = _clamp_int(n0 - x, 1, n0)  # 至少保留 1 个 expert
                return B + n1 * per

            x = _choose_best_int_by_target(x_real, build_params, target_params, lo=0, hi=n0 - 1)
            n1 = n0 - x
            after = B + n1 * per

            method_hparams["num_pruning"] += x  # 如果 P 出现一次就是它；这里写成 += 以防你未来扩展
            s.n_experts = n1

        elif method == "M":
            # B + g*per = target => g = (target-B)/per
            n0 = s.n_experts
            per = s.per_expert_params
            g_real = (target_params - B) / per

            def build_params(g: int) -> int:
                g = _clamp_int(g, 1, n0)
                return B + g * per

            g = _choose_best_int_by_target(g_real, build_params, target_params, lo=1, hi=n0)
            after = B + g * per

            method_hparams["num_group"] = g
            s.n_experts = g

        elif method == "D":
            # B + n*(rank*unit) = target => rank = (target-B)/(n*unit)
            n0 = s.n_experts
            r_real = (target_params - B) / (n0 * unit)

            def build_params(rk: int) -> int:
                rk = _clamp_int(rk, 1, rank_cap)
                return B + n0 * (rk * unit)

            rk = _choose_best_int_by_target(r_real, build_params, target_params, lo=1, hi=rank_cap)
            after = B + n0 * (rk * unit)

            method_hparams["svd_rank"] = rk
            s.per_expert_params = rk * unit
            s.svd_rank = rk

        else:
            raise RuntimeError("unexpected method")

        before = cur_params
        delta = before - after
        steps.append({
            "method": method,
            "target_params": target_params,
            "before_params": before,
            "after_params": after,
            "delta_removed": delta,
            "note": "",
            "state_after": {"n_experts": s.n_experts, "per_expert": s.per_expert_params, "svd_rank": s.svd_rank},
        })

    final_params = s.total_params(B)
    achieved_all_ratio = 1.0 - (final_params / T)

    removed_total = T - final_params
    removed_by_method = {"P": 0, "M": 0, "D": 0}
    for st in steps:
        removed_by_method[st["method"]] += st["delta_removed"]

    def safe_div(a: float, b: float) -> float:
        return 0.0 if b == 0 else float(a) / float(b)

    return method_hparams
    # achieved_contrib = {
    #     "pruning": safe_div(removed_by_method["P"], removed_total),
    #     "merging": safe_div(removed_by_method["M"], removed_total),
    #     "svd": safe_div(removed_by_method["D"], removed_total),
    # }

    # return {
    #     "input": {
    #         "all_ratio": all_ratio,
    #         "pruning_ratio": pruning_ratio,
    #         "merging_ratio": merging_ratio,
    #         "svd_ratio": svd_ratio,
    #         "order": order,
    #     },
    #     "hparams": method_hparams,
    #     "final": {
    #         "final_n_experts": s.n_experts,
    #         "final_per_expert_params": s.per_expert_params,
    #         "final_params": final_params,
    #         "achieved_all_ratio": achieved_all_ratio,
    #     },
    #     "steps": steps,
    #     "stats": {
    #         "orig_params": T,
    #         "removed_total": removed_total,
    #         "removed_by_method": removed_by_method,
    #         "achieved_contrib_ratio": achieved_contrib,
    #         "rank_break_even": model.rank_break_even,
    #         "rank_cap": rank_cap,
    #     },
    # }


# # ====== 简单示例 ======
# if __name__ == "__main__":
#     for order in ["PMD", "PDM", "MPD", "MDP", "DPM", "DMP"]:
#         out = solve_pmd_any_order(
#             all_ratio=0.5,
#             pruning_ratio=0.3,
#             merging_ratio=0,
#             order=order,
#         )
#         print(order, out["hparams"], out["final"]["achieved_all_ratio"], out["stats"]["achieved_contrib_ratio"])

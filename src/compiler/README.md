# Compiler Notes

## Evaluator 设计理念

`MappingEvaluator` 的职责是重放已经确定的 mapping 结果，并在同一套 runtime state 上重新统计最终成本。它不应该重新选择 partition、mapper 或通信策略；它只负责回答两个问题：

- 这个 mapping 在执行时会产生多少 local / remote / payload / protocol 成本。
- 执行完每个 record 后，逻辑比特和通信槽的 runtime 物理状态应该如何连续传递给下一个 record。

## 统一评估入口

Baseline 编译器与 NAVI 系列在产出 `MappingRecordList` 并完成 mapper 之后，统一调用 `MappingEvaluator` 重算成本。这样可以保证所有 pipeline 最终输出的 `record.costs` / `total_costs` 都来自同一套 runtime 回放逻辑，算法内部的启发式或估算只用于搜索，不作为最终报告。

评估策略由 `evaluator_policy` 控制，默认 `full_realistic`。可选值：`full_realistic` / `comm_to_all` / `local_all_to_all`。

因此 evaluator 里需要区分三类概念：

- **payload gate**：原始电路语义中的门，或者 `CommOp.gate_list` 中代表原始 payload 的门。
- **communication event**：一次显式 `CommOp`，或者普通跨 QPU 门被 evaluator 合成为 synthetic telegate 后产生的一次通信事件。
- **communication primitive**：为了实现通信协议额外引入的本地门，例如 source-side `cx/h`、return/landing `swap` 等。

### Runtime state

Evaluator 维护连续状态，而不是只看每个 record 的静态 partition：

- `state.logical_pos[q]` 表示原始线路中的 logical qubit 当前在哪个 QPU、哪根 local wire。这个 local wire 可以是 computation wire，也可以是 communication wire。
- `state.wire_phy_map[qpu][local_wire]` 表示该 QPU 上每根 local wire 当前对应的 physical qubit。
- `state.wire_owners[qpu][local_wire]` 表示该 QPU 上每根 local wire 当前被哪类 runtime state 占用。
- `state.routed_buffers[qpu]` 暂存需要交给 Qiskit routing 的本地门。

目标设计中，local wire 在每个 QPU 内统一编号：

```text
0 .. len(partition[qpu]) - 1
    computation wires

len(partition[qpu]) .. len(partition[qpu]) + comm_slot_reserve - 1
    communication wires
```

因此 `RuntimeLocation` 不必重复保存 `wire_kind` 或 `physical_slot`：

```python
@dataclass(frozen=True)
class RuntimeLocation:
    """
    Current runtime local wire of one logical qubit.

    local_wire is indexed in the QPU's full local-wire space:
      0 .. comp_wire_count - 1
          computation wires
      comp_wire_count .. comp_wire_count + comm_slot_reserve - 1
          communication wires
    """
    qpu_id: int
    local_wire: int

    def is_comp(self, comp_wire_count: int) -> bool:
        return int(self.local_wire) < int(comp_wire_count)

    def is_comm(self, comp_wire_count: int) -> bool:
        return int(self.local_wire) >= int(comp_wire_count)

    def comm_offset(self, comp_wire_count: int) -> int:
        if self.is_comp(comp_wire_count):
            raise ValueError("computation wire has no communication offset")
        return int(self.local_wire) - int(comp_wire_count)
```

每根 wire 的 runtime owner 用 `WireOwner` 表示：

```python
WireOwnerKind = Literal["resident", "entangled_copy", "protocol"]


@dataclass(frozen=True)
class WireOwner:
    """
    Runtime owner of one QPU-local wire.

    resident:
        The wire carries the true state of one original logical qubit.
        It must be consistent with logical_pos[logical_qid].

    entangled_copy:
        The wire carries a cat-ent destination-side entangled copy of one
        logical qubit. It is usable as that qubit's payload operand in the
        CAT destination block, but it does not update logical_pos.

    protocol:
        The wire is reserved for temporary communication protocol resources.
        It is not a payload operand location and does not update logical_pos.
    """
    kind: WireOwnerKind
    logical_qid: int | None = None
    label: str | None = None
```

`EvaluationState` 的目标结构如下：

```python
@dataclass
class EvaluationState:
    """
    Mutable replay state carried across records.

    logical_pos maps each original-circuit logical qubit to its current
    QPU-local wire. The wire may be computation or communication.

    wire_phy_map maps every local wire on every QPU to its current backend
    physical qubit. It is the only physical layout source.

    wire_owners maps every local wire on every QPU to its current runtime
    owner. It is the allocation source for computation and communication wires.

    routed_buffers and active_buffer_use_coupling_map track pending local ops
    before flush. Each QPU active buffer has one routing mode.
    """
    costs: ExecCosts = field(default_factory=ExecCosts)

    # logical qid -> current QPU-local wire
    logical_pos: dict[int, RuntimeLocation] = field(default_factory=dict)

    # qpu_id -> local_wire -> physical_qubit | None
    wire_phy_map: dict[int, list[int | None]] = field(default_factory=dict)

    # qpu_id -> local_wire -> WireOwner | None
    wire_owners: dict[int, list[WireOwner | None]] = field(default_factory=dict)

    # qpu_id -> pending local-wire circuit
    routed_buffers: list[QuantumCircuit] = field(default_factory=list)

    # qpu_id -> whether current pending buffer should use backend coupling_map.
    # None means the buffer is empty / mode not chosen yet.
    active_buffer_use_coupling_map: list[bool | None] = field(default_factory=list)
```

logical qubit 的 physical qubit 由统一的 wire map 推导：

```python
loc = state.logical_pos[q]
physical_qubit = state.wire_phy_map[loc.qpu_id][loc.local_wire]
```

这个目标模型可以替代当前拆开的两份 physical state：

```text
logical_pos[q].physical_slot      -> wire_phy_map[loc.qpu_id][loc.local_wire]
comm_phy_map[qpu][offset]         -> wire_phy_map[qpu][len(partition[qpu]) + offset]
```

这样做的目的，是让 logical qubit 可以自然地位于 communication wire 上。比如 cat-ent / TP / teledata 的中间阶段，如果某个 logical state 当前真实位于 comm wire，`state.logical_pos[q]` 就直接指向对应的 comm local wire。Payload gate 也可以直接在 comm wire 和 comp wire 之间执行；如果协议要求先 landing 到 comp wire，则应显式插入 `swap(comm_wire, comp_wire)`。

Evaluator 实现不应再把 `RuntimeLocation.physical_slot` 或独立的 `comm_phy_map` 作为权威状态。`MappingRecord.logical_phy_map` / `MappingRecord.comm_phy_map` 只是对外输出快照，必须从 `state.logical_pos` 和 `state.wire_phy_map` 投影得到。

### Record replay and partition transition

`evaluate()` 的主循环应把每条 record 的成本作为局部成本重新计算，最后由 `MappingRecordList.summarize_total_costs()` 汇总。因此进入每条 record 时必须重置 `state.costs = ExecCosts()`，避免复用 mapper 阶段或旧 evaluator 留在 `record.costs` 里的估算值，也避免把上一条 record 的成本累计进当前 record。

partition transition 的成本归属约定是：

- `record[0]` 没有进入成本，只统计自己的 body 成本。
- 对 `record[i] (i > 0)`，`record[i - 1].partition -> record[i].partition` 的 transition 成本计入 `record[i].costs`。
- `record[i].costs = 进入 record[i] 的 transition 成本 + record[i] body 成本`。

主循环语义应是：

```python
state = initialize_state_from_partition(records[0].partition)

for i, record in enumerate(records):
    # Costs are local to this record, not cumulative.
    state.costs = ExecCosts()

    if i > 0:
        evaluate_partition_transition(
            prev_partition=records[i - 1].partition,
            target_partition=record.partition,
            state=state,
            network=network,
            policy=policy,
        )
        # evaluate_partition_transition() finishes by compacting state to
        # record.partition and preparing target-shaped local buffers.
    else:
        prepare_local_buffers_for(record.partition)

    replay_record_body(record, state, network, policy)
    flush_local_ops(state, record.partition, network, policy)
    commit_state_snapshot_to_record(record, state)
```

这里最重要的约束是：**不能在 partition transition 之前把 `wire_owners` / `wire_phy_map` 粗暴 resize 成目标 partition 的形状。** transition 开始时，`state.logical_pos`、`state.wire_owners`、`state.wire_phy_map` 必须仍表示上一条 record 执行完后的 runtime state。否则旧 computation wire 可能被误判成目标 partition 的 communication wire，或者 resident owner 被隐式截断。

`resize` 如果仍作为实现 helper 存在，只能在 partition transition 内部或末尾使用，并且必须满足：

- 不在 source-side protocol 之前截断旧 wires。
- 不靠 resize 隐式释放 resident owner。
- 不靠 resize 隐式改变 logical qubit 的 local wire。
- 跨 QPU teledata、同 QPU compaction、owner 释放/占用都必须由显式协议步骤和 reservation helper 完成。
- resize 后的 `wire_owners` / `wire_phy_map` 必须与已经显式迁移完成的 `logical_pos` 一致。

partition transition 内部应使用临时 expanded wire space，避免旧 partition 和目标 partition 的 computation wire 数不一致时误判 wire kind。对每个 QPU：

```python
transition_comp_count[qpu] = max(
    len(prev_partition[qpu]),
    len(target_partition[qpu]),
)
transition_wire_count[qpu] = transition_comp_count[qpu] + comm_slot_reserve
```

transition 期间的 communication wires 从 `transition_comp_count[qpu]` 开始，而不是从 `len(target_partition[qpu])` 开始。这样旧 computation wire 在 transition 完成前仍然是 computation-like wire，不会因为目标 partition 较小而被误当成 communication wire。所有 source-side protocol、remote staging、local compaction 和 landing swaps 都在这个 expanded wire space 中执行。transition 完成后，再把 state 收敛到 target partition 的连续 wire shape。

partition transition 不只是跨 QPU teledata。相邻 record 的 partition 变化可能同时包含：

- **跨 QPU teledata**：logical qubit 的 resident state 从一个 QPU 移到另一个 QPU。
- **同 QPU local compaction / placement repair**：logical qubit 仍在同一个 QPU，但目标 partition 的 computation wire 数可能变小，导致它当前所在 wire 不再属于目标 comp 区间。
- **wire-space shape update**：每个 QPU 的 computation wire 数可能变化，communication wire 起点也随之变化。这个形状变化只能在上述显式迁移和 compaction 完成后落地。

`partition[qpu]` 的 list 顺序不定义 runtime local wire 顺序；它只定义哪些 logical qubits 属于该 QPU。transition 完成时只要求每个 logical qubit 位于目标 partition 指定的 QPU，并且位于该 QPU 的有效 computation wire 区间 `0 .. len(target_partition[qpu])-1`。如果当前 local wire 已经在这个区间内，即使它和 `partition[qpu]` 中的 list index 不一致，也不需要移动。只有当 resident qubit 位于将被裁掉的旧 comp wire、或 landing 需要腾出有效 comp wire 时，才需要通过本地 `swap` 做 compaction / placement repair。这个 swap 属于 partition transition 的 communication primitive，不计入 `payload_gate_num`，但应计入 local gate / fidelity 成本。

### Wire reservation helpers

后续 evaluator 应使用统一的 wire reservation helper 维护 `wire_owners` 和 `logical_pos`。这个 helper 只处理 runtime ownership，不插入本地门，不调用 transpile，不统计成本，也不修改 `wire_phy_map`。

`WireOwner.kind` 的语义如下：

- `resident`：这根 wire 持有某个 logical qubit 的真实 runtime state。一个 logical qubit 同时只能有一个 `resident` owner，并且必须和 `state.logical_pos[q]` 一致。
- `entangled_copy`：这根 wire 持有 cat-ent 在 destination QPU 上生成的 entangled copy。它和某个 logical qubit 关联，但不释放 source resident，也不更新 `state.logical_pos[q]`。
- `protocol`：这根 wire 被通信协议临时占用，例如 cat-ent source-side helper comm wire。它不表示 logical state 的真实位置，也不是 payload operand 的默认查找位置。

`resolve_free_or_explicit_wire` 只负责选择和校验目标 wire，不修改 state：

```python
def resolve_free_or_explicit_wire(
    state: EvaluationState,
    partition: list[list[int]],
    qpu_id: int,
    local_wire: int | None,
    wire_kind: Literal["comp", "comm"] | None,
) -> int:
    comp_wire_count = len(partition[qpu_id])
    wire_count = len(state.wire_owners[qpu_id])

    if local_wire is None:
        if wire_kind is None:
            raise ValueError("local_wire and wire_kind cannot both be None")

        if wire_kind == "comp":
            candidates = range(0, comp_wire_count)
        elif wire_kind == "comm":
            candidates = range(comp_wire_count, wire_count)
        else:
            raise ValueError(f"unknown wire_kind: {wire_kind}")

        for candidate in candidates:
            if state.wire_owners[qpu_id][candidate] is None:
                return candidate

        raise RuntimeError(f"no free {wire_kind} wire on qpu {qpu_id}")

    if local_wire < 0 or local_wire >= wire_count:
        raise ValueError(f"local_wire out of range: {local_wire}")

    if wire_kind is not None:
        is_comp = local_wire < comp_wire_count
        is_comm = local_wire >= comp_wire_count

        if wire_kind == "comp" and not is_comp:
            raise ValueError("expected comp wire, got comm wire")
        if wire_kind == "comm" and not is_comm:
            raise ValueError("expected comm wire, got comp wire")

    owner = state.wire_owners[qpu_id][local_wire]
    if owner is not None:
        raise RuntimeError(
            f"target wire already occupied: qpu={qpu_id}, "
            f"wire={local_wire}, owner={owner}"
        )

    return local_wire
```

`release_old_resident_owner` 只释放 logical qubit 的旧真实位置。它不释放 `entangled_copy`，因为 entangled copy 不是 `logical_pos[q]` 指向的真实 state：

```python
def release_old_resident_owner(
    state: EvaluationState,
    logical_qid: int,
) -> None:
    old_loc = state.logical_pos.get(logical_qid)
    if old_loc is None:
        return

    old_owner = state.wire_owners[old_loc.qpu_id][old_loc.local_wire]
    if (
        old_owner is None
        or old_owner.kind != "resident"
        or old_owner.logical_qid != logical_qid
    ):
        raise RuntimeError(
            f"logical_pos and wire_owners inconsistent for logical qubit "
            f"{logical_qid}: loc={old_loc}, owner={old_owner}"
        )

    state.wire_owners[old_loc.qpu_id][old_loc.local_wire] = None
```

`reserve_wire` 是统一公开 helper。它可以自动分配空闲 comp / comm wire，也可以使用调用者指定的 `local_wire`。只有 `owner_kind="resident"` 会释放旧 resident 并更新 `state.logical_pos`；`entangled_copy` 和 `protocol` 只占用目标 wire。调用 `reserve_wire(..., owner_kind="resident")` 前，调用者必须保证旧 resident 所在 QPU 没有 pending local ops 仍引用旧 wire；如果不确定，应先 `flush_local_ops(qpu_ids=[old_loc.qpu_id])`。这个 helper 默认表示真实迁移，不支持把 resident reserve 到自己当前所在 wire 的 no-op；如果调用点可能出现 no-op，应在调用前直接跳过：

```python
def reserve_wire(
    state: EvaluationState,
    partition: list[list[int]],
    qpu_id: int,
    local_wire: int | None = None,
    wire_kind: Literal["comp", "comm"] | None = None,
    owner_kind: WireOwnerKind = "protocol",
    logical_qid: int | None = None,
    label: str | None = None,
) -> RuntimeLocation:
    target_wire = resolve_free_or_explicit_wire(
        state=state,
        partition=partition,
        qpu_id=qpu_id,
        local_wire=local_wire,
        wire_kind=wire_kind,
    )
    target_loc = RuntimeLocation(qpu_id=qpu_id, local_wire=target_wire)

    if owner_kind == "resident":
        if logical_qid is None:
            raise ValueError("resident owner requires logical_qid")

        release_old_resident_owner(state, logical_qid)
        state.wire_owners[qpu_id][target_wire] = WireOwner(
            kind="resident",
            logical_qid=logical_qid,
            label=label,
        )
        state.logical_pos[logical_qid] = target_loc
        return target_loc

    if owner_kind == "entangled_copy":
        if logical_qid is None:
            raise ValueError("entangled_copy owner requires logical_qid")
        if logical_qid not in state.logical_pos:
            raise RuntimeError(
                f"cannot create entangled_copy for unknown logical qubit "
                f"{logical_qid}"
            )

        state.wire_owners[qpu_id][target_wire] = WireOwner(
            kind="entangled_copy",
            logical_qid=logical_qid,
            label=label,
        )
        return target_loc

    if owner_kind == "protocol":
        state.wire_owners[qpu_id][target_wire] = WireOwner(
            kind="protocol",
            logical_qid=logical_qid,
            label=label,
        )
        return target_loc

    raise ValueError(f"unknown owner_kind: {owner_kind}")
```

协议结束时可以用 `release_wire` 释放不再使用的 `entangled_copy` / `protocol` owner。这个函数不应该释放 `resident`，因为 resident 表示真实 logical state；如果要移动 resident，应调用 `reserve_wire(..., owner_kind="resident")`：

```python
def release_wire(
    state: EvaluationState,
    qpu_id: int,
    local_wire: int,
    expected_owner_kind: Literal["entangled_copy", "protocol"] | None = None,
) -> None:
    owner = state.wire_owners[qpu_id][local_wire]
    if owner is None:
        raise RuntimeError(f"wire is already free: qpu={qpu_id}, wire={local_wire}")

    if expected_owner_kind is not None and owner.kind != expected_owner_kind:
        raise RuntimeError(
            f"unexpected owner kind on qpu={qpu_id}, wire={local_wire}: "
            f"expected={expected_owner_kind}, actual={owner}"
        )

    if owner.kind == "resident":
        raise RuntimeError("resident owner must be moved, not released")

    state.wire_owners[qpu_id][local_wire] = None
```

典型调用方式：

```python
# TP / RTP / teledata remote move: logical state really moves to a comm wire.
dst_comm = reserve_wire(
    state,
    partition,
    qpu_id=dst_qpu,
    wire_kind="comm",
    owner_kind="resident",
    logical_qid=source,
    label="tp-dst-comm",
)

# CAT destination: create an entangled copy, but keep source resident unchanged.
dst_comm = reserve_wire(
    state,
    partition,
    qpu_id=dst_qpu,
    wire_kind="comm",
    owner_kind="entangled_copy",
    logical_qid=source,
    label="cat-dst-comm",
)

# CAT source-side comm wire: reserve a protocol comm wire.
src_comm = reserve_wire(
    state,
    partition,
    qpu_id=src_qpu,
    wire_kind="comm",
    owner_kind="protocol",
    label="cat-src-comm",
)
```

协议结束时还需要显式释放不再使用的 `entangled_copy` / `protocol` owner。`resident` owner 通常通过下一次 `reserve_wire(..., owner_kind="resident")` 迁移，不应被随意释放。

### `add_local_ops` 与 `flush_local_ops`

后续 evaluator 应把本地操作拆成两个职责清晰的步骤：

- `add_local_ops`：只负责把本地操作 enqueue 到对应 QPU 的完整 local-wire buffer。
- `flush_local_ops`：唯一负责调用 Qiskit transpile、统计本地成本、写回 `wire_phy_map`。

这样普通 local gate、communication primitive、landing swap 都可以复用同一套 route/charge/persist 机制，不再需要按操作来源维护多套路径。

#### add_local_ops

- 在目标 QPU 的完整 local-wire buffer 上追加一个或多个本地操作。这个 buffer 应包含该 QPU 的所有 computation wires 和 reserved communication wires。
- 每个 op 携带 `LocalGateKind`，用于后续 flush 时决定 routing policy 和成本归因。
- 不直接调用 Qiskit transpile。
- 不直接更新 `wire_phy_map`。
- 不直接改变 `state.logical_pos`。

推荐接口语义可以理解为：

```python
add_local_ops(
    qpu_id,
    ops=[(gate: Qiskit Gate Type, [local_wire0, local_wire1])],
    kind=LocalGateKind.PAYLOAD,
)
```

`add_local_ops` 在真正 append 前必须保证该 QPU active buffer 的 routing mode 与新 op 兼容。伪代码如下：

```python
def add_local_ops(qpu_id, ops, kind):
    use_coupling_map = should_use_coupling_map(kind, policy)

    current_mode = active_buffer_use_coupling_map[qpu_id]
    if current_mode is None:
        active_buffer_use_coupling_map[qpu_id] = use_coupling_map
    elif current_mode != use_coupling_map:
        flush_local_ops(qpu_ids=[qpu_id])
        active_buffer_use_coupling_map[qpu_id] = use_coupling_map

    for gate, wires in ops:
        routed_buffers[qpu_id].append(gate, wires)
```

一次 `add_local_ops` 调用中的 ops 应该具有同一个 `kind`。因此，不同 routing mode 的 ops 不会混进同一个 flush block。`flush_local_ops` 可以假设当前 buffer 内所有 ops 都使用同一个 `use_coupling_map` 模式。

#### flush_local_ops

`flush_local_ops` 是本地 routing 和 layout 落地的唯一入口：

```python
flush_local_ops(
    state: EvaluationState,
    partition: list[list[int]],
    network: Network,
    policy: EvaluationPolicy,
    qpu_ids: Sequence[int] | None = None,
) -> EvaluationState
```

它负责：

- 对指定 QPU，或所有 QPU，transpile 当前 local-wire buffer。
- 从 `network.backends[qpu_id]` 获取 backend、basis gates、coupling map 和 calibration 信息。
- 从 `policy` 获取 optimization level、fill initial layout 等 evaluation 选项。
- 从 `state.active_buffer_use_coupling_map[qpu_id]` 判断本次 flush 是否使用 backend coupling map。
- 从 `partition` 推导每个 QPU 的 computation wire 数量和 communication wire 区间。
- 根据 transpiled circuit 统计 local gate / fidelity 成本。
- 读取 Qiskit final layout，并写回 `wire_phy_map[qpu][local_wire]`。
- 清空已经 flush 的 local-wire buffer。

`flush_local_ops` 不直接改变 `logical_pos` 或 `wire_owners`。例如 teledata landing 执行 `swap(comm_wire, target_comp_wire)` 后，应先 flush 相关 QPU，让 `wire_phy_map` 落地；然后调用 `reserve_wire(..., owner_kind="resident", local_wire=target_comp_wire)` 更新 logical ownership。该 logical qubit 最终在哪个 physical qubit 上，由更新后的 `wire_phy_map[qpu][target_comp_wire]` 推导。flush 也不自动释放 `entangled_copy` / `protocol` owner；即使 flush 是由 routing mode 变化自动触发，临时 owner 仍必须由协议代码显式 `release_wire(...)`。

抽象伪代码：

```python
def flush_local_ops(
    state: EvaluationState,
    partition: list[list[int]],
    network: Network,
    policy: EvaluationPolicy,
    qpu_ids: Sequence[int] | None = None,
) -> EvaluationState:
    # None means this is a global semantic boundary, so flush every QPU.
    targets = all_qpus(network) if qpu_ids is None else qpu_ids

    for qpu_id in targets:
        # Empty buffers have no layout effect, but their pending routing mode
        # should be reset so the next add_local_ops can choose a fresh mode.
        if buffer_is_empty(state, qpu_id):
            reset_buffer_mode(state, qpu_id)
            continue

        backend = network.backends[qpu_id]
        use_coupling_map = state.active_buffer_use_coupling_map[qpu_id]

        # The initial layout is derived from the single source of truth:
        # wire_phy_map[qpu_id][local_wire] -> physical_qubit.
        initial_layout = build_initial_layout(
            wire_phy_map=state.wire_phy_map[qpu_id],
            fill_missing=True,
            backend=backend,
        )

        # route_* policy has already been collapsed into use_coupling_map
        # when ops were enqueued. Here we only execute the active buffer mode.
        transpiled = transpile_local_buffer(
            buffer=state.routed_buffers[qpu_id],
            backend=backend,
            use_coupling_map=use_coupling_map,
            initial_layout=initial_layout,
            optimization_level=policy.optimization_level,
        )

        # Count the actual post-transpile local gates, including routing gates
        # when use_coupling_map=True and basis decomposition in all modes.
        accumulate_local_costs(
            state=state,
            backend=backend,
            transpiled=transpiled,
        )

        # Persist Qiskit's final local-wire -> physical-qubit layout.
        state.wire_phy_map[qpu_id] = extract_final_wire_layout(
            transpiled=transpiled,
            wire_count=len(state.wire_phy_map[qpu_id]),
        )

        # Start a fresh full-QPU local-wire buffer for the next segment.
        clear_local_buffer(state, partition, network, qpu_id)

    validate_physical_state(state, partition, network)
    return state
```

#### Routing policy

`route_payload_gates`、`route_comm_gates` 不应该决定是否调用 transpile；它们应该决定 flush 时是否使用 backend coupling map。

- `route_* = True`：flush 时传入 `backend.coupling_map`，按真实拓扑 route，统计 routing 产生的额外本地门，并写回 `wire_phy_map`。
- `route_* = False`：flush 时不传入 `coupling_map`，按 all-to-all topology 统计本地成本。仍然可以传入 `basis_gates` 和 `optimization_level` 做 basis decomposition / optimization，但不应引入拓扑 SWAP。

这样 `route_* = False` 不等于“完全不 transpile”，而是“不做拓扑 routing”。这比直接用原始 gate 计费更细，因为它仍可以统计 basis decomposition 后的本地门成本。

实现时要避免把不同 routing mode 的 ops 混在同一个 flush block 中。例如 `route_payload_gates=False` 但 `route_comm_gates=True` 时，payload ops 应按 all-to-all flush，comm primitive ops 应按 coupling-map flush。

第一版实现不需要完整的 `FlushKey`。由于 buffer 已经按 QPU 分开，而 `basis_gates`、`coupling_map` 和 `optimization_level` 当前分别由 backend 和 policy 固定，所以每个 QPU 的 active buffer 只需要记录一个模式：

```python
active_buffer_use_coupling_map[qpu_id]: bool | None
```

当下一个 op 的 `use_coupling_map` 与该 QPU 当前 active buffer 的模式不同时，必须先 flush 当前 buffer，再开启新的 buffer segment。flush 时，如果 `use_coupling_map=True` 就传入 `backend.coupling_map`；如果为 `False` 就不传 `coupling_map`，按 all-to-all topology 处理。

未来如果引入 per-op basis gates、per-op optimization level、多种 coupling constraints，或者同一 QPU 内多种 routing backend，再把这个布尔模式升级成完整的 `FlushKey`。

在这个模型下，普通 payload gate、source-side `cx/h`、CAT/RTP/TP protocol gate、teledata landing swap、empty TP return swap 都可以复用同一套本地执行器。landing / return 的 `swap(comm, comp)` 属于 `COMM_PRIMITIVE`：它是通信协议为了完成状态落地或归还引入的本地门，不是 payload gate，也不是跨 QPU 间共享 EPR pair 的 entanglement swapping。区别只在：

- `ops` 里追加哪些 gate。
- `kind` 用于成本归因和 policy 判断。
- flush 后是否需要调用 `reserve_wire(..., owner_kind="resident")` 更新 resident owner。

普通 payload / resident local gate 通常不需要 resident ownership 迁移，因为 logical qubit 仍在原 local wire 上；routing 只会改变 `wire_phy_map`。Placement-defined landing / return 操作需要在 flush 后调用 `reserve_wire(..., owner_kind="resident")`，因为 logical state 会从 comm wire 变成 comp wire，或者从临时 wire 回到 home wire。

#### Flush 粒度

第一版重构中，flush 粒度应主要由每个 QPU active buffer 的 `use_coupling_map` 模式和语义边界决定，而不是单独依赖 `strict_flush_on_remote` / `flush_each_comm_gate`：

- **routing mode 变化**：当新 op 的 `use_coupling_map` 与该 QPU 当前 active buffer 不同时，必须 flush 当前 buffer。
- **语义需要最新 layout**：如果后续操作需要读取最新 `wire_phy_map`，必须先 flush。
- **resident ownership change**：landing / return 操作必须先 enqueue 对应 local ops，然后立即 flush 相关 QPU，并在 flush 后通过 `reserve_wire(..., owner_kind="resident")` 迁移 resident owner。
- **record 结束**：record 结束前必须 flush 所有 QPU buffer。

`strict_flush_on_remote` 和 `flush_each_comm_gate` 保留为 debug / conservative mode：

- `strict_flush_on_remote=True`：即使 `use_coupling_map` 模式没变，也在 remote communication boundary 前后强制 flush。
- `flush_each_comm_gate=True`：即使 `use_coupling_map` 模式没变，也在 `CommOp.gate_list` 内每个 payload gate 后强制 flush 对应 QPU。

Evaluator flush 前应始终为 `wire_phy_map` 中未知的 local wire 补齐 initial physical qubit，并把完整 local-wire layout 传给 Qiskit `transpile`。这样 replay 不依赖 Qiskit 自动布局选择，也避免 all-to-all / no-layout 场景下 final layout 缺失。旧的 `policy.fill_initial_layout` 字段可保留为配置兼容项，但不应再改变 evaluator 行为。

Placement-defined landing / return routing 不能只构造包含 `comm` 和目标 `comp` 两根 wire 的小电路来 transpile。例如物理拓扑是 `0 -- 1 -- 2`，`comm wire -> physical 0`、目标 `comp wire -> physical 2`，而另一个 resident `comp wire -> physical 1`。如果只把 `swap(comm, target_comp_wire)` 交给 Qiskit，transpiler 可能会把 physical 1 当成可自由使用的路由位置，但真实 state 里 physical 1 已经承载了其它 resident qubit。因此 evaluator 对每个 QPU 使用完整 local-wire buffer，包含该 QPU 的所有 computation wires 和 reserved communication wires；landing / return / compaction swap 都 append 到这个完整 buffer。transpile 后读取所有 local wire 的 final physical layout，并写回整个 `wire_phy_map[qpu]`，而不是只更新参与 swap 的两根 wire。

此前代码里存在 `_add_local_gate()` 和 `_add_transient_local_gates()` 两套路径。它们应被视为过渡实现：后续重构时应以 `add_local_ops + flush_local_ops` 为目标，让所有真实本地操作都 enqueue / route / charge / persist `wire_phy_map`，不再用 transient 路径隐藏 routing 对 physical layout 的影响。


### 计数原则

成本计数按来源分开：

- 普通本地 payload 门增加 local gate / fidelity 成本。
- 显式 `CommOp` 增加 `comm_block_events`。
- 普通跨 QPU 门如果没有被包装成 `CommOp`，会被 evaluator 合成为 synthetic CAT telegate，并增加 `telegate_exec_events`。
- 所有原始/payload gate 都计入 `payload_gate_num`，包括普通本地 payload gate、synthetic telegate payload，以及 `CommOp.gate_list` 中真正执行的 payload gate。
- remote move 通过 `CompilerUtils.update_remote_move_costs()` 计入 e-pair、hop、 remote fidelity 等成本。
- communication primitive 引入的本地门也要计入 local gate / fidelity 成本；如果 policy 要求 routing，则应统计 routing 后的实际本地门成本。

换句话说，communication primitive 不是免费操作。即使它不是原始 payload，也必须根据 policy 统计本地执行或 routing 开销。

### Telegate

Telegate 是原始 subcircuit 里直接出现的普通跨 QPU payload gate。它本身不是 `CommOp`，但 evaluator 不应该为它维护一套独立协议；它应该被包装成 synthetic `CommOp(comm_type="cat")`，然后进入统一的 CommOp replay 逻辑。

Simple telegate 的处理策略：

1. 外层 record replay loop 先从 `state.logical_pos` 判断普通 gate 是否跨 QPU。
2. 如果所有 operand 在同一 QPU，则它不是 telegate，外层直接作为本地 payload gate 交给 `add_local_ops`。
3. 只有已经确认跨 QPU 的普通 gate 才进入 `process_simple_telegate`。
4. `process_simple_telegate` 选择 source / destination，构造 synthetic `CommOp(comm_type="cat")`。
5. 把原始 gate 复制进 synthetic `CommOp.gate_list`，并保留 `_global_lqids` 元数据。
6. synthetic `CommOp` 进入 `process_commop(..., stats_kind=TELEGATE_EXEC)`。

伪代码：

```python
def process_record_gate(gate, global_lqids):
    qpus = {state.logical_pos[q].qpu_id for q in global_lqids}
    if len(qpus) == 1:
        qpu_id = only(qpus)
        wires = [state.logical_pos[q].local_wire for q in global_lqids]
        add_local_ops(qpu_id, ops=[(gate, wires)], kind=PAYLOAD)
        return

    process_simple_telegate(gate, global_lqids)


def process_simple_telegate(gate, global_lqids):
    # Called only for ordinary gates already known to span multiple QPUs.
    source = global_lqids[0]
    src_qpu = state.logical_pos[source].qpu_id
    dst_qpu = choose_destination_qpu(global_lqids, state.logical_pos)

    gate_copy = copy_gate_with_global_lqids(gate, global_lqids)
    synthetic = CommOp(
        comm_type="cat",
        source_qubit=source,
        src_qpu=src_qpu,
        dst_qpu=dst_qpu,
        involved_qubits=global_lqids,
        gate_list=[gate_copy],
    )
    process_commop(synthetic, stats_kind=TELEGATE_EXEC)
```

Telegate 计数字段：

- `telegate_exec_events`：simple telegate 被包装成 synthetic `CommOp` 的次数。
- `payload_gate_num`：所有原始/payload gate 数量；这里包含 simple telegate 被包装成 synthetic `CommOp` 后真正执行的 payload gate。
- `telegate_exec_remote_*`：synthetic `CommOp` 引入的 remote 成本归因。
- local gate / fidelity 的全局字段包含 payload routing 和 communication primitive routing 的实际本地成本。若需要 debug 级别归因，再把这些 local 成本细分到 telegate execution 的局部字段。

### CommOp

`CommOp` 是显式通信块。它可能来自 AutoComm / NAVI，也可能由 simple telegate 在 evaluator 内部 synthetic 出来。Mapper 在重排 partition 时需要同步维护 explicit `CommOp.src_qpu` 和 `CommOp.dst_qpu`；evaluator 执行前要做 endpoint precheck，避免 stale endpoint metadata 把协议门写入错误的 QPU buffer。

CommOp 的通用处理顺序：

1. **endpoint precheck**：验证 `src_qpu` / `dst_qpu` 与当前 runtime state 一致。若 logical qubit 当前位于 comm wire，endpoint 应从该实际位置推导。空 TP return 没有 payload 端点可推导，因此 `dst_qpu` 用 partition 中 source 的 home QPU 校验。
2. **event attribution**：explicit `CommOp` 计入 `comm_block_events`；simple telegate 包装出来的 synthetic `CommOp` 计入 `telegate_exec_events`。
3. **remote cost**：根据 `comm_type` 统计 e-pair、hop、remote fidelity 等成本。
4. **protocol local cost**：source / destination 侧协议门按 `COMM_PRIMITIVE` 统计，并通过 `add_local_ops` / `flush_local_ops` route、charge、persist `wire_phy_map`。
5. **payload cost**：`gate_list` 中的 gate 是 payload gate，计入 `payload_gate_num`。普通本地 payload gate 也计入同一字段。payload 可以直接作用在 comm wire 和 comp wire 上，不需要 evaluator 隐式 landing。
6. **state transition**：如果协议让 logical qubit 从 comp wire 移到 comm wire，或从 comm wire 回到 comp wire，应通过 `reserve_wire(..., owner_kind="resident")` 更新 `state.logical_pos` / `wire_owners`；routing 改变 QPU layout 时只更新 `wire_phy_map`。

CommOp 计数字段：

- `comm_block_events`：explicit `CommOp` 的次数，不包含 simple telegate。
- `comm_block_remote_*`：explicit `CommOp` 引入的 remote 成本归因。
- `payload_gate_num`：所有原始/payload gate 数量，包含普通本地 payload gate 和所有 `CommOp.gate_list` 中真正执行的 payload gate。
- local gate / fidelity 的全局字段包含 CommOp payload 和 communication primitive 的实际本地成本。

#### CommOp: cat-ent / CAT

CAT 的语义是：source logical qubit 逻辑上留在 source QPU；destination comm wire 上生成的是用于执行 destination payload 的 entangled copy，不把 `state.logical_pos[source]` 永久迁移到 destination。source-side protocol 和 routing 可以改变 source qubit 在 source QPU 上的 physical qubit，但 source 的 logical ownership 仍指向 source QPU 的 local wire。

CAT 处理伪代码：

```python
def process_cat(comm_op):
    source = comm_op.source_qubit
    src_qpu = comm_op.src_qpu
    dst_qpu = comm_op.dst_qpu

    src_loc = state.logical_pos[source]
    require(src_loc.qpu_id == src_qpu)

    src_wire = src_loc.local_wire
    src_comm = reserve_wire(
        state,
        partition,
        qpu_id=src_qpu,
        wire_kind="comm",
        owner_kind="protocol",
        label="cat-src-comm",
    )
    dst_comm = reserve_wire(
        state,
        partition,
        qpu_id=dst_qpu,
        wire_kind="comm",
        owner_kind="entangled_copy",
        logical_qid=source,
        label="cat-dst-comm",
    )

    # 1. Source-side entangling primitive. This may update wire_phy_map[src_qpu]
    #    after flush, but source ownership remains src_wire.
    add_local_ops(
        qpu_id=src_qpu,
        ops=[(Gate("cx"), [src_wire, src_comm.local_wire])],
        kind=COMM_PRIMITIVE,
    )

    # 2. Remote entanglement cost.
    update_remote_move_costs(src_qpu, dst_qpu, num_pairs=1)

    # 3. Payload gates. If payload references source, map source operand to
    #    the destination entangled copy; do not update state.logical_pos[source].
    for gate in comm_op.gate_list:
        qs = gate.global_lqids
        if source in qs:
            wires = []
            for q in qs:
                if q == source:
                    wires.append(dst_comm.local_wire)
                else:
                    loc = state.logical_pos[q]
                    require(loc.qpu_id == dst_qpu)
                    wires.append(loc.local_wire)
            add_local_ops(qpu_id=dst_qpu, ops=[(gate, wires)], kind=PAYLOAD)
        else:
            qpu_id = unique_runtime_qpu(qs)
            wires = [state.logical_pos[q].local_wire for q in qs]
            add_local_ops(qpu_id=qpu_id, ops=[(gate, wires)], kind=PAYLOAD)

    # 4. Destination-side disentangler / cleanup on dst_comm.
    add_local_ops(
        qpu_id=dst_qpu,
        ops=[(Gate("h"), [dst_comm.local_wire])],
        kind=COMM_PRIMITIVE,
    )

    # 5. The temporary communication wires remain reserved until their pending local
    #    ops are flushed; then they can be reused by later communication.
    flush_local_ops(qpu_ids=[src_qpu, dst_qpu])
    release_wire(state, src_comm.qpu_id, src_comm.local_wire, "protocol")
    release_wire(state, dst_comm.qpu_id, dst_comm.local_wire, "entangled_copy")
```

CAT 计数规则：

- explicit CAT 增加 `comm_block_events`；synthetic CAT 增加 `telegate_exec_events`。
- remote entanglement 增加 e-pair / hop / remote fidelity 成本。
- source-side primitive、destination-side cleanup 和 payload routing 都计入 local gate / fidelity 成本。
- payload gate 计入 `payload_gate_num`。
- `state.logical_pos[source]` 不迁移到 destination。
- 当前伪代码采用 conservative per-CAT flush：每个 CAT 结束后 flush 并释放 `src_comm` / `dst_comm`。后续如果要 batch 多个 CAT，可以延迟 flush / release，但必须保证所有临时 comm wire 在被 pending ops 引用期间一直保持 occupied。

#### CommOp: TP

TP 的语义是：source logical state 可以临时移动到 destination comm wire 执行 payload；空 `gate_list` 的 TP 是 return block，用来把 source state 从 comm wire 显式 landing / return 回 source 所属 QPU 的某个空 comp wire。这个语义符合“一个 logical qubit 当前只有一个真实位置”的模型，也不依赖 partition list 内部顺序。

非空 TP 处理伪代码：

```python
def process_tp_payload(comm_op):
    source = comm_op.source_qubit
    src_qpu = comm_op.src_qpu
    dst_qpu = comm_op.dst_qpu

    src_loc = state.logical_pos[source]
    require(src_loc.qpu_id == src_qpu)

    src_wire = src_loc.local_wire
    src_comm = reserve_wire(
        state,
        partition,
        qpu_id=src_qpu,
        wire_kind="comm",
        owner_kind="protocol",
        label="tp-src-comm",
    )

    # 1. Source-side teleport primitive.
    add_local_ops(qpu_id=src_qpu, ops=[
        (Gate("cx"), [src_wire, src_comm.local_wire]),
        (Gate("h"), [src_wire]),
    ], kind=COMM_PRIMITIVE)

    # Flush before resident migration because reserve_wire(... resident)
    # releases the old owner referenced by source-side pending ops.
    flush_local_ops(qpu_ids=[src_qpu])
    release_wire(state, src_comm.qpu_id, src_comm.local_wire, "protocol")

    # 2. Remote move places source logical state at destination comm wire.
    #    Reserve resident only after source-side pending ops are flushed,
    #    because resident reservation releases the old source owner.
    update_remote_move_costs(src_qpu, dst_qpu, num_pairs=1)
    dst_comm = reserve_wire(
        state,
        partition,
        qpu_id=dst_qpu,
        wire_kind="comm",
        owner_kind="resident",
        logical_qid=source,
        label="tp-dst-comm",
    )

    # 3. Payload executes directly on source's current comm wire and other
    #    destination runtime wires.
    for gate in comm_op.gate_list:
        wires = [state.logical_pos[q].local_wire for q in gate.global_lqids]
        require(all(state.logical_pos[q].qpu_id == dst_qpu for q in gate.global_lqids))
        add_local_ops(qpu_id=dst_qpu, ops=[(gate, wires)], kind=PAYLOAD)
```

空 TP return 处理伪代码：

```python
def process_empty_tp_return(comm_op):
    source = comm_op.source_qubit
    src_qpu = comm_op.src_qpu          # where source state currently is
    dst_qpu = comm_op.dst_qpu          # source home QPU

    src_loc = state.logical_pos[source]
    require(src_loc.qpu_id == src_qpu)
    require(src_loc.is_comm(comp_wire_count=len(partition[src_qpu])))

    home_wire = select_free_target_comp_wire(state, partition, dst_qpu)

    # 1. Flush current source QPU before resident migration. Pending payload
    #    ops may still reference source's current comm wire.
    flush_local_ops(qpu_ids=[src_qpu])

    # 2. Return remote move places source state at home QPU comm wire.
    update_remote_move_costs(src_qpu, dst_qpu, num_pairs=1)
    dst_comm = reserve_wire(
        state,
        partition,
        qpu_id=dst_qpu,
        wire_kind="comm",
        owner_kind="resident",
        logical_qid=source,
        label="tp-return-dst-comm",
    )

    # 3. Landing swap. Route on the full QPU local-wire buffer, then after
    # flush set source ownership to the selected comp wire. physical qubit is
    # inferred from wire_phy_map[dst_qpu][home_wire].
    add_local_ops(
        qpu_id=dst_qpu,
        ops=[(Gate("swap"), [dst_comm.local_wire, home_wire])],
        kind=COMM_PRIMITIVE,
    )
    flush_local_ops(qpu_ids=[dst_qpu])
    reserve_wire(
        state,
        partition,
        qpu_id=dst_qpu,
        local_wire=home_wire,
        wire_kind="comp",
        owner_kind="resident",
        logical_qid=source,
        label="tp-return-home",
    )
```

TP 计数规则：

- non-empty TP 的 source-side `cx/h` 计入 local gate / fidelity 成本。
- TP remote move 计入 e-pair / hop / remote fidelity 成本。
- `gate_list` payload 计入 `payload_gate_num`。
- empty TP return 的 return remote move 和 landing swap 分别计入 remote / local gate 成本，但不计入 `payload_gate_num`。
- non-empty TP 会把 `state.logical_pos[source]` 临时指到 destination comm wire；empty TP return 再把它指回 source 所属 QPU 的某个空 comp wire。

#### CommOp: RTP

RTP 是 round-trip teleportation。它可以看作 TP payload + return 的组合：source state 临时到 destination comm wire 执行 payload，然后在同一个 CommOp 内返回 source。

RTP 处理伪代码：

```python
def process_rtp(comm_op):
    source = comm_op.source_qubit
    src_qpu = comm_op.src_qpu
    dst_qpu = comm_op.dst_qpu

    src_loc = state.logical_pos[source]
    require(src_loc.qpu_id == src_qpu)

    src_wire = src_loc.local_wire
    src_comm = reserve_wire(
        state,
        partition,
        qpu_id=src_qpu,
        wire_kind="comm",
        owner_kind="protocol",
        label="rtp-src-comm",
    )

    # 1. Teleport source to destination comm wire.
    add_local_ops(qpu_id=src_qpu, ops=[
        (Gate("cx"), [src_wire, src_comm.local_wire]),
        (Gate("h"), [src_wire]),
    ], kind=COMM_PRIMITIVE)

    # Flush before resident migration because reserve_wire(... resident)
    # releases the old owner referenced by source-side pending ops.
    flush_local_ops(qpu_ids=[src_qpu])
    release_wire(state, src_comm.qpu_id, src_comm.local_wire, "protocol")

    update_remote_move_costs(src_qpu, dst_qpu, num_pairs=1)
    dst_comm0 = reserve_wire(
        state,
        partition,
        qpu_id=dst_qpu,
        wire_kind="comm",
        owner_kind="resident",
        logical_qid=source,
        label="rtp-dst-comm0",
    )

    # 2. Payload at destination.
    for gate in comm_op.gate_list:
        wires = [state.logical_pos[q].local_wire for q in gate.global_lqids]
        require(all(state.logical_pos[q].qpu_id == dst_qpu for q in gate.global_lqids))
        add_local_ops(qpu_id=dst_qpu, ops=[(gate, wires)], kind=PAYLOAD)

    # 3. Return protocol at destination.
    dst_comm1 = reserve_wire(
        state,
        partition,
        qpu_id=dst_qpu,
        wire_kind="comm",
        owner_kind="protocol",
        label="rtp-dst-comm1",
    )
    add_local_ops(qpu_id=dst_qpu, ops=[
        (Gate("cx"), [dst_comm0.local_wire, dst_comm1.local_wire]),
        (Gate("h"), [dst_comm0.local_wire]),
    ], kind=COMM_PRIMITIVE)
    # Flush before returning resident because destination payload / return
    # ops may still reference dst_comm0.
    flush_local_ops(qpu_ids=[dst_qpu])
    release_wire(state, dst_comm1.qpu_id, dst_comm1.local_wire, "protocol")

    update_remote_move_costs(dst_qpu, src_qpu, num_pairs=1)
    src_comm1 = reserve_wire(
        state,
        partition,
        qpu_id=src_qpu,
        wire_kind="comm",
        owner_kind="resident",
        logical_qid=source,
        label="rtp-src-comm1",
    )

    # 4. Land returned state back to source wire.
    add_local_ops(
        qpu_id=src_qpu,
        ops=[(Gate("swap"), [src_comm1.local_wire, src_wire])],
        kind=COMM_PRIMITIVE,
    )
    flush_local_ops(qpu_ids=[src_qpu, dst_qpu])
    reserve_wire(
        state,
        partition,
        qpu_id=src_qpu,
        local_wire=src_wire,
        wire_kind="comp",
        owner_kind="resident",
        logical_qid=source,
        label="rtp-src-home",
    )
```

RTP 计数规则：

- RTP remote cost 是两次 one-way remote move。
- source-side teleport primitive、destination-side return primitive、source-side landing swap 都计入 local gate / fidelity 成本。
- payload gate 计入 `payload_gate_num`。
- RTP 结束后 source logical qubit 回到 source QPU 的 source wire；physical qubit 由 `wire_phy_map[src_qpu][src_wire]` 推导。

### Partition transition / Teledata

Partition transition 发生在相邻 record 的 partition 改变时。它的目标是把上一条 record 执行完后的 runtime state 显式迁移到下一条 record 的 target partition，并把 transition 成本计入目标 record。

Teledata 是 partition transition 中的跨 QPU 部分：它表示 resident logical qubit 跨 QPU 迁移，因此执行结束后必须更新 `state.logical_pos`，使跨 QPU moved qubits 和目标 partition 一致。

同一个 transition 还必须处理不跨 QPU 的 local compaction。只检查 `src_qpu != dst_qpu` 不够；如果 `src_qpu == dst_qpu`，但该 qubit 当前位于目标 partition 的有效 comp 区间之外，仍需要显式执行本地 compaction，使最终 `state.logical_pos[q].qpu_id == partition_qpus(target_partition)[q]` 且 `state.logical_pos[q].local_wire < len(target_partition[qpu])`。

在统一 local-wire 模型下，teledata 的基本执行模型是：先把 moved logical qubit remote 到目标 QPU 的 comm wire，再显式执行 `swap(comm, target_comp_wire)` landing 到目标 comp wire。remote move 计入 e-pair、hop、remote fidelity 等成本；source-side `cx/h` 和 target-side landing swap 都按 `COMM_PRIMITIVE` 计入 local gate / fidelity 成本。

Teledata 调度伪代码使用两个小结构：

```python
@dataclass(frozen=True)
class TeledataMove:
    logical_qid: int
    src_qpu: int
    dst_qpu: int
    # None means: choose any currently free target comp wire on dst_qpu.
    dst_wire: int | None = None


def partition_qpus(partition: list[list[int]]) -> dict[int, int]:
    # Return membership only:
    # logical_qid -> qpu_id
    ...


def copy_current_resident_locations(state: EvaluationState) -> dict[int, RuntimeLocation]:
    # Freeze source locations at the start of this partition transition.
    # Only resident logical locations are copied; entangled_copy / protocol
    # owners are temporary communication resources and must not be migration
    # sources.
    ...


def append_move_to_graph(graph, src_qpu: int, dst_qpu: int, logical_qid: int) -> None:
    # Keep graph[src][dst]["weight"] == len(graph[src][dst]["qubits"]).
    if graph.has_edge(src_qpu, dst_qpu):
        graph[src_qpu][dst_qpu]["qubits"].append(logical_qid)
        graph[src_qpu][dst_qpu]["weight"] += 1
    else:
        graph.add_edge(src_qpu, dst_qpu, weight=1, qubits=[logical_qid])


def decrement_or_remove_edge(graph, src_qpu: int, dst_qpu: int) -> None:
    # Called after exactly one qubit has been removed from graph[src][dst]["qubits"].
    graph[src_qpu][dst_qpu]["weight"] -= 1
    require(graph[src_qpu][dst_qpu]["weight"] == len(graph[src_qpu][dst_qpu]["qubits"]))
    if graph[src_qpu][dst_qpu]["weight"] == 0:
        graph.remove_edge(src_qpu, dst_qpu)
```

`evaluate_partition_transition` 是 partition transition 的入口。它应先冻结上一条 record 结束后的 resident runtime locations，然后按目标 QPU membership 识别跨 QPU moves。跨 QPU 部分可保留现有 teledata 架构：建 residual QPU migration graph 时直接处理 bidirectional pair；建完残余图后处理 directed cycles；最后处理 remaining one-way moves。所有跨 QPU teledata landing 都选择目标 QPU 上任意空的有效 target comp wire，不绑定 `target_partition[qpu]` 中的 list index。teledata 结束后，再对仍位于目标 comp 区间外的 resident qubits 做同 QPU local compaction。

```python
def evaluate_partition_transition(prev_partition, target_partition, network, policy, state):
    target_qpus = partition_qpus(target_partition)
    transition_partition = build_transition_partition(prev_partition, target_partition)

    # This resizes only to expanded transition space, not to target space. It
    # must preserve all old resident owners and physical wire mappings.
    resize_state_to_transition_space(state, prev_partition, transition_partition)

    # Source locations must be read from this fixed snapshot while building the
    # graph. Bidirectional pair processing mutates state.logical_pos.
    old_locations = copy_current_resident_locations(state)

    qpu_teledata_graph = nx.DiGraph()
    qpu_teledata_graph.add_nodes_from(range(network.num_backends))

    for q, dst_qpu in target_qpus.items():
        src_loc = old_locations[q]
        src_qpu = src_loc.qpu_id

        if src_qpu == dst_qpu:
            continue

        # Bidirectional pairs are consumed immediately while building graph.
        if qpu_teledata_graph.has_edge(dst_qpu, src_qpu):
            reverse_qubits = qpu_teledata_graph[dst_qpu][src_qpu]["qubits"]
            if reverse_qubits:
                partner_q = reverse_qubits.pop(0)
                decrement_or_remove_edge(qpu_teledata_graph, dst_qpu, src_qpu)

                process_bidirectional_teledata_pair(
                    moves=[
                        TeledataMove(q, src_qpu, dst_qpu),
                        TeledataMove(partner_q, dst_qpu, src_qpu),
                    ],
                    state=state,
                    partition=transition_partition,
                    target_partition=target_partition,
                    network=network,
                    policy=policy,
                )
                continue

        append_move_to_graph(
            qpu_teledata_graph,
            src_qpu=src_qpu,
            dst_qpu=dst_qpu,
            logical_qid=q,
        )

    process_cycle_teledata(qpu_teledata_graph, state, transition_partition, target_partition, network, policy)
    process_remaining_teledata(qpu_teledata_graph, state, transition_partition, target_partition, network, policy)
    repair_local_compaction(state, transition_partition, target_partition, network, policy)

    compact_state_to_target_partition(state, transition_partition, target_partition)

    require_logical_positions_match_partition_membership(state, target_partition)
    validate_physical_state(state, target_partition, network)
    return state
```

#### Teledata 调度和状态更新

Partition transition 需要按迁移图和同 QPU wire dependency 调度，避免同一个 QPU 上的 wire ownership 冲突：

- unit pair / unit cycle / unit path 调度必须为同一个 batch 内每个 QPU 同时预留 outgoing source-side protocol comm wires 和 incoming destination resident comm wires。双向 teledata 中，每个参与 QPU 既有 1 个 outgoing source-side protocol，又有 1 个 incoming resident landing，因此每个参与 QPU 至少需要 2 根 communication wires。更一般地，batch 对某个 QPU 的 comm 需求是 `outgoing_count[qpu] + incoming_count[qpu]`；如果可用 reserved comm wires 不足，应直接报错，而不是依赖分阶段实现复用同一根 comm wire。
- 一对互换迁移应在建图时直接处理掉，不进入后续 cycle / remaining 阶段。
- 对有向环迁移，按 QPU-level cycle 上的 unit-weight cycle 逐个处理：每条边 pop 一个 logical qubit，先全部 remote 到目标 comm wires，再全部 landing 到目标 QPU 上任意空的有效 target comp wire。
- 对单向链式迁移，目标 QPU 可能需要同一 transition 中其它 qubit 先迁出才能出现空 comp wire。Evaluator 应把这类依赖组成 unit path：path 中间 QPU 的 outgoing remote 会释放一个有效 comp wire，供 incoming landing 使用。
- 同 QPU local compaction 只处理最终仍位于目标 comp 区间外的 resident qubit。它不按 `partition[qpu]` list index 排序，也不能直接改 `logical_pos[q].local_wire`。
- 在任意时刻，一个 logical qubit 只能有一个真实 `RuntimeLocation`。remote move 后它可以位于 comm wire；landing 后它位于 comp wire。
- `wire_phy_map` 是 physical layout 的唯一来源。source-side protocol routing、 remote landing、target landing swap 都必须通过它读取 initial layout，并在需要持久化时写回 final layout。
- transition 完成后必须检查每个 logical qubit 的 QPU 与 `partition_qpus(target_partition)[q]` 一致，并且 `local_wire < len(target_partition[qpu])`。不要求 local wire 等于 logical qubit 在 partition list 中的 index。

`require_comp_wire_available_for_landing` 是 landing 前的防线。它应检查目标 wire 是 computation wire，并且当前没有 owner。pair / cycle / path 的 Phase 2 已经把本 batch 中所有 moving qubits 从旧 comp wire 迁到 destination comm wire，所以中间节点的目标 comp wire 应该已经为空；如果仍不为空，说明 path/cycle decomposition 或 owner 维护有错：

```python
def require_comp_wire_available_for_landing(state, partition, qpu_id: int, local_wire: int) -> None:
    require(local_wire < len(partition[qpu_id]))
    owner = state.wire_owners[qpu_id][local_wire]
    if owner is not None:
        raise RuntimeError(
            f"teledata landing target comp wire is occupied: "
            f"qpu={qpu_id}, wire={local_wire}, owner={owner}"
        )
```

`_process_teledata_batch` 是 teledata 的唯一执行 helper。bidirectional pair、unit cycle 和 unit path 都只负责生成 `moves`，然后调用这个 helper。一个 batch 内每个 QPU 至多一个 outgoing source-side protocol wire 和一个 incoming destination resident comm wire。

```python
def _process_teledata_batch(moves, label_prefix, state, partition, target_partition, network, policy):
    require_teledata_batch_comm_capacity(moves, network)
    src_comms = {}
    dst_comms = {}
    move_landing_wire = {}

    # Phase 1: source-side protocol on every source QPU.
    for move in moves:
        src_loc = state.logical_pos[move.logical_qid]
        require(src_loc.qpu_id == move.src_qpu)

        src_comms[move.logical_qid] = reserve_wire(
            state,
            partition,
            qpu_id=move.src_qpu,
            wire_kind="comm",
            owner_kind="protocol",
            label=f"{label_prefix}-src-comm",
        )
        add_local_ops(qpu_id=move.src_qpu, ops=[
            (Gate("cx"), [src_loc.local_wire, src_comms[move.logical_qid].local_wire]),
            (Gate("h"), [src_loc.local_wire]),
        ], kind=COMM_PRIMITIVE)

    flush_local_ops(qpu_ids=unique(move.src_qpu for move in moves))

    for src_comm in src_comms.values():
        release_wire(state, src_comm.qpu_id, src_comm.local_wire, "protocol")

    # Phase 2: remote residents to destination comm wires. This releases the
    # old comp wires owned by moving qubits in this batch.
    for move in moves:
        update_remote_move_costs(move.src_qpu, move.dst_qpu, num_pairs=1)
        dst_comms[move.logical_qid] = reserve_wire(
            state,
            partition,
            qpu_id=move.dst_qpu,
            wire_kind="comm",
            owner_kind="resident",
            logical_qid=move.logical_qid,
            label=f"{label_prefix}-dst-comm",
        )

    # Phase 3: landing swaps into free target comp wires. If move.dst_wire is
    # None, choose any free wire in range(len(target_partition[dst_qpu])).
    for move in moves:
        dst_wire = move.dst_wire
        if dst_wire is None:
            dst_wire = select_free_target_comp_wire(state, target_partition, move.dst_qpu)
        require_comp_wire_available_for_landing(
            state,
            qpu_id=move.dst_qpu,
            local_wire=dst_wire,
        )
        add_local_ops(
            qpu_id=move.dst_qpu,
            ops=[(Gate("swap"), [dst_comms[move.logical_qid].local_wire, dst_wire])],
            kind=COMM_PRIMITIVE,
        )
        move_landing_wire[move.logical_qid] = dst_wire

    flush_local_ops(qpu_ids=unique(move.dst_qpu for move in moves))

    # Phase 4: commit resident ownership to target comp wires.
    for move in moves:
        reserve_wire(
            state,
            partition,
            qpu_id=move.dst_qpu,
            local_wire=move_landing_wire[move.logical_qid],
            wire_kind="comp",
            owner_kind="resident",
            logical_qid=move.logical_qid,
            label=f"{label_prefix}-target",
        )
```

`process_bidirectional_teledata_pair` 处理一对互换迁移。它是一个特殊的 unit batch：两个 moved qubits 先同时离开各自 source comp wire 到对方 comm wire，然后 landing 到对方目标 comp wire。

```python
def process_bidirectional_teledata_pair(moves, state, partition, target_partition, network, policy):
    require(len(moves) == 2)
    _process_teledata_batch(
        moves=moves,
        label_prefix="teledata-pair",
        state=state,
        partition=partition,
        target_partition=target_partition,
        network=network,
        policy=policy,
    )
```

`process_cycle_teledata` 可以保留原有 “枚举所有 simple cycles + valid 检查” 的写法。对每个 weighted cycle，取 cycle 上的 `min_weight`，把它当作 `min_weight` 个边权均为 1 的 QPU cycle 逐个执行：

```python
def process_cycle_teledata(graph, state, partition, target_partition, network, policy):
    cycles_by_length: dict[int, list[list[int]]] = defaultdict(list)
    for cycle in nx.simple_cycles(graph):
        if len(cycle) >= 3:
            cycles_by_length[len(cycle)].append([int(qpu_id) for qpu_id in cycle])

    for length in sorted(cycles_by_length):
        for cycle in cycles_by_length[length]:
            min_weight = None
            valid = True

            # Previous cycle processing mutates graph, so the precomputed cycle
            # may have become stale. Re-check edges before using it.
            for i in range(length):
                src_qpu = cycle[i]
                dst_qpu = cycle[(i + 1) % length]
                if not graph.has_edge(src_qpu, dst_qpu):
                    valid = False
                    break

                weight = int(graph[src_qpu][dst_qpu]["weight"])
                if weight <= 0:
                    valid = False
                    break

                min_weight = weight if min_weight is None else min(min_weight, weight)

            if not valid or min_weight is None:
                continue

            for _ in range(min_weight):
                moves = []

                # Pop one logical qubit from each QPU edge. This produces one
                # edge-weight-1 cycle, so with two comm wires per QPU the batch
                # has at most one incoming and one outgoing teledata move per QPU.
                for i in range(length):
                    src_qpu = cycle[i]
                    dst_qpu = cycle[(i + 1) % length]
                    q = graph[src_qpu][dst_qpu]["qubits"].pop(0)

                    moves.append(TeledataMove(
                        logical_qid=q,
                        src_qpu=src_qpu,
                        dst_qpu=dst_qpu,
                    ))

                _process_teledata_batch(
                    moves=moves,
                    label_prefix="teledata-cycle",
                    state=state,
                    partition=partition,
                    target_partition=target_partition,
                    network=network,
                    policy=policy,
                )

            for i in range(length):
                src_qpu = cycle[i]
                dst_qpu = cycle[(i + 1) % length]
                graph[src_qpu][dst_qpu]["weight"] -= min_weight
                if graph[src_qpu][dst_qpu]["weight"] == 0:
                    graph.remove_edge(src_qpu, dst_qpu)
```

`process_remaining_teledata` 处理 residual graph 中不再属于已处理 cycles 的 moves。不能简单逐个 one-way move landing，因为目标 QPU 可能暂时没有空的有效 target comp wire。第一版应把 residual graph 分解成 unit paths：path 上每条边选择一个 logical qubit；如果当前 destination QPU 没有空 target comp wire，就沿该 QPU 的一个 outgoing move 继续延长 path，让这个 outgoing move 在同一个 batch 的 remote phase 释放一个有效 comp wire。path 的最后一个 destination QPU 必须已经有空 target comp wire，或者其空位会由 path 中后续 remote phase 释放；找不到可落地 path 时直接报错。

```python
def process_remaining_teledata(graph, state, partition, target_partition, network, policy):
    while graph.number_of_edges() > 0:
        path = find_landing_safe_unit_path(graph, target_partition, state)
        if path is None:
            raise RuntimeError("remaining teledata graph has no landing-safe path")

        moves = []
        for src_qpu, dst_qpu, q in path:
            graph[src_qpu][dst_qpu]["qubits"].remove(q)
            moves.append(TeledataMove(
                logical_qid=q,
                src_qpu=int(src_qpu),
                dst_qpu=int(dst_qpu),
            ))

            decrement_or_remove_edge(graph, src_qpu, dst_qpu)

        _process_teledata_batch(
            moves=moves,
            label_prefix="teledata-path",
            state=state,
            partition=partition,
            target_partition=target_partition,
            network=network,
            policy=policy,
        )
```

`find_landing_safe_unit_path` 只负责在 residual graph 里找一条可落地 path。因为 partition list 顺序不定义 wire，path tail 只需要目标 QPU 当前存在任意空的有效 target comp wire；如果目标 QPU 没有空 comp wire，则必须沿着该 QPU 的一个 outgoing edge 继续延长 path，让该 outgoing move 在同一个 batch 的 remote phase 释放一个 comp wire。如果遇到 cycle，说明 cycle 阶段没有完全消掉，应报错或回到 cycle 处理。

```python
def find_landing_safe_unit_path(graph, target_partition, state):
    for src_qpu, dst_qpu in list(graph.edges()):
        path = []
        visited_qpus = set()
        curr_src = int(src_qpu)
        curr_dst = int(dst_qpu)
        curr_q = graph[curr_src][curr_dst]["qubits"][0]

        while True:
            if curr_src in visited_qpus:
                raise RuntimeError("remaining teledata path unexpectedly forms a cycle")
            visited_qpus.add(curr_src)

            path.append((curr_src, curr_dst, curr_q))

            if has_free_target_comp_wire(state, target_partition, curr_dst):
                return path

            next_edge = choose_outgoing_edge(graph, curr_dst)
            if next_edge is None:
                raise RuntimeError(
                    f"remaining teledata destination has no free target comp wire "
                    f"and no outgoing move: qpu={curr_dst}"
                )
            _, next_dst = next_edge
            blocking_q = graph[curr_dst][next_dst]["qubits"][0]

            curr_src = curr_dst
            curr_dst = next_dst
            curr_q = blocking_q

    return None
```

#### 同 QPU local compaction / placement repair

同 QPU compaction 是 partition transition 的一部分，不是普通 payload。它不处理“partition list 顺序变化”，只处理 resident qubit 当前 wire 不属于目标 comp 区间的情况。典型例子：

```text
prev QPU0:   [0, 1]
target QPU0: [1]
```

logical qubit `1` 没有跨 QPU。因为 target QPU0 只有 1 个 computation wire，有效 comp 区间只有 wire `0`，所以如果 `q1` 仍在旧 wire `1`，Evaluator 必须通过本地操作把它迁移到某个有效 comp wire。这里不是因为 `q1` 在 target list 中的 index 是 `0`，而是因为旧 wire `1` 在 target shape 里不再是 comp wire。

第一版实现可以用保守策略：cross-QPU teledata 全部完成后，每个 QPU 检查属于该 QPU 的 resident qubits。如果某个 resident 的 `local_wire >= len(target_partition[qpu])`，则选择一个空的 target comp wire，执行本地 `swap(old_wire, free_target_comp_wire)`，flush 后调用 `reserve_wire(... owner_kind="resident")` 更新 ownership。所有本地 swaps 都按 `COMM_PRIMITIVE` 进入 `add_local_ops + flush_local_ops`。

伪代码：

```python
def repair_local_compaction(state, transition_partition, target_partition, network, policy):
    target_qpus = partition_qpus(target_partition)

    for q, qpu_id in target_qpus.items():
        loc = state.logical_pos[q]
        require(loc.qpu_id == qpu_id)

        target_comp_count = len(target_partition[qpu_id])
        if loc.local_wire < target_comp_count:
            continue

        dst_wire = select_free_target_comp_wire(state, target_partition, qpu_id)
        add_local_ops(
            qpu_id=qpu_id,
            ops=[(Gate("swap"), [loc.local_wire, dst_wire])],
            kind=COMM_PRIMITIVE,
        )
        flush_local_ops(qpu_ids=[qpu_id])
        reserve_wire(
            state,
            transition_partition,
            qpu_id=qpu_id,
            local_wire=dst_wire,
            wire_kind="comp",
            owner_kind="resident",
            logical_qid=q,
            label="local-compaction-target",
        )
```

如果以后需要支持“强制指定 local wire 顺序”的模式，那才需要恢复完整 same-QPU reindex / permutation 调度。该调度必须注意：

- `reserve_wire(... owner_kind="resident")` 会释放旧 resident owner，因此调用前必须 flush 所有仍引用旧 wire 的 pending ops。
- 如果需要用 communication wire 打破 local permutation cycle，应先把 temp comm wire 作为 `protocol` 占用，append swap，flush，然后再 reserve 成 `resident`。不能在 swap 前直接 `reserve_wire(... owner_kind="resident")` 到 temp wire。

示例：

```python
tmp_comm = reserve_wire(
                state,
                partition,
                qpu_id=qpu_id,
                wire_kind="comm",
                owner_kind="protocol",
                label="local-reindex-temp-protocol",
)
add_local_ops(qpu_id=qpu_id, ops=[(Gate("swap"), [src_wire, tmp_comm.local_wire])], kind=COMM_PRIMITIVE)
flush_local_ops(qpu_ids=[qpu_id])
release_wire(state, tmp_comm.qpu_id, tmp_comm.local_wire, "protocol")
reserve_wire(
    state,
    partition,
    qpu_id=qpu_id,
    local_wire=tmp_comm.local_wire,
    wire_kind="comm",
    owner_kind="resident",
    logical_qid=q,
    label="local-reindex-temp",
)
```

实现时要注意：

- `reserve_wire(... owner_kind="resident")` 会释放旧 resident owner，因此调用前必须 flush 所有仍引用旧 wire 的 pending ops。
- 如果 resident 已经位于有效 target comp 区间，不要为了匹配 partition list index 去移动它。
- 同 QPU compaction 引入的 swap 不计入 `payload_gate_num`。
- 如果没有可用 communication wire 打破 cycle，应直接报错，而不是静默修改 owner。

#### Teledata 计数字段

Partition transition / teledata 推荐计数规则：

- remote move 成本计入全局 e-pair、hop、remote fidelity 字段。
- source-side `cx/h` 等通信协议本地门计入 local gate / fidelity 成本。
- target-side landing swap 计入 local gate / fidelity 成本。
- 同 QPU local compaction swap 计入 local gate / fidelity 成本。
- landing swap 不计入 `payload_gate_num`，因为它不是原始 payload。
- teledata 本身不是 simple telegate，也不是 explicit `CommOp`，因此不应增加 `telegate_exec_events` 或 `comm_block_events`；它只改变 record 间 runtime placement 并贡献 remote/local protocol 成本。

### 需要保持的约束

Evaluator 每次更新 physical state 后应保持：

- 同一 QPU 上 `wire_phy_map[qpu]` 内非空 physical qubit 不重复。
- logical qubit 指向的 `(qpu_id, local_wire)` 必须在该 QPU local wire 范围内。
- 每个 `state.logical_pos[q]` 指向的 wire 必须有 `WireOwner(kind="resident", logical_qid=q)`。
- 每个 `WireOwner(kind="resident", logical_qid=q)` 必须被 `state.logical_pos[q]` 反向指回。
- `WireOwner(kind="entangled_copy", logical_qid=q)` 必须引用一个已经存在 resident owner 的 logical qubit，但不能更新 `state.logical_pos[q]`。
- `WireOwner(kind="protocol")` 只能表示通信协议临时资源，不能作为普通 payload operand 的默认 runtime location。
- communication local wire 必须位于该 QPU 的 reserved comm wire 区间内。
- physical qubit 不越过 backend qubit 数。
- `wire_phy_map[qpu]` 的长度应等于 `len(partition[qpu]) + comm_slot_reserve`。
- `wire_owners[qpu]` 的长度应等于 `len(partition[qpu]) + comm_slot_reserve`。

这些约束由 `_validate_physical_state()` 统一检查。新增 teledata、telegate 或 CommOp 语义时，应优先让状态维护满足这些约束，再考虑成本字段归因。

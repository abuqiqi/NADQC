# Deferred Evaluator TODO

## Goal

Add a deferred local evaluation mode so all baselines can be compared under a
consistent local routing/transpile policy.

Current evaluator behavior flushes local buffers frequently. That makes
`local_payload_gate_num` depend on record/CommOp/teledata boundaries and can
inflate payload routing cost. The deferred mode should still maintain runtime
qubit locations exactly, but postpone local transpilation until the full
processed execution stream has been collected.

The first implementation should prioritize a reliable deferred total local gate
count. Strict payload/protocol/teledata attribution is not required for deferred
mode.

## Target Modes

- `immediate`: current behavior.
- `deferred`: append local ops into long-lived per-QPU physical-sized buffers,
  then transpile once per QPU at the end.

## Implementation Tasks

1. Add evaluator configuration
   - Add a local evaluation mode option, e.g. `local_eval_mode`.
   - Default to `immediate` to preserve current behavior.
   - Allow `deferred` from config/CLI without changing baseline code.
   - Add a deferred routing switch, e.g. `deferred_route_local_gates`.
   - In deferred mode, use one routing choice for payload, CommOp protocol, and
     teledata:
     - `true`: transpile with the real backend coupling map.
     - `false`: transpile without backend coupling map.
   - Do not use per-category route flags in deferred mode.

2. Extend `EvaluationState`
   - Add long-lived per-QPU deferred local buffers.
   - Each deferred buffer should be initialized with `backend.num_qubits`, not
     the current partition size plus communication reserve.
   - This makes each local wire a stable QPU-local wire in a physical-sized
     address space and avoids resizing/compacting the buffer during replay.
   - Keep existing runtime state updates:
     - logical qubit to current QPU
     - logical qubit to runtime/local wire
     - QPU wire occupancy/reuse
   - Separate logical/runtime wire occupancy from physical layout updates. In
     deferred mode, the physical layout should not be updated after every
     protocol fragment because there is no intermediate transpile result.
   - In deferred replay, maintain runtime wire ownership only. Do not rely on
     intermediate `wire_phy_map` extraction.

3. Refactor local op append/flush path
   - In `immediate`, preserve current flush behavior.
   - In `deferred`, do not flush only because the record changes or the category
     changes.
   - Append payload, CommOp protocol, and teledata local gates into the same
     per-QPU buffer so they share one local routing configuration.
   - Existing call sites that currently require a flush for physical-layout
     refresh must be audited:
     - `add_local_ops`
     - `flush_local_ops`
     - `_process_teledata_batch`
     - `_move_resident_with_local_swap`
     - `_repair_local_compaction`
     - CommOp protocol helpers
   - In deferred mode, these call sites should update logical/runtime state and
     should not force local transpilation.

4. Add semantic barriers where needed
   - Insert barriers around:
     - CommOp protocol boundaries
     - measurement/reset/classical dependency boundaries
     - teledata migration boundaries
     - wire reuse boundaries
   - Barriers are for semantic ordering only, not for category attribution.

5. Final deferred flush
   - At the end of evaluation, transpile each QPU buffer once.
   - Use the same local hardware/routing configuration across all baselines:
     - coupling map
     - basis gates
     - optimization level
     - seed
     - layout/routing method
     - initial wire identity policy
   - Accumulate total local costs:
     - `local_gate_num`
     - `local_fidelity_loss`
     - `local_fidelity`
     - `local_fidelity_log_sum`
   - Deferred mode does not need to provide strict
     `local_payload_gate_num`/`local_comm_protocol_gate_num`/`local_teledata_gate_num`
     breakdown.
   - If the existing CSV schema requires a closed breakdown, put deferred local
     gates into `local_uncategorized_gate_num` and keep
     `local_gate_breakdown_gap == 0`.
   - Initial layout in deferred mode should be deterministic and stable for the
     full per-QPU circuit. Do not depend on intermediate `wire_phy_map` updates.

6. Preserve and validate ExecCosts accounting
   - Ensure `local_gate_breakdown_num == local_gate_num`.
   - Ensure `local_gate_breakdown_gap == 0` in normal final evaluator output.
   - In deferred mode, category breakdown is intentionally not interpreted as
     payload/protocol/teledata attribution unless a later implementation adds a
     validated attribution path.

7. Add tests
   - Unit test that `immediate` mode remains unchanged on small examples.
   - Unit test that `deferred` does not flush at record/category boundaries.
   - Unit test that deferred buffers are initialized with `backend.num_qubits`.
   - Unit test that qubit migration still updates runtime locations correctly.
   - Regression test that breakdown totals close with zero gap.

8. Run QFT4x50 comparison
   - Re-run WBCP, AutoComm, and NAVI Hybrid under `immediate`.
   - Re-run the same outputs under `deferred`.
   - Compare:
     - `local_gate_num`
     - `local_uncategorized_gate_num`
     - `F_eff`
   - Main question: does NAVI's total local gate overhead over WBCP shrink when
     routing is deferred and consistent?

## Design Constraints

- Do not skip runtime location maintenance. Deferred mode only delays local
  transpilation; it does not relax distributed execution semantics.
- Do not merge physical wires across QPUs. A migrated logical qubit must move
  from one QPU-local wire identity to another.
- Keep `immediate` as the default until deferred results are validated.
- Deferred mode reports total local cost first. Do not over-interpret category
  breakdown columns in deferred output.

## Decisions To Confirm

1. Deferred routing policy
   - Proposed: in `deferred`, route the full per-QPU circuit with either the
     real backend coupling map or no coupling map, controlled by one switch.
   - Rationale: payload, CommOp protocol, and teledata should share the same
     local hardware configuration.
   - Confirmed direction: do not split routing policy by category in deferred
     mode.

2. Breakdown attribution
   - Proposed: do not attempt strict payload/protocol/teledata attribution in
     the first deferred implementation.
   - Confirmed direction: report one unified deferred local cost.

3. Physical layout updates during deferred replay
   - Proposed: do not update `wire_phy_map` during replay. Maintain only
     logical/runtime wire ownership, then use one deterministic initial layout
     for the final per-QPU transpile.
   - Rationale: there is no valid intermediate transpiled circuit from which to
     extract a new physical layout.
   - Confirmed direction: initialize each QPU buffer with `backend.num_qubits`
     to avoid compacting/resizing the deferred circuit.

4. Scope of first implementation
   - Proposed: implement `deferred` only. Leave category attribution and
     additional deferred variants as later experiments.

## Acceptance Criteria

- Existing evaluator tests pass in `immediate` mode.
- Deferred mode produces internally consistent `ExecCosts` with zero breakdown
  gap.
- QFT4x50 output includes comparable breakdown columns for WBCP, AutoComm, and
  NAVI Hybrid.
- The result can answer whether NAVI's higher local gate count is caused mainly
  by frequent flush boundaries or by the mapping/partition itself.

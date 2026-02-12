# Kill Switch Runbook

This runbook documents the manual kill-switch procedure for scheduled trading runs.

## Scope

- Workflow: `.github/workflows/monthly_rebalance_execute.yml`
- Runtime check: `check_kill_switch()` in `src/execution_safe.py`
- Control variable: `KILL_SWITCH_ENABLED`

## Trigger Conditions

Enable kill switch immediately if any of the following occur:

- Unexpected order bursts or duplicate submissions.
- Alpaca rejects multiple orders with unclear cause.
- Data quality anomaly (stale/incomplete bars) that bypassed upstream checks.
- Strategy logic regression suspected after deployment.
- Market event requiring manual pause.

## Activation Procedure

1. Open GitHub repo settings:
`Settings -> Secrets and variables -> Actions -> Variables`
2. Set `KILL_SWITCH_ENABLED=true`.
3. Optional context:
set `KILL_SWITCH_REASON` in workflow environment/runner env for audit clarity.
4. Re-run (or wait for next) rebalance workflow.

Expected behavior:

- `scripts/execute_rebalance_safe.py` aborts before order submission.
- Discord receives blocked/failure summary with reason.
- `latest_state.json` execution status reflects failed/blocked run.

## Verification Checklist

1. Confirm workflow log contains kill-switch enabled message.
2. Confirm no submitted orders in Alpaca for that run window.
3. Confirm Discord alert was posted.
4. Confirm state branch update was pushed.

## Deactivation Procedure

1. Set `KILL_SWITCH_ENABLED=false`.
2. Run one `DRY_RUN=true` rebalance execution first.
3. Review markdown summary:
- target trades
- what changed today
- execution results all dry-run
4. If clean, run normal paper/live schedule.

## Rollback Path

If issues persist after deactivation:

1. Re-enable `KILL_SWITCH_ENABLED=true`.
2. Set `DRY_RUN=true` and `SMOKE_TEST=false`.
3. Investigate using latest run payload in Discord + `latest_state.json.execution.last_run`.


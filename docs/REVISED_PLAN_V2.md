# revised_implementation_plan_v2.md

## Framework Integration Goals
- **Qlib**: Replace custom feature engineering and model training with a standardized pipeline.
- **LEAN**: Provide an alternative, industry-standard backtesting path.
- **Postgres**: Maintain as the central state hub.

## Phase 1: Qlib Integration (High Priority)
- [ ] Install Qlib and its dependencies.
- [ ] Implement `integrations/qlib_pipeline/dataset.py` to bridge Alpaca data -> Qlib.
- [ ] Implement `scripts/run_qlib_train.py` for model training.
- [ ] Implement `scripts/run_qlib_score.py` to write weekly ranks to Postgres.

## Phase 2: LEAN Integration (Medium Priority)
- [ ] Add `scripts/setup_lean.sh` to clone and prepare the LEAN engine.
- [ ] Implement a C# or Python strategy for LEAN that reads our Postgres signals (Alpha Model).
- [ ] Build `src/integrations/lean_export.py` to ingest LEAN results into the dashboard database.

## Phase 3: Bot & Dashboard Harmonization
- [ ] Update `src/bot.py` to fetch signals from the `signals` table populated by Qlib.
- [ ] Update Streamlit dashboard to compare Native Backtest vs. LEAN Backtest.

## Acceptance Criteria
1. `run_qlib_score.py` successfully populates Postgres with ~500 ticker scores.
2. `src/bot.py` executes trades on Tuesday based on the Monday scores in DB.
3. LEAN backtest can be triggered via CLI and results appear in Streamlit.
4. All safety switches (BROKER_MODE) remain functional and enforced.




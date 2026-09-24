# Capability scorecard (generated)

```
cube: 3 parity, 1 gap
  [GAP ] gap    fan_and_chasm_trap_detection  -- close: a multi-target check refusing two fan-outs from one source (small); adopt: Cube needs its own server
  [ok  ] parity many_to_many_needs_bridge
  [ok  ] parity one_to_one_is_safe
  [ok  ] parity primary_key_required  -- a foreign key must target a key
dagster: 4 parity, 2 beyond, 1 gap
  [ok  ] parity asset_freshness  -- materialization key changes iff inputs change
  [ok  ] parity asset_metadata  -- row counts and staging time on every materialization
  [ok  ] parity idempotent_rematerialize
  [ok  ] parity io_manager_swap  -- swapping the store is changing the sink
  [ok  ] beyond multi_asset_atomic_publish  -- several datasets flip together or not at all 
  [ok  ] beyond multi_object_asset  -- Dagster models one object per asset
  [GAP ] gap    partitions  -- close: per-key incremental staging in the sinks (large); adopt: not possible, partitions need Dagster's run/asset model
dbt: 7 parity, 2 gap
  [GAP ] gap    accepted_values_test  -- close: a manifest `checks` field run in stage; adopt: none needed (dbt tests need dbt runtime)
  [ok  ] parity constraints_before_insert  -- PK/FK declared in staging DDL
  [ok  ] parity constraints_not_on_views
  [GAP ] gap    docs_export  -- close: inverse of graph_from_spec (Frictionless descriptor); adopt: frictionless-py, optional
  [ok  ] parity fk_survives_swap  -- reduces the FK-across-swap risk
  [ok  ] parity incremental_skip  -- state:modified analogue
  [ok  ] parity model_contract  -- output columns and order are fixed by the manifest
  [ok  ] parity relationships_test
  [ok  ] parity staging_then_swap
dlt: 2 parity, 2 beyond
  [ok  ] beyond load_state_marker  -- dlt _dlt_loads has no staged-vs-published distinction
  [ok  ] parity nested_to_child_rows  -- polygon parts become zone_part rows keyed by (zone, part_no)
  [ok  ] beyond references_enforced  -- dlt references are annotations, not verified
  [ok  ] parity schema_contract  -- dlt 'freeze' mode: an unexpected schema change stops the load
metricflow: 7 parity
  [ok  ] parity ambiguous_path  -- two routes: refuse, or the caller picks with via=
  [ok  ] parity cycles_rejected
  [ok  ] parity fan_out_rejected  -- primary -> foreign
  [ok  ] parity hop_cap  -- MetricFlow limits multi-hop joins to 2
  [ok  ] parity many_to_one_traversal
  [ok  ] parity null_keys_rejected
  [ok  ] parity transitive_join
parquet_export: 4 beyond
  [ok  ] beyond column_names_and_order
  [ok  ] beyond column_types_preserved  -- types survive the SQL round trip
  [ok  ] beyond export_equals_original  -- the export is taken from the published version
  [ok  ] beyond unpublished_fails
snakemake: 1 beyond
  [ok  ] beyond freshness_via_storage_plugin  -- dbt/Dagster need their own scheduler for this
sqlmesh: 1 parity
  [ok  ] parity virtual_layer_repoint
```

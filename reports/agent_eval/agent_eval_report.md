# Agent Evaluation Report

| Metric | Value |
| --- | ---: |
| case_count | 30 |
| completed_count | 30 |
| tool_decision_accuracy | 1.0000 |
| tool_selection_accuracy | 0.9000 |
| tool_argument_accuracy | 1.0000 |
| evidence_hit_rate | 1.0000 |
| citation_accuracy | 0.9545 |
| false_positive_count | 0 |
| false_positive_rate | 0.0000 |
| false_negative_count | 0 |
| false_negative_rate | 0.0000 |
| average_latency_ms | 18378.5697 |

## Case Results

| ID | Tools | Decision | Selection | Arguments | Evidence | Citation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| agent-001 | search_codebase | True | True | True | True | True |
| agent-002 | search_codebase | True | True | True | True | True |
| agent-003 | analyze_code | True | True | True | True | True |
| agent-004 | analyze_log | True | True | True | True | True |
| agent-005 | complete_code | True | True | True | True | True |
| agent-006 | generate_tests, complete_code | True | False | True | True | True |
| agent-007 | generate_dockerfile | True | True | True | None | True |
| agent-008 | - | True | True | True | None | None |
| agent-009 | - | True | True | True | None | None |
| agent-010 | analyze_code | True | True | True | None | None |
| agent-011 | search_codebase, analyze_code | True | True | True | True | True |
| agent-012 | search_codebase | True | True | True | True | True |
| agent-013 | search_codebase | True | True | True | True | True |
| agent-014 | search_codebase, search_codebase, search_codebase, search_codebase | True | False | True | True | True |
| agent-015 | analyze_code | True | True | True | True | True |
| agent-016 | analyze_code | True | True | True | True | True |
| agent-017 | analyze_log | True | True | True | None | None |
| agent-018 | analyze_log | True | True | True | True | True |
| agent-019 | complete_code | True | True | True | True | True |
| agent-020 | complete_code | True | True | True | True | True |
| agent-021 | generate_tests | True | True | True | True | False |
| agent-022 | generate_tests | True | True | True | True | True |
| agent-023 | generate_dockerfile | True | True | True | True | True |
| agent-024 | - | True | True | True | None | None |
| agent-025 | - | True | True | True | None | None |
| agent-026 | - | True | True | True | None | None |
| agent-027 | analyze_code | True | True | True | None | None |
| agent-028 | search_codebase, analyze_code, complete_code, search_codebase | True | False | True | True | True |
| agent-029 | search_codebase, analyze_code | True | True | True | True | True |
| agent-030 | search_codebase, analyze_code | True | True | True | True | True |

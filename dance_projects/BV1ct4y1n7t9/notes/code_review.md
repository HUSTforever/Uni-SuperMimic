# Independent code review and fixes

Fresh read-only Codex review covered new retarget and video-preparation paths. No critical findings; three important findings were reproduced and fixed:

1. Empty/EOF-truncated videos could exit FFmpeg successfully. Added real decoding/coverage checks and EOF rejection. VFR colour/timestamp regression also caught misleading container duration (0.68s header vs1.04s decoded), so source timing uses decoded timestamps.
2. Existing NPZ/JSON could be overwritten. Both targets are checked before processing; NPZ and metadata use exclusive publication. Uppercase NPZ filename regression passes without hidden extension append.
3. Per-frame grounding erased airborne trajectories. The approximation is now explicitly named support; preserve-flight is selectable through both human grounding and robot conversion, with documented world-height limitations. Current dance uses support.

Each fix observed failing tests before implementation and passing tests afterwards. An independent deterministic concurrency regression also replaced a shared-directory glob assertion with validation of the current call's own temporary-file cleanup. Production rendering cleanup was unchanged.

Final verification:192 passed,2 skipped,6 slow deselected; Ruff and git diff whitespace checks pass. This does not claim arbitrary jump/occluded/multi-person video support. Source/robot camera-heading comparison was separately audited and corrected before accepting delivery.

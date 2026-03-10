"""
Article 2 Plotting: Multi-Phase Transfer Learning Experiments.

This package provides analysis and visualization scripts for the
multi-phase transfer learning experiments (Phase 0-3).

Directory structure:
- outputs/: LaTeX tables (.tex files)
- figures/: Plots (.pdf, .png)
- cache/: Cached WandB data (.csv)

Scripts:
- phase0_baselines.py: Baselines analysis (deferred)
- phase1_architecture.py: Architecture search + selection
- phase2_techniques.py: Transfer techniques + selection (deferred)
- phase2_global_results.py: Phase 2 global fine-tuning technique comparison
- phase3_hero.py: Hero model refinement (deferred)
- phase3_global_results.py: Phase 3 global hero model refinement
- cross_phase_summary.py: Cross-phase summary (deferred)

Usage:
    cd Experiments/article_2
    python phase1_architecture.py [--refresh] [--select EXPERIMENT_ID]
"""

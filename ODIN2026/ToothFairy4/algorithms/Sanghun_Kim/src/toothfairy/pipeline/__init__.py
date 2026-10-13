"""The reproduction pipeline: the programs that build the caches, train the stages,
run rejection-sampling fine-tuning, train and apply the narrative rewriter, and score
a run.

Each module here is one program with its own ``main(argv)``. The top-level ``main_*.py``
entry points group them into the five things one actually runs, so the order of the
recipe is visible from the repository root rather than from a directory listing:

    toothfairy.cli.cache    cache_dental, cache_teeth
    toothfairy.cli.train    toothfairy.training.stage_a, toothfairy.training.trainer
    toothfairy.cli.rft      rft_rollout, rft_filter_cache, rft_rescore, rft_finetune, rft_merge
    toothfairy.cli.rewrite  rewrite_pairs, rewrite_pairs_from_run, rewrite_train, rewrite_apply
    toothfairy.cli.eval     generate, captioning, final_score

Every module is also runnable on its own (``python -m toothfairy.pipeline.generate --help``)
for the cases where only one step has to be repeated.
"""

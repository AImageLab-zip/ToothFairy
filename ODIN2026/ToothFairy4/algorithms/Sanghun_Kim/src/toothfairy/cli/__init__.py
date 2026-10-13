"""Entry points — the six things one actually runs, in the order one runs them.

    python -m toothfairy.cli.cache     both feature caches
    python -m toothfairy.cli.train     the three-stage curriculum
    python -m toothfairy.cli.rft       rejection-sampling fine-tuning
    python -m toothfairy.cli.rewrite   the narrative rewriter (train / apply)
    python -m toothfairy.cli.eval      generate -> captioning -> official score
    python -m toothfairy.cli.report    one volume -> one report

Each is thin: it fixes the arguments the recipe calls for, or dispatches to one of the
programs in `toothfairy.pipeline`, which is where the work lives. Every pipeline program is
also runnable on its own (`python -m toothfairy.pipeline.generate --help`) when only one
step has to be repeated.
"""

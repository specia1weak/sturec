from pathlib import Path
import runpy
import sys


# New architecture note:
# SequenceSetting(element_setting=existing_item_setting) is the direct
# replacement for the old SharedVocabSeqSetting pattern, because the
# sequence container reuses the element setting's vocab and embedding.

target = Path(__file__).with_name("ml1m-new.py")
sys.argv = [
    str(target),
    "--dataset_name",
    "movie-lens-shared-new-seqset",
    "--experiment_name",
    "ml1m-shared-new",
    *sys.argv[1:],
]
runpy.run_path(str(target), run_name="__main__")

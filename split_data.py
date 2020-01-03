import split_folders

# Split with a ratio
split_folders.ratio('data', output="split", seed=1337, ratio=(.8, .2))
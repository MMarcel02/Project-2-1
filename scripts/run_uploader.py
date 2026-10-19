# so idea is:
#
#   after complete a run, will probably have all the hyperparameter configs in memory
#   also all the hardware specs in memory, and then once run is completed can get the
#   throughput and steps to convergence etc etc
#
#   Lets say all this is inside of some kind of run_data class
#   so then the process is:
#
#   save run locally in data/runs/
#   attempt upload
#   if upload fails update local journal in data/
#   if succeeds this time delete from journal



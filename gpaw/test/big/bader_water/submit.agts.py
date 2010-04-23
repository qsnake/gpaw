def agts(queue):
    calc = queue.add('bader_water.py', ncpus=8, walltime=4)
    queue.add('bader_plot.py', walltime=5, deps=[calc])


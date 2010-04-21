def agts(queue):
    jobs = [queue.add('ruslab.py', tmax=5 * 60, ncpus=8),
            queue.add('ruslab.py H', tmax=5 * 60, ncpus=8),
            queue.add('ruslab.py N', tmax=5 * 60, ncpus=8),
            queue.add('ruslab.py O', tmax=5 * 60, ncpus=16),
            queue.add('molecules.py', tmax=20, ncpus=8)]
    queue.add('results.py', ncpus=1, deps=jobs)

def agts(queue):
    job = queue.add('HAl100.py')
    queue.add('stm.agts.py', ncpus=1, deps=[job])

if __name__ == '__main__':
    import sys
    from gpaw.test import wrap_pylab
    wrap_pylab()
    sys.argv = ['', 'HAl100.gpw']
    execfile('stm.py')

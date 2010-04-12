from gpaw.test.big.environment import Collector, DryProfile, AGTSRunner

c = Collector()
ensemble = c.get_ensemble()

d = DryProfile()
r = AGTSRunner(d)


r.run(ensemble)

print ensemble

# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

import pymc3 as pm
import numpy as np

# Datos de ejemplo
datos = np.random.randn(100)

# Modelo Bayesiano
with pm.Model():
    mu = pm.Normal('mu', mu=0, sd=1)
    sigma = pm.HalfNormal('sigma', sd=1)
    observaciones = pm.Normal('observaciones', mu=mu, sd=sigma, observed=datos)
    
    # Realizar inferencia con MCMC
    traza = pm.sample(1000, tune=1000)

# Imprimir los resultados
pm.traceplot(traza)

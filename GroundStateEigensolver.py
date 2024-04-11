from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
import qiskit.circuit.library
import qiskit_nature.settings as settings
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms import NumPyEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
import numpy as np
from qiskit.opflow import MatrixOp
import matplotlib.pyplot as plt
from ChemProblem import ChemProb
import json
settings.use_pauli_sum_op = False
settings.B3LYP_WITH_VWN5 = True

'''Tried Molecules
"H 0 0 0; H 0 0 0.735" : Result = -1.1373; Real = -1.17
"Li 0 0 0; H 0 0 1.619"
"D 0 0 0; H 0 0 0.761"
'''

numsteps = range(1000)
# #[
#   [-1.57079619e+00, 3.64753999e-06, -5.78152655e-01, -3.13959364e+00, -3.14468552e+00, 1.99903858e-03, -6.28627819e+00],
#   [4.19602975e-07, 4.71238864e+00, 7.04595506e+00, 1.99908763e-03, -6.28627829e+00, 3.14359173e+00, -3.14468563e+00],
#   [-3.14164168e+00, 1.57080952e+00, -1.23177674e+00, 2.00023326e-03, -6.28627990e+00, -3.13959337e+00, -3.14468666e+00]
# ]
#range(1000)
driver = PySCFDriver(
    atom = "H 0 0 0; H 0 0 0.735",
    #atom="Li 0 0 0; H 0 0 1.619",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

es_problem = driver.run()

print("Run Chemistry Problem")
mapperbk = BravyiKitaevMapper()
mapperjw = JordanWignerMapper()


npsolver = NumPyEigensolver()

results = []
for usedmapper in [mapperjw]:
    #,mapperbk]:
    #ansatzs = [ansatz,heansatz]
    ansatzs = ['uccsd']
    for ans in ansatzs:
        params=[]
        for i in numsteps:
            cp = ChemProb(ans,usedmapper)
            #initstate = [-1.57078773e+00,  3.14159107e+00,  2.17396898e+00,  3.14359187e+00, -3.14468573e+00,  1.99896299e-03, -6.28627823e+00]
            vqes = VQE(Estimator(),cp.ansatz,SLSQP())
            calc = GroundStateEigensolver(usedmapper,vqes)
            res = calc.solve(cp.esp)
            newr = min(res.eigenvalues)
            results.append(newr)
            if newr < -1.85:
                print(res.raw_result.optimal_point)
            params.append(res.raw_result.optimal_point)
            # if i % 100 == 0:
            #     print("Step ",i," of ",numsteps)
        print("/n")
        namee = ans+usedmapper.__class__.__name__+str(numsteps)
        print("Mapper:", usedmapper.__class__.__name__)
        print("Ansatz:", ans)
        # Set up the NumPyEigensolver

        # Get the first three lowest eigenvalues
        # bfres,nres = np.linalg.eig(cp.ham)
        # bfres = np.real(list(bfres))
        # bfres.sort()
        # bfres = np.unique(bfres)[:7]
        # print("Brute Force Results: ", bfres)
        results_array = np.array(results)
        mean = np.mean(results_array)
        median = np.median(results_array)
        std_dev = np.std(results_array)
        min_val = np.min(results_array)
        max_val = np.max(results_array)
        print("Mean:", mean)
        print("Median:", median)
        print("Standard Deviation:", std_dev)
        print("Minimum Value:", min_val)
        print("Maximum Value:", max_val)
        print("\n\n")
        print(results)
        print(params)
        print(namee)
        # params = [list(p) for p in params]
        # Save results array to a JSON file
        # with open(namee+'results.json', 'w') as f:
        #     json.dump(results_array.tolist(), f)

        # with open(namee+'params.json', 'w') as f:
        #     json.dump(params, f)

        # # Create a dictionary for data analysis results
        # data_analysis_results = {
        #     'mean': mean,
        #     'median': median,
        #     'std_dev': std_dev,
        #     'min_val': min_val,
        #     'max_val': max_val
        # }

        # # Save data analysis results to a JSON file
        # with open(namee+'data_analysis_results.json', 'w') as f:
        #     json.dump(data_analysis_results, f)
        
        results = []
        
        


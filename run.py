import numpy as np
import matplotlib.pyplot as plt
import qiskit
from ChemProblem import ChemProb
from  chemaqcmain import aqcpqcSolver as aqcpqcs
import json
import time

iterations = 1
thirdder = False
nullspace = False
nullderiv = False
stepnums = [100]
thresholdparam = 1
mapList = ["jw"]
save = True
ansatzList = ['uccsd']
moleculestring = "H 0 0 0; H 0 0 0.735"

for steps in stepnums:
    for map in mapList:
        for ansatznm in ansatzList:
            start = time.time()
            results = []
            initparams=[]
            finalparams=[]
            initsvs = []
            finalsvs = []
            hyperparams = []
            namee = str(map)+'_'+str(ansatznm)+'_'+str(steps)+'_'
            for i in range(iterations):
                cp = ChemProb(ansatznm,map,atomstr=moleculestring)
                nps = cp.ansatz.num_parameters
                nqs = cp.num_qubits
                print("Setup the Molecule and Ansatz")
                print("Number of qubits: ", nqs)
                caqc = aqcpqcs(ansatz_family=ansatznm,numqs=nqs,hamiltonian=cp.ham,nps=nps,steps=steps,ansatz=cp.ansatz,
                            use_third_derivatives = thirdder, use_null_space = nullspace, threshold=thresholdparam,use_null_derivatives = nullderiv)
                print("Setup the AQC-PQC Solver")
                trueground = caqc.adiabatic_solver()
                results.append(trueground)
                initparams.append(list(caqc.initthetas))
                finalparams.append(list(caqc.thetas))
                initsvs.append(list(caqc.initsv.data.flatten()))
                finalsvs.append(list(caqc.finalsv.data.flatten()))
            end = time.time()
            hyperparams = {"iterations": iterations, "steps": steps, "ansatz": ansatznm, "map": map,
                            "use_third_derivatives": thirdder, "use_null_space": nullspace,
                            "use_null_derivatives": nullderiv, "threshold":caqc.init_threshold, "time taken": end-start}
            print("Mapper: ", map)
            print("Ansatz: ", ansatznm)
            results_array = np.array(results)
            #+difference
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
            if save:
                if hyperparams["threshold"] > 1:
                    namee = namee + "noinitc"
                else:
                    namee = namee + "withinitc"
                with open(namee+'results.json', 'w') as f:
                    json.dump(results_array.tolist(), f)

                with open(namee+'initparams.json', 'w') as f:
                    json.dump(initparams, f)

                with open(namee+'finalparams.json', 'w') as f:
                    json.dump(finalparams, f)


                with open(namee+'initsvs.json', 'w') as f:
                    json.dump([[str(j) for j in i] for i in initsvs], f)
                
                with open(namee+'hyperparams.json', 'w') as f:
                    json.dump(hyperparams, f)
                with open(namee+'finalsvs.json', 'w') as f:
                    json.dump([[str(j) for j in i] for i in finalsvs], f)
                # Create a dictionary for data analysis results
                data_analysis_results = {
                    'mean': mean,
                    'median': median,
                    'std_dev': std_dev,
                    'min_val': min_val,
                    'max_val': max_val
                }

                # Save data analysis results to a JSON file
                with open(namee+'data_analysis_results.json', 'w') as f:
                    json.dump(data_analysis_results, f)



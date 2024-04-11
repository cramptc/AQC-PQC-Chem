from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import *
import numpy as np
from qiskit.circuit import ParameterVector

class QCir():
    def __init__(self, number_of_qubits, thetas,ansatz_family,ans):
        
        self.number_of_qubits = number_of_qubits
        self.thetas = thetas 
        #parameter_values = {param: thetas for param, thetas in zip(list(map(lambda x: x.name, ans.parameters)), thetas)}
        ans = ans.bind_parameters(thetas)
        self.qcir = QuantumCircuit(number_of_qubits)
        for i,instr in enumerate(ans):
            self.qcir.append(instr,instr.qubits)
            
    def printParams(self):
        print(self.parameters)




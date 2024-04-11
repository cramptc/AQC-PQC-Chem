from ChemProblem import ChemProb as cp
import numpy as np
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import math

identity = np.array([[1, 0], [0, 1]])
paulix = np.array([[0, 1], [1, 0]])

def initial_hamiltonian(number_of_qubits): #Here we define the initial Hamiltonian which we choose it to be -sigma_x for all qubits.
    initial_ham = np.zeros((2**number_of_qubits, 2**number_of_qubits))
    for qubit in range(number_of_qubits):
        initial_ham -= tensor_pauli(number_of_qubits, qubit, paulix)
    return initial_ham

def tensor_pauli(number_of_qubits, which_qubit, pauli_matrix): #This matrix represents a Pauli matrix acting in a single qubit to a higher dimensional Hilbert Spaces.

    if which_qubit == 0:
        matrix = pauli_matrix
    else:
        matrix = identity

    for qubit in range(1, number_of_qubits):
        if which_qubit == qubit:
            matrix = np.kron(pauli_matrix, matrix)
        else:
            matrix = np.kron(identity, matrix)

    return matrix

def calculate_expectation_value( matrix,circuit,number_of_qubits):
    sv1 = Statevector.from_label('0'*number_of_qubits)
    sv1 = sv1.evolve(circuit)
    expectation_value = sv1.expectation_value(matrix)
    return np.real(expectation_value)

def gencirc(nq): 
    ansatz = QuantumCircuit(nq)
    for i in range(nq-1):
        ansatz.x(i)
    for i in range(nq):
        ansatz.ry(Parameter('y'+str(i)), i)
    return ansatz

def init(number_of_qubits, thetas, ans):
    number_of_qubits = number_of_qubits
    thetas = thetas 
    #parameter_values = {param: thetas for param, thetas in zip(list(map(lambda x: x.name, ans.parameters)), thetas)}
    ans = ans.bind_parameters(thetas)
    qcir = QuantumCircuit(number_of_qubits)
    for i, instr in enumerate(ans):
        qcir.append(instr, instr.qubits)
    return qcir

cpp = cp('uccsd','bk',"Li 0 0 0; H 0 0 1.619")
cpa = cpp.ansatz
print(cpa.num_parameters)
angs = list(np.zeros(cpa.num_parameters-cpa.num_qubits))+[np.pi/2*np.power(-1,i) for i in range(cpa.num_qubits)]

circ = init(cpp.num_qubits,angs,cpa)
print(circ.parameters)
print(calculate_expectation_value(initial_hamiltonian(cpp.num_qubits),circ,cpp.num_qubits))
#print(circ.decompose().draw('mpl',filename='circ.png',scale=0.5))
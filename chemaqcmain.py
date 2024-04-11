from qiskit.visualization import *
import numpy as np
from quantum_circuit import QCir
from qiskit.quantum_info import Statevector
import scipy.optimize as optimize
import collections
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit.quantum_info.operators import Operator

pauliz = np.array([[1, 0], [0, -1]])
pauliy = np.array([[0, -1j], [1j, 0]])
identity = np.array([[1, 0], [0, 1]])
paulix = np.array([[0, 1], [1, 0]])

class aqcpqcSolver():
    def __init__(self,ansatz_family = "", nps = 0,hamiltonian=None,numqs=0, steps=0, ansatz=None,threshold=100,use_third_derivatives = False, use_null_space = True, use_null_derivatives = False,testing=False):
        #We start by defining the Pauli Matrices.
        self.pauliz = np.array([[1, 0], [0, -1]])
        self.paulix = np.array([[0, 1], [1, 0]])
        self.pauliy = np.array([[0, -1j], [1j, 0]])
        self.identity = np.array([[1, 0], [0, 1]])

        self.optimal_angles_per_step = [] 
        self.number_of_qubits = numqs
        self.number_of_parameters = nps
        self. H_0 = self.initial_hamiltonian()
        self.H_1 = hamiltonian
        self.ansatz = ansatz
        self.ansatz_family = ansatz_family
        self.steps = steps
        self.init_dif = 0
        self.energies_aqcpqc = [-numqs]
        self.energies_exact = [-numqs]
        self.energies = [-numqs]
        self.init_threshold = threshold
        if not testing:
            self.thetas = self.choose_initial_optimal_thetas()
        else:
            self.thetas = [0 for _ in range(self.number_of_parameters)]
        self.initthetas = np.copy(np.array(self.thetas))
        self.initsv = Statevector.from_label('0'*self.number_of_qubits).evolve(QCir(self.number_of_qubits, self.thetas, self.ansatz_family,ans=self.ansatz).qcir)
        self.use_null_space = use_null_space
        self.use_null_derivatives = use_null_derivatives
        self.use_third_derivatives = use_third_derivatives
        self.finalsv = None


    def startfunct(self):
        initial_eigenvalues = np.linalg.eigvals(self.H_1)
        print("Eigenvalues of the initial Hamiltonian:")
        initial_eigenvalues.sort()
        print(initial_eigenvalues)
        #for i in p:
        #    print(self.calculate_expectation_value(i, self.H_1))

    def tensor_pauli(self,number_of_qubits, which_qubit, pauli_matrix): #This matrix represents a Pauli matrix acting in a single qubit to a higher dimensional Hilbert Spaces.

        if which_qubit == 0:
            matrix = pauli_matrix
        else:
            matrix = self.identity

        for qubit in range(1, number_of_qubits):
            if which_qubit == qubit:
                matrix = np.kron(pauli_matrix, matrix)
            else:
                matrix = np.kron(self.identity, matrix)

        return matrix
            
    def initial_hamiltonian(self): #Here we define the initial Hamiltonian which we choose it to be -sigma_x for all qubits.
        initial_ham = np.zeros((2**self.number_of_qubits, 2**self.number_of_qubits))
        for qubit in range(self.number_of_qubits):
            initial_ham -= self.tensor_pauli(self.number_of_qubits, qubit, self.paulix)
        return initial_ham

    
    def choose_initial_optimal_thetas(self): #This function returns the initial optimal angles depending on the ansatz. 
        if self.ansatz_family == 'uccsd' and self.init_threshold <= 1:
            self.init_threshold = 1e-10
            init_thetas = [0,0,0,-np.pi/2,np.pi/2,-np.pi/2,+np.pi/2]
            return init_thetas
        hamiltonian_op = Operator(self.H_0)
        init_point= None
        #[0 for _ in range(self.number_of_parameters)]
        # Choose an optimizer
        optimizer = SLSQP()
        # Create the VQE solver
        vqe = VQE(Estimator(),self.ansatz, optimizer=optimizer, initial_point=init_point)
        i = 1
        dif = 1000
        while np.abs(dif) > self.init_threshold:
            if i == 100:
                print("Consider different initial angles or ansatz")
                break
            # Solve for the ground state energy
            print("VQE starting Point Iteration: "+str(i))
            vqe = VQE(Estimator(),self.ansatz, optimizer=optimizer, initial_point=init_point)
            result = vqe.compute_minimum_eigenvalue(operator=hamiltonian_op)
            dif = self.minimum_eigenvalue(self.H_0) - result.eigenvalue
            print(dif)
            i+=1
        self.init_dif = dif
        params = result.optimal_parameters
        init_thetas = list(params.values())

        print("Exact Starting Eigen: "+ str(self.minimum_eigenvalue(self.H_0)))
        print("Ansatz best Eigen: "+str(self.calculate_expectation_value(init_thetas,self.H_0)))
        return init_thetas

    def get_derivative(self, matrix, which_parameter, parameters):

        derivative = 0
        parameters_plus, parameters_minus = parameters.copy(), parameters.copy()
        parameters_plus[which_parameter] += np.pi/2
        parameters_minus[which_parameter] -= np.pi/2

        derivative += 1/2*self.calculate_expectation_value(parameters_plus, matrix)
        derivative -= 1/2*self.calculate_expectation_value(parameters_minus, matrix)
        
        return derivative
    
    def calculate_expectation_value(self, thetas, matrix): #This function calculates the expectation value of a given matrix (matrix).
        circuit = QCir(self.number_of_qubits, thetas, self.ansatz_family,ans=self.ansatz)
        sv1 = Statevector.from_label('0'*self.number_of_qubits)
        sv1 = sv1.evolve(circuit.qcir)
        expectation_value = sv1.expectation_value(matrix)
        return np.real(expectation_value)

    def get_third_derivatives(self, matrix, angles):
        third_derivatives = np.zeros((self.number_of_parameters, self.number_of_parameters, self.number_of_parameters))

        for parameter1 in range(self.number_of_parameters):
            for parameter2 in range(self.number_of_parameters):
                for parameter3 in range(self.number_of_parameters):

                    if parameter1<=parameter2 and parameter2<=parameter3:

                        third_order_thetas1, third_order_thetas2, third_order_thetas3, third_order_thetas4, third_order_thetas5, third_order_thetas6, third_order_thetas7, third_order_thetas8 = angles.copy(), angles.copy(), angles.copy(), angles.copy(), angles.copy(), angles.copy(), angles.copy(), angles.copy()

                        third_order_thetas1[parameter1] += np.pi/2
                        third_order_thetas1[parameter2] += np.pi/2
                        third_order_thetas1[parameter3] += np.pi/2

                        third_order_thetas2[parameter1] += np.pi/2
                        third_order_thetas2[parameter2] += np.pi/2
                        third_order_thetas2[parameter3] -= np.pi/2

                        third_order_thetas3[parameter1] -= np.pi/2
                        third_order_thetas3[parameter2] += np.pi/2
                        third_order_thetas3[parameter3] += np.pi/2

                        third_order_thetas4[parameter1] -= np.pi/2
                        third_order_thetas4[parameter2] += np.pi/2
                        third_order_thetas4[parameter3] -= np.pi/2

                        third_order_thetas5[parameter1] += np.pi/2
                        third_order_thetas5[parameter2] -= np.pi/2
                        third_order_thetas5[parameter3] += np.pi/2

                        third_order_thetas6[parameter1] += np.pi/2
                        third_order_thetas6[parameter2] -= np.pi/2
                        third_order_thetas6[parameter3] -= np.pi/2

                        third_order_thetas7[parameter1] -= np.pi/2
                        third_order_thetas7[parameter2] -= np.pi/2
                        third_order_thetas7[parameter3] += np.pi/2

                        third_order_thetas8[parameter1] -= np.pi/2
                        third_order_thetas8[parameter2] -= np.pi/2
                        third_order_thetas8[parameter3] -= np.pi/2

                        third_derivatives[parameter1, parameter2, parameter3] += self.calculate_expectation_value(third_order_thetas1, matrix)/8
                        third_derivatives[parameter1, parameter2, parameter3] -= self.calculate_expectation_value(third_order_thetas2, matrix)/8  
                        third_derivatives[parameter1, parameter2, parameter3] -= self.calculate_expectation_value(third_order_thetas3, matrix)/8
                        third_derivatives[parameter1, parameter2, parameter3] += self.calculate_expectation_value(third_order_thetas4, matrix)/8
                        third_derivatives[parameter1, parameter2, parameter3] -= self.calculate_expectation_value(third_order_thetas5, matrix)/8
                        third_derivatives[parameter1, parameter2, parameter3] += self.calculate_expectation_value(third_order_thetas6, matrix)/8
                        third_derivatives[parameter1, parameter2, parameter3] += self.calculate_expectation_value(third_order_thetas7, matrix)/8
                        third_derivatives[parameter1, parameter2, parameter3] -= self.calculate_expectation_value(third_order_thetas8, matrix)/8

                        third_derivatives[parameter1, parameter3, parameter2] = third_derivatives[parameter1, parameter2, parameter3]
                        third_derivatives[parameter2, parameter1, parameter3] = third_derivatives[parameter1, parameter2, parameter3]
                        third_derivatives[parameter2, parameter3, parameter1] = third_derivatives[parameter1, parameter2, parameter3]
                        third_derivatives[parameter3, parameter1, parameter2] = third_derivatives[parameter1, parameter2, parameter3]
                        third_derivatives[parameter3, parameter2, parameter1] = third_derivatives[parameter1, parameter2, parameter3]

        return np.array(third_derivatives)
    def calculate_instantaneous_hamiltonian(self, time):
        return (1-time)*self.H_0 + time*self.H_1

    def get_hessian_third_derivs(self, hessian_at_point, third_order_derivatives, epsilons):
        hessian_matrix = hessian_at_point.copy()

        for parameter in range(self.number_of_parameters):
            hessian_matrix += epsilons[parameter]*third_order_derivatives[parameter]

        return hessian_matrix
    
    def calculate_instantaneous_hamiltonian(self, time): #This function calculates the instantaneous Hamiltonian at a given time.
        return (1-time)*self.H_0 + time*self.H_1

    def find_indices(self, s, threshold=0.1):
        indices = []
        for k in range(self.number_of_parameters):
            if s[k] <= threshold:
                indices.append(k)

        return indices
    def minimum_instantaneous(self, time): #This gives the minimum energy at a given time.
        hamil = self.calculate_instantaneous_hamiltonian(time)
        eigenvalues, v1 = np.linalg.eigh(hamil)
        min_eig = np.min(eigenvalues)
        return np.real(min_eig)

    #We must first calculate the linear system of equations.

    def get_hessian_from_null_vectors(self, hessian, hessian_elements_dir_derivs, coefs):

        hessian_matrix = hessian.copy()
        for _ in range(len(coefs)):
            hessian_matrix += coefs[_]*hessian_elements_dir_derivs[_]

        return hessian_matrix
    
    def calculate_linear_system(self, hamiltonian, angles): #This function calculates the linear system of equations
        zero_order_terms = np.zeros((self.number_of_parameters,))
        first_order_terms = np.zeros((self.number_of_parameters, self.number_of_parameters))

        #We start with zero order terms.
        for parameter in range(self.number_of_parameters):
            zero_order_terms[parameter] += self.get_derivative(hamiltonian, parameter, angles)

        first_order_terms = self.get_hessian_matrix(hamiltonian, angles)
        return np.array(zero_order_terms), np.array(first_order_terms)

    def get_directional_diretivative(self, matrix, vector, parameters, h=0.001):
        
        shifted_parameters = [parameters[i] + h*vector[i] for i in range(self.number_of_parameters)]

        exp_value1, exp_value2 = self.calculate_expectation_value(shifted_parameters, matrix), self.calculate_expectation_value(parameters, matrix)
        directional_derivative = (exp_value1 - exp_value2)/h

        return directional_derivative
    def get_hessian_matrix(self, matrix, angles): #This function calculates the exact Hessian matrix.

        hessian = np.zeros((self.number_of_parameters, self.number_of_parameters))
    
        for parameter1 in range(self.number_of_parameters):
            for parameter2 in range(self.number_of_parameters):
                if parameter1 < parameter2:    
                    
                    hessian_thetas_1, hessian_thetas_2, hessian_thetas_3, hessian_thetas_4 = angles.copy(), angles.copy(), angles.copy(), angles.copy()

                    hessian_thetas_1[parameter1] += np.pi/2
                    hessian_thetas_1[parameter2] += np.pi/2


                    hessian_thetas_2[parameter1] -= np.pi/2
                    hessian_thetas_2[parameter2] += np.pi/2

                    hessian_thetas_3[parameter1] += np.pi/2
                    hessian_thetas_3[parameter2] -= np.pi/2

                    hessian_thetas_4[parameter1] -= np.pi/2
                    hessian_thetas_4[parameter2] -= np.pi/2

                    hessian[parameter1, parameter2] += self.calculate_expectation_value(hessian_thetas_1, matrix)/4
                    hessian[parameter1, parameter2] -= self.calculate_expectation_value(hessian_thetas_2, matrix)/4
                    hessian[parameter1, parameter2] -= self.calculate_expectation_value(hessian_thetas_3, matrix)/4
                    hessian[parameter1, parameter2] += self.calculate_expectation_value(hessian_thetas_4, matrix)/4

                    hessian[parameter2, parameter1] = hessian[parameter1, parameter2]
                    
                if parameter1 == parameter2:

                    hessian_thetas_1 , hessian_thetas_2 = angles.copy(), angles.copy()

                    hessian_thetas_1[parameter1] += np.pi
                    hessian_thetas_2[parameter1] -= np.pi
                    
                    hessian[parameter1, parameter1] += self.calculate_expectation_value(hessian_thetas_1, matrix)/4
                    hessian[parameter1, parameter1] += self.calculate_expectation_value(hessian_thetas_2, matrix)/4
                    hessian[parameter1, parameter1] -= self.calculate_expectation_value(angles, matrix)/2

        return hessian

    def minimum_eigenvalue(self, matrix):
        min_eigen = np.min(np.linalg.eigh(matrix)[0])
        return min_eigen
    def get_hessian_elements_directional_derivative(self, hessian, vector, parameters, hamiltonian, h=0.0001):
        
        hessian_shifted = self.get_hessian_matrix(hamiltonian,  [parameters[i] + h*vector[i] for i in range(self.number_of_parameters)])

        hessian_elements_dir_dervs = (hessian_shifted - hessian)/h
        return hessian_elements_dir_dervs

    def adiabatic_solver(self): #This is the main function that step by step interpolates from the initial Hamiltonian H_0 to the final Hamiltonian H_1
        lambdas = [i for i in np.linspace(0, 1, self.steps+1)][1:]
        optimal_thetas = self.thetas.copy()
        self.optimal_angles_per_step.append(optimal_thetas)
        for lamda in lambdas:
            print(lamda)
            hamiltonian = self.calculate_instantaneous_hamiltonian(lamda)
            zero, first = self.calculate_linear_system( hamiltonian, optimal_thetas)
            def equations(x):
                array = np.array([x[_] for _ in range(self.number_of_parameters)])
                equations = zero + first@array
                y = np.array([equations[_] for _ in range(self.number_of_parameters)])
                return y@y
            
            if not self.use_null_space:

                if not self.use_third_derivatives or np.all(zero) == 0:
                    
                    def minim_eig_constraint(x):
                        new_thetas = [optimal_thetas[i] + x[i] for i in range(self.number_of_parameters)]
                        return self.minimum_eigenvalue(self.get_hessian_matrix(hamiltonian, new_thetas))

                else:
                    
                    third_derivs = self.get_third_derivatives(hamiltonian, optimal_thetas)
                    hessian_at_optimal_point = self.get_hessian_matrix(hamiltonian, optimal_thetas)
            
                    def minim_eig_constraint(x):
                        return self.minimum_eigenvalue(self.get_hessian_third_derivs(hessian_at_optimal_point, third_derivs, x))
                    
                    
                    
                cons = [{'type': 'ineq', 'fun':minim_eig_constraint}]
                res = optimize.minimize(equations, x0 = [0 for _ in range(self.number_of_parameters)], constraints=cons,  method='SLSQP',  options={'disp': True}) 
                epsilons = [res.x[_] for _ in range(self.number_of_parameters)]
                print(res.tolerance) 
                
                optimal_thetas = [optimal_thetas[_] + epsilons[_] for _ in range(self.number_of_parameters)]


            else:

                u, s, v = np.linalg.svd(first)
                indices = self.find_indices(s)
                if len(indices) == 0:
                    indices = list(range(self.number_of_parameters))

                null_space_approx = [v[index] for index in indices]

                unconstrained_optimization = optimize.minimize(equations, x0 = [0 for _ in range(self.number_of_parameters)], method='SLSQP',  options={'disp': True, 'maxiter':400})
                print(unconstrained_optimization.tolerance) 
                epsilons_0 = unconstrained_optimization.x

                optimal_thetas = [optimal_thetas[i] + epsilons_0[i] for i in range(self.number_of_parameters)]


                def norm(x):
                    vector = epsilons_0.copy()
                    for _ in range(len(null_space_approx)):
                        vector += x[_]*null_space_approx[_]

                    norm = np.linalg.norm(vector)
                    return norm
                
                if not self.use_null_derivatives:

                    def minim_eig_constraint(x):
                        new_thetas = optimal_thetas.copy()
                        for _ in range(len(null_space_approx)):
                            new_thetas += x[_]*null_space_approx[_]
                        return self.minimum_eigenvalue(self.get_hessian_matrix(hamiltonian, new_thetas))
                    

                else:
                
                    #We can further make use that the null vectors are small! We construct a linear model of the Hessian using the directional derivatives of the Hessian elements, at the directional of the null vectors.
                    #First of all, we calculate the Hessian of the perturbed Hamiltonian at the previous optimal.
                    perturbed_hessian_at_optimal = self.get_hessian_matrix(hamiltonian, optimal_thetas)


                    #Then, we find the directional derivatives of the hessian elements at the optimal point.
                    directional_derivs_of_hessian_elements = []
                    for _ in range(len(null_space_approx)):
                        directional_derivs_of_hessian_elements.append(self.get_hessian_elements_directional_derivative(perturbed_hessian_at_optimal, null_space_approx[_], optimal_thetas, hamiltonian))


                    #Once we have the directional derivatives of the matrix elements, we can further proceed and impose the new minimum_eigenvalue constraint.

                    def minim_eig_constraint(x):
                        return self.minimum_eigenvalue(self.get_hessian_from_null_vectors(perturbed_hessian_at_optimal, directional_derivs_of_hessian_elements, x))

                
                
                cons = [{'type': 'ineq', 'fun':minim_eig_constraint}]
                constrained_optimization = optimize.minimize(norm, x0=[0 for _ in range(len(null_space_approx))], constraints=cons, method='SLSQP', options={'disp':True, 'maxiter':400})
                print(constrained_optimization.tolerance) 

                for _ in range(len(null_space_approx)):
                    optimal_thetas += constrained_optimization.x[_]*null_space_approx[_]


            self.thetas = optimal_thetas
            final_exp_value = self.calculate_expectation_value(optimal_thetas, hamiltonian)
            self.finalsv = Statevector.from_label('0'*self.number_of_qubits).evolve(QCir(self.number_of_qubits, optimal_thetas, self.ansatz_family,ans=self.ansatz).qcir)
        print(f'and the instantaneous expectation values is {final_exp_value}') 
        print(f'and the final angles are {optimal_thetas}')
        return final_exp_value

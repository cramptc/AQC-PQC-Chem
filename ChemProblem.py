from qiskit.algorithms import VQE
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
import qiskit_nature.settings as settings
import qiskit.circuit.library
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
settings.use_pauli_sum_op = False

class ChemProb():
    def __init__(self,ansatzname,mapperp, atomstr = "H 0 0 0; H 0 0 0.735"):
        driver = PySCFDriver(
            atom=atomstr,
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        es_problem = driver.run()
        self.esp = es_problem
        mapper = JordanWignerMapper()
        if mapperp == "bk":
            mapper = BravyiKitaevMapper()
        self.hamop = mapper.map(es_problem.second_q_ops()[0])
        self.ham = self.hamop.to_matrix()
        self.num_qubits = int(math.log2(self.ham.shape[0]))
        
        if ansatzname == "uccsd":
            ansatz = UCCSD(
                es_problem.num_spatial_orbitals,
                es_problem.num_particles,
                mapper,
                initial_state=HartreeFock(
                    es_problem.num_spatial_orbitals,
                    es_problem.num_particles,
                    mapper,
                ),   
            )
            for i in range(self.num_qubits):
                ansatz.ry(Parameter('y'+str(i)), i)
        elif ansatzname[:4] == "esu2":
            ansatz = qiskit.circuit.library.EfficientSU2(
                num_qubits=self.num_qubits,
                reps=1,
                su2_gates=["ry", "rz"],
                entanglement='linear',
                skip_final_rotation_layer=True,
                skip_unentangled_qubits=False,
                parameter_prefix="y",
                insert_barriers=False,
                initial_state=HartreeFock(
                es_problem.num_spatial_orbitals,
                es_problem.num_particles,
                mapper,
        ),
            )

        else:
            print("Invalid ansatz")
            return
        self.ansatz = ansatz
        self.mapper = mapper

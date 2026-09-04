from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.circuit.library import TwoLocal

driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
problem = driver.run()
mapper = JordanWignerMapper()
qubit_op = mapper.map(problem.second_q_ops()[0])

ansatz = TwoLocal(qubit_op.num_qubits, "ry", "cz", reps=2)
vqe = VQE(Estimator(), ansatz, COBYLA())
result = vqe.compute_minimum_eigenvalue(qubit_op)
print("Ground state energy:", result.eigenvalue.real)
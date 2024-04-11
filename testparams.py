import numpy as np
import matplotlib.pyplot as plt
import qiskit
from ChemProblem import ChemProb
from  chemaqcmain import aqcpqcSolver as aqcpqcs
import json
import textwrap
import math
mapList = ["bk"]
ansatzList = ['esu2']
steps = 20
def getdata(name):
    with open(name, 'r') as f:
        data = json.load(f)
    return data
cond = "noinitc"
path = "./UsefulData/JSONs/"
def innerproduct(v1,v2,uselog=False):
    if uselog:
        return np.array([math.log(np.power(np.linalg.norm(v1 @ v2[:,i]),2),10) for i in range(v2.shape[1])])
    return np.array([np.power(np.linalg.norm(v1 @ v2[:,i]),2) for i in range(v2.shape[1])])
def draw_bargraph(x_labels, values,othervals, ansatzname,map,cond,done=False,n="",build=None):
    title = nicename(ansatzname,map,cond)
    save = "".join([ansatzname,map,cond])
    positions = np.arange(len(x_labels))
    plt.bar(positions, values, width=0.1,label="Initial State")
    plt.bar(positions+0.1, othervals, width=0.1,label="Final State")
    if done:
        wrappedtitle = textwrap.fill(title, width=70)
        plt.xlabel('Energy Levels')
        plt.ylabel('Probability')
        plt.title(wrappedtitle)
        plt.xticks(positions, x_labels)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.legend()
        plt.savefig(save+str(n)+'.png')
        plt.clf()
def convert(a,eigs):
    l = [0]
    last = np.round(eigs[0],8)
    for i in range(len(a)):
        if np.round(eigs[i],8) != last:
            l.append(a[i])
        else:
            l[-1]+=a[i]
        last = np.round(eigs[i],8)
    return l
def nicename(ans,map,cond):
    ansatz = ""
    mapper = ""
    conder = ""
    if ans == "uccsd":
        ansatz = "UCCSD"
    if ans == "esu2":
        ansatz = "ESU2"
    if map == "jw":
        mapper = "Jordan-Wigner"
    if map == "bk":
        mapper = "Bravyi-Kitaev"
    if cond == "noinitc":
        conder = "No Initial Conditions"
    if cond == "initc":
        conder = "Initial Conditions"
    return "Using the "+mapper+" Mapping with the "+ansatz+" Ansatz: Initial and Final State Probability Distribution with "+conder
distinction = 0
for map in mapList:
    for ansatznm in ansatzList:
        namee = str(map)+'_'+str(ansatznm)
        cp = ChemProb(ansatznm,map)
        nps = cp.ansatz.num_parameters
        nqs = cp.num_qubits
        print("Setup the Molecule and Ansatz")
        print("Number of qubits: ", nqs)
        caqc = aqcpqcs(ansatz_family=ansatznm,numqs=nqs,hamiltonian=cp.ham,nps=nps,steps=steps,ansatz=cp.ansatz,
                        use_third_derivatives = False, use_null_space = True, threshold=100,use_null_derivatives = False,testing = True)
        ieigenvalues, eigenvectors = np.linalg.eigh(caqc.H_0)
        initg = eigenvectors
        feigenvalues, feigenvectors = np.linalg.eigh(caqc.H_1)
        finalg = feigenvectors
        initvecs = getdata(path+map+ansatznm+"aqcpqcnulltrueinitsvs"+cond+".json")
        finvecs = getdata(path+map+ansatznm+"aqcpqcnulltruefinalsvs"+cond+".json")
        print(len(initvecs))
        res = getdata(path+map+ansatznm+"aqcpqcnulltrueresults"+cond+".json")
        for ivec,fvec,r in zip(initvecs,finvecs,res):
            isv = ivec
            fsv = fvec
            a = np.array([np.complex128(x) for x in isv])
            b = np.array([np.complex128(x) for x in fsv])
            thispi = convert(innerproduct(a,initg),np.round(ieigenvalues,8))
            thispf = convert(innerproduct(b,finalg),np.round(feigenvalues,8))
            inite = np.abs(np.conj(a).T@caqc.H_0 @ a)
            finale = np.abs(np.conj(b).T@caqc.H_1 @ b)
            for i in range(np.abs(len(thispf)-len(thispi))):
                thispi.append(0)
            for i in range(np.abs(len(thispi)-len(thispf))):
                thispf.append(0)
            if r < -1.8:
                print("Initial Energy: ",inite)
                print("Final Energy: ",finale)
                print("True Ground State Energy: ",r)
                draw_bargraph([f'S{j}' for j in range(len(thispf))],thispi,thispf,ansatznm,map,cond,n=distinction,done=True)
                distinction+=1
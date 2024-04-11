
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


jwfinalparams = []

jwresults=[]




#####################################
perfectvqeuccsdparams = [[4.71238708, -4.71236496, 3.25337632, -3.14152344, -3.14153855, 3.14142405, -3.14191249],
[3.14159504e+00, -2.39849559e-04, 3.02980748e+00, 6.28337475e+00, -7.60840701e-04, -3.06814164e-05, -2.41691807e-04],
[-3.14136394e+00, -3.14124182e+00, -1.11696968e-01, 6.28347092e+00, 9.71865217e-05, -6.28260141e+00, -6.28339496e+00],
[3.14180173, 6.28292143, 4.82403086, 3.14217566, 3.14164871, -3.14029784, -3.1421462],
[1.57109205e+00, 4.71186489e+00, 1.45914340e+00, -6.28326275e+00, 4.26379332e-04, -6.28312423e+00, -6.28206908e+00],
[-3.14159616e+00, -3.04793798e-04, -4.60099662e+00, 3.14062336e+00, 3.14073233e+00, 3.14191799e+00, -3.14132644e+00]]

perfectvqeuccsdparams = np.mod(perfectvqeuccsdparams, 2*np.pi)
print(max(perfectvqeuccsdparams[2]))
for i, params in enumerate(perfectvqeuccsdparams):
  fig = plt.figure(figsize=(6, 6))
  print(params)
  ax = fig.add_subplot(111, polar=True)
  params += np.array(params[0])
  angles = np.linspace(0, 2 * np.pi, len(params), endpoint=True).tolist()
  ax.set_yticks(list(np.arange(0, 2.5 * np.pi, np.pi / 2)))
  ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$',r'$2\pi$'], fontsize=12)
  alm = list(range(0,len(params)))
  alm.append(0)
  ax.set_xticklabels(alm)
  ax.plot(angles, params, 'o-', linewidth=2)
  ax.fill(angles, params, alpha=0.25)

  #ax.set_thetagrids(np.degrees(angles[:-1]), fontsize=12)
  #ax.set_rlabel_position(0)
  #ax.set_ylim(0, 2 * np.pi)
  #ax.set_yticklabels([])

  plt.savefig(f"radarchart_{i+1}.png")
  plt.close(fig)
"""These are what generated the radial diagrams to illustrate how many different ground states get to the ground state"""
######################################
#####100steps########################
jwuccsdparaminit = [[3.1415982762053813, 1.5707944955932651, 0.22072739871532104, -1.5707950752261473, -4.712397992806711, -4.71240633678972, -1.5708060581034093]
,[3.141597186549971, 7.853967262078324, -0.4252919768180405, -1.5707958379282667, -4.712388769321492, 1.570800859310646, -1.5708061813408736]
]
jwuccsdparamfinal = [[3.14159828e+00, 1.57079450e+00, 2.20729322e-01, -3.52646551e-05, -6.28325848e+00, -3.14162792e+00, -3.14166582e+00],
           [3.14159719e+00, 7.85396726e+00, -4.25298025e-01, -3.52842134e-05, -6.28325847e+00, 3.14155739e+00, -3.14166584e+00]]

jwuccsdres = [-1.8369679888416122,-1.8369679889995048]


######################################
#########ESU2 - null space true########################
initparams = [[-1.5705678034387536, -1.5711153100391793, -1.5703791407720835, 1.5705982677534833, 0.0, 0.0, 0.0, 0.0]]
finalparams = [[-3.03808284e+00, -3.13388386e+00, 6.44678184e-02, 1.44608531e+00, -2.75607976e-07, -1.03887833e-05, 1.41699174e-05, 8.24443820e-07]]
bk = [-1.2498243868865202]
"""This happened 10/10 times exactly with vqe intialisation"""
#######################################

correcteigenvalues = [-1.85727503, -1.25633907, -1.24458455, -1.16063174, -0.88272215]

######################################


'''
We dont have a group
Eddie
quicksstart
use scratch
Match python qiskitaer cuda versions w eachother
create bash files to submit jobs
qacct -j [id]]
use interactive session to test

'''
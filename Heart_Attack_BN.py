import os
import matplotlib.pyplot as plt
import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pyAgrum.lib.notebook as gnb


# Defining Node (top->bottom, left->right)
bn=gum.BayesNet('Heart_Disease')
a=bn.add(gum.LabelizedVariable  ('age','Age',["27-44", "45-60", "61-77"]))
c=bn.add(gum.LabelizedVariable  ('chol','Cholesterol',["1-200", "201-240", "241-603"]))
bp=bn.add(gum.LabelizedVariable ('bp','Resting Blood Pressure',["0-120", "121-139", "140-200"]))
mhr=bn.add(gum.LabelizedVariable('mhr','Max Heart Rate Achieved',["60-100", "101-175", "176-220"]))      #Change 5 to list of values
ecg=bn.add(gum.LabelizedVariable('ecg','Resting ECG',["Normal", "ST", "LVH"]))
s=bn.add(gum.LabelizedVariable  ('sex','Sex',["F", "M"]))
h=bn.add(gum.LabelizedVariable  ('hd','Heart Disease',2))
cp=bn.add(gum.LabelizedVariable ('cp','Chest Pain',["ASY", "ATA", "NAP", "TA"]))

# Defining Arcs (top->bottom, left->right)
for link in [   (a,bp), (a,mhr),                                #Top row
                (c,bp), (bp, mhr), (mhr,h), (ecg, h),           #Mid row
                (h,s), (h, cp)]:                                #Bot row
    bn.addArc(*link)

### Adding weights (top->bottom, left->right)
#Age
bn.cpt(a).fillWith([0.191841, 0.241455, 0.566703])
print(bn.cpt(a))

#Cholesterol
bn.cpt(c).fillWith([0.194030, 0.327001, 0.478969])
print(bn.cpt(c))

#Resting EcG
bn.cpt(ecg).fillWith([0.601985, 0.206174, 0.191841])
print(bn.cpt(ecg))

#RestingBP | (Cholesterol, Age)
bn.cpt(bp)[{'chol': 0, 'age': 0}] = [0.514286, 0.257143, 0.228571]
bn.cpt(bp)[{'chol': 0, 'age': 1}] = [0.342105, 0.302632, 0.355263]
bn.cpt(bp)[{'chol': 0, 'age': 2}] = [0.103448, 0.310345, 0.586207]

bn.cpt(bp)[{'chol': 1, 'age': 0}] = [0.538462, 0.282051, 0.179487]
bn.cpt(bp)[{'chol': 1, 'age': 1}] = [0.282051, 0.434211, 0.276316]
bn.cpt(bp)[{'chol': 1, 'age': 2}] = [0.22, 0.24, 0.54]

bn.cpt(bp)[{'chol': 2, 'age': 0}] = [0.508197, 0.245902, 0.245902]
bn.cpt(bp)[{'chol': 2, 'age': 1}] = [0.227979, 0.378238, 0.393782]
bn.cpt(bp)[{'chol': 2, 'age': 2}] = [0.224490, 0.255102, 0.520408]
print(bn.cpt(bp))

#Heart Rate | (Resting Blood Pressure, Age)
bn.cpt(mhr)[{'age': 0, 'bp': 0}] = [0.012048, 0.807229, 0.180723]
bn.cpt(mhr)[{'age': 0, 'bp': 1}] = [0.025641, 0.769231, 0.205128]
bn.cpt(mhr)[{'age': 0, 'bp': 2}] = [0.030303, 0.757576, 0.212121]

bn.cpt(mhr)[{'age': 1, 'bp': 0}] = [0.087838, 0.871622, 0.040541]
bn.cpt(mhr)[{'age': 1, 'bp': 1}] = [0.096257, 0.860963, 0.042781]
bn.cpt(mhr)[{'age': 1, 'bp': 2}] = [0.097561, 0.871951, 0.030488]

bn.cpt(mhr)[{'age': 2, 'bp': 0}] = [0.218182, 0.781818, 0.0]
bn.cpt(mhr)[{'age': 2, 'bp': 1}] = [0.088235, 0.911765, 0.0]
bn.cpt(mhr)[{'age': 2, 'bp': 2}] = [0.120968, 0.870968, 0.008065]
print(bn.cpt(mhr))

#P(HeartDisease|MaxHR, RestingECG)
bn.cpt(h)[{'mhr': 0, 'ecg': 0}] = [0.444444, 0.555556]
bn.cpt(h)[{'mhr': 0, 'ecg': 1}] = [0.211538, 0.788462]
bn.cpt(h)[{'mhr': 0, 'ecg': 2}] = [0.136364, 0.863636]

bn.cpt(h)[{'mhr': 1, 'ecg': 0}] = [0.392638, 0.607362]
bn.cpt(h)[{'mhr': 1, 'ecg': 1}] = [0.485961, 0.514039]
bn.cpt(h)[{'mhr': 1, 'ecg': 2}] = [0.346939, 0.653061]

bn.cpt(h)[{'mhr': 2, 'ecg': 0}] = [0.866667, 0.133333]
bn.cpt(h)[{'mhr': 2, 'ecg': 1}] = [0.833333, 0.166667]
bn.cpt(h)[{'mhr': 2, 'ecg': 2}] = [0.8, 0.2]
print(bn.cpt(h))

#P(ChestPainType|HeartDisease)
bn.cpt(cp)[{'hd': 0}] = [0.2550, 0.3675, 0.3175, 0.0600]
bn.cpt(cp)[{'hd': 1}] = [0.771203, 0.047337, 0.142012, 0.39448]
print(bn.cpt(cp))

#P(Sex|HeartDisease)
bn.cpt(s)[{'hd': 0}] = [0.35, 0.65]
bn.cpt(s)[{'hd': 1}] = [0.098619, 0.901381]
print(bn.cpt(s))

plt.imshow(gimg.export(bn))
plt.show()
plt.imshow(gimg.exportInference(bn))
plt.show()

# Testing ability to make inferences using given probabilities
ie=gum.VariableElimination(bn)
ie.setEvidence({bp:2, a:0})
ie.makeInference()
print(ie.posterior("chol"))

# Saving network as bifxml file
gum.saveBN(bn,"Heart_Attack_BN.bifxml")
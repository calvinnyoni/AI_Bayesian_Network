import os
import matplotlib.pyplot as plt
import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pyAgrum.lib.notebook as gnb


# Defining Node (top->bottom, left->right)
dn = gum.InfluenceDiagram()
a=dn.add(gum.LabelizedVariable  ('age','Age',["27-44", "45-60", "61-77"]))
c=dn.add(gum.LabelizedVariable  ('chol','Cholesterol',["1-200", "201-240", "241-603"]))
bp=dn.add(gum.LabelizedVariable ('bp','Resting Blood Pressure',["0-120", "121-139", "140-200"]))
mhr=dn.add(gum.LabelizedVariable('mhr','Max Heart Rate Achieved',["60-100", "101-175", "176-220"]))      #Change 5 to list of values
ecg=dn.add(gum.LabelizedVariable('ecg','Resting ECG',["Normal", "ST", "LVH"]))
s=dn.add(gum.LabelizedVariable  ('sex','Sex',["F", "M"]))
h=dn.add(gum.LabelizedVariable  ('hd','Heart Disease',2))
cp=dn.add(gum.LabelizedVariable ('cp','Chest Pain',["ASY", "ATA", "NAP", "TA"]))

# Defining Arcs (top->bottom, left->right)
for link in [   (a,bp), (a,mhr),                                #Top row
                (c,bp), (bp, mhr), (mhr,h), (ecg, h),           #Mid row
                (h,s), (h, cp),]:                               #Bottom row
    dn.addArc(*link)

## Special Nodes
# Decision Node
decision = dn.addDecisionNode(gum.LabelizedVariable("VisitDoctor", "VisitDoctor", ["Don't Visit Doctor", "Visit Doctor"]))
#  Utility Node
u = dn.addUtilityNode(gum.LabelizedVariable("U", "Utility", 1))

# More arcs
dn.addArc(decision, u) #decision -> utility
dn.addArc(h, u) #likelihood of disease -> utility

### Adding weights (top->bottom, left->right)
#Age
dn.cpt(a).fillWith([0.191841, 0.241455, 0.566703])
print(dn.cpt(a))

#Cholesterol
dn.cpt(c).fillWith([0.194030, 0.327001, 0.478969])
print(dn.cpt(c))

#Resting EcG
dn.cpt(ecg).fillWith([0.601985, 0.206174, 0.191841])
print(dn.cpt(ecg))

#RestingBP | (Cholesterol, Age)
dn.cpt(bp)[{'chol': 0, 'age': 0}] = [0.514286, 0.257143, 0.228571]
dn.cpt(bp)[{'chol': 0, 'age': 1}] = [0.342105, 0.302632, 0.355263]
dn.cpt(bp)[{'chol': 0, 'age': 2}] = [0.103448, 0.310345, 0.586207]

dn.cpt(bp)[{'chol': 1, 'age': 0}] = [0.538462, 0.282051, 0.179487]
dn.cpt(bp)[{'chol': 1, 'age': 1}] = [0.282051, 0.434211, 0.276316]
dn.cpt(bp)[{'chol': 1, 'age': 2}] = [0.22, 0.24, 0.54]

dn.cpt(bp)[{'chol': 2, 'age': 0}] = [0.508197, 0.245902, 0.245902]
dn.cpt(bp)[{'chol': 2, 'age': 1}] = [0.227979, 0.378238, 0.393782]
dn.cpt(bp)[{'chol': 2, 'age': 2}] = [0.224490, 0.255102, 0.520408]
print(dn.cpt(bp))

#Heart Rate | (Resting Blood Pressure, Age)
dn.cpt(mhr)[{'age': 0, 'bp': 0}] = [0.012048, 0.807229, 0.180723]
dn.cpt(mhr)[{'age': 0, 'bp': 1}] = [0.025641, 0.769231, 0.205128]
dn.cpt(mhr)[{'age': 0, 'bp': 2}] = [0.030303, 0.757576, 0.212121]

dn.cpt(mhr)[{'age': 1, 'bp': 0}] = [0.087838, 0.871622, 0.040541]
dn.cpt(mhr)[{'age': 1, 'bp': 1}] = [0.096257, 0.860963, 0.042781]
dn.cpt(mhr)[{'age': 1, 'bp': 2}] = [0.097561, 0.871951, 0.030488]

dn.cpt(mhr)[{'age': 2, 'bp': 0}] = [0.218182, 0.781818, 0.0]
dn.cpt(mhr)[{'age': 2, 'bp': 1}] = [0.088235, 0.911765, 0.0]
dn.cpt(mhr)[{'age': 2, 'bp': 2}] = [0.120968, 0.870968, 0.008065]
print(dn.cpt(mhr))

#P(HeartDisease|MaxHR, RestingECG)
dn.cpt(h)[{'mhr': 0, 'ecg': 0}] = [0.444444, 0.555556]
dn.cpt(h)[{'mhr': 0, 'ecg': 1}] = [0.211538, 0.788462]
dn.cpt(h)[{'mhr': 0, 'ecg': 2}] = [0.136364, 0.863636]

dn.cpt(h)[{'mhr': 1, 'ecg': 0}] = [0.392638, 0.607362]
dn.cpt(h)[{'mhr': 1, 'ecg': 1}] = [0.485961, 0.514039]
dn.cpt(h)[{'mhr': 1, 'ecg': 2}] = [0.346939, 0.653061]

dn.cpt(h)[{'mhr': 2, 'ecg': 0}] = [0.866667, 0.133333]
dn.cpt(h)[{'mhr': 2, 'ecg': 1}] = [0.833333, 0.166667]
dn.cpt(h)[{'mhr': 2, 'ecg': 2}] = [0.8, 0.2]
print(dn.cpt(h))

#P(ChestPainType|HeartDisease)
#cp=dn.add(gum.LabelizedVariable ('cp','Chest Pain',["ASY", "ATA", "NAP", "TA"]))
dn.cpt(cp)[{'hd': 0}] = [0.2550, 0.3675, 0.3175, 0.0600]
dn.cpt(cp)[{'hd': 1}] = [0.771203, 0.047337, 0.142012, 0.39448]
print(dn.cpt(cp))

#P(Sex|HeartDisease)
dn.cpt(s)[{'hd': 0}] = [0.35, 0.65]
dn.cpt(s)[{'hd': 1}] = [0.098619, 0.901381]
print(dn.cpt(s))

#Utilty table
dn.utility("U")[{'VisitDoctor':0,'hd':0}] =  0
dn.utility("U")[{'VisitDoctor':0,'hd':1}] = -50000
dn.utility("U")[{'VisitDoctor':1,'hd':0}] = -1100
dn.utility("U")[{'VisitDoctor':1,'hd':1}] = -22100

plt.imshow(gimg.export(dn))
plt.show()
plt.imshow(gimg.exportInference(dn))
plt.show()

# Testing ability to make inferences using given probabilities
ie = gum.ShaferShenoyLIMIDInference(dn)
ie.setEvidence({bp:2, a:0})
ie.makeInference()
print(ie.optimalDecision("VisitDoctor"))
print(ie.posteriorUtility("VisitDoctor"))

# Saving network as bifxml file
gum.saveBN(dn,"Heart_Attack_DN.bifxml")
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

# Defining Node (top->bottom, left->right)
bn=gum.BayesNet('Heart_Disease')
a=bn.add(gum.LabelizedVariable  ('a','Age',["27-44", "45-60", "61-77"]))
c=bn.add(gum.LabelizedVariable  ('c','Cholesterol',["0-200", "201-240", "241-603"]))
bp=bn.add(gum.LabelizedVariable ('bp','Resting Blood Pressure',["0-120", "121-139", "140-200"]))
mhr=bn.add(gum.LabelizedVariable('m','Max Heart Rate Achieved',["60-140", "141-175", "176-220"]))      #Change 5 to list of values
ecg=bn.add(gum.LabelizedVariable('ecg','Resting ECG',["Normal", "ST", "LVH"]))
e=bn.add(gum.LabelizedVariable  ('e','Exercise Induced Angina',["N", "Y"]))
s=bn.add(gum.LabelizedVariable  ('s','Sex',["F", "M"]))
h=bn.add(gum.LabelizedVariable  ('h','Heart Disease',2))
cp=bn.add(gum.LabelizedVariable ('cp','Chest Pain',["TA", "ATA", "NAP", "ASY"]))

# Defining Arcs (top->bottom, left->right)
for link in [   (a,bp), (a,mhr),                                #Top row
                (c,bp), (bp, mhr), (mhr,h), (ecg, h), (e,cp),   #Mid row
                (h,s), (h, cp)]:                                #Bot row
    bn.addArc(*link)

### Adding weights (top->bottom, left->right)
#Age
bn.cpt(a).fillWith([0.193900, 0.240741, 0.565359])
print(bn.cpt(a))

#Cholesterol
bn.cpt(c).fillWith([0.201072, 0.323056, 0.475871])
print(bn.cpt(c))

#RestingBP | (Cholesterol, Age)
bn.cpt(bp)[{'c': 0, 'a': 0}] = [0.500000, 0.289474, 0.210526]
bn.cpt(bp)[{'c': 0, 'a': 1}] = [0.3500, 0.2875, 0.3625]
bn.cpt(bp)[{'c': 0, 'a': 2}] = [0.103448, 0.310345, 0.586207]

bn.cpt(bp)[{'c': 1, 'a': 0}] = [0.538462, 0.282051, 0.179487]
bn.cpt(bp)[{'c': 1, 'a': 1}] = [0.289474, 0.434211, 0.276316]
bn.cpt(bp)[{'c': 1, 'a': 2}] = [0.22, 0.24, 0.54]

bn.cpt(bp)[{'c': 2, 'a': 0}] = [0.516129, 0.241935, 0.241935]
bn.cpt(bp)[{'c': 2, 'a': 1}] = [0.227979, 0.378238, 0.252525]
bn.cpt(bp)[{'c': 2, 'a': 2}] = [0.241935, 0.393782, 0.525253]
print(bn.cpt(bp))

#Heart Rate | (Resting Blood Pressure, Age)
bn.cpt(mhr)[{'a': 0, 'bp': 0}] = [0.500000, 0.289474, 0.210526]
bn.cpt(mhr)[{'a': 0, 'bp': 1}] = [0.500000, 0.289474, 0.210526]
bn.cpt(mhr)[{'a': 0, 'bp': 2}] = [0.500000, 0.289474, 0.210526]

bn.cpt(mhr)[{'a': 1, 'bp': 0}] = [0.500000, 0.289474, 0.210526]
bn.cpt(mhr)[{'a': 1, 'bp': 1}] = [0.500000, 0.289474, 0.210526]
bn.cpt(mhr)[{'a': 1, 'bp': 2}] = [0.500000, 0.289474, 0.210526]

bn.cpt(mhr)[{'a': 2, 'bp': 0}] = [0.500000, 0.289474, 0.210526]
bn.cpt(mhr)[{'a': 2, 'bp': 1}] = [0.500000, 0.289474, 0.210526]
bn.cpt(mhr)[{'a': 2, 'bp': 2}] = [0.500000, 0.289474, 0.210526]
print(bn.cpt(mhr))

# Testing ability to make inferences using given probabilities
ie=gum.VariableElimination(bn)
ie.setEvidence({bp:0, a:2})
ie.makeInference()
print(ie.posterior("c"))

# Saving network as bifxml file
gum.saveBN(bn,"Heart_Attack_BN.bifxml")
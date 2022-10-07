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
h=bn.add(gum.LabelizedVariable  ('h','Heart Disease',["N", "Y"]))
cp=bn.add(gum.LabelizedVariable ('cp','Chest Pain',["TA", "ATA", "NAP", "ASY"]))

# Defining Arcs (top->bottom, left->right)
for link in [   (a,bp), (a,mhr),                                #Top row
                (c,bp), (bp, mhr), (mhr,h), (ecg, h), (e,cp),   #Mid row
                (h,s), (h, cp)]:                                #Bot row
    bn.addArc(*link)

### Adding weights (breadth-first order)

#Age
bn.cpt(a).fillWith([0.193900, 0.240741, 0.565359])
print(bn.cpt(a))

#Cholesterol
bn.cpt(c).fillWith([0.201072, 0.323056, 0.475871])
print(bn.cpt(c))

#RestingBP | (Cholesterol, Age)
bn.cpt(bp)[{'c': 0, 'a': 0}] = [0.500000, 0.289474, 0.210526]
bn.cpt(bp)[{'c': 0, 'a': 1}] = [0.500000, 0.289474, 0.210526]
bn.cpt(bp)[{'c': 0, 'a': 2}] = [0.500000, 0.289474, 0.210526]

bn.cpt(bp)[{'c': 1, 'a': 0}] = [0.500000, 0.289474, 0.210526]
bn.cpt(bp)[{'c': 1, 'a': 1}] = [0.500000, 0.289474, 0.210526]
bn.cpt(bp)[{'c': 1, 'a': 2}] = [0.500000, 0.289474, 0.210526]

bn.cpt(bp)[{'c': 2, 'a': 0}] = [0.500000, 0.289474, 0.210526]
bn.cpt(bp)[{'c': 2, 'a': 1}] = [0.500000, 0.289474, 0.210526]
bn.cpt(bp)[{'c': 2, 'a': 2}] = [0.500000, 0.289474, 0.210526]
print(bn.cpt(bp))

# bn.cpt(c)[0] = [0.043478, 0.869565, 0.086957]
# bn.cpt(c)[1] = [0.021186, 0.877119, 0.101695]
# bn.cpt(c)[2] = [0.020772, 0.839763, 0.139466]



#Exercise induced Angina

# Saving network as bifxml file
gum.saveBN(bn,"Heart_Attack_BN.bifxml")
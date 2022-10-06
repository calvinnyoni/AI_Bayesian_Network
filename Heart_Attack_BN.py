import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

# Defining Node (breadth-first order)
bn=gum.BayesNet('Heart_Attack')
a=bn.add(gum.LabelizedVariable  ('a','Age',["27-39", "40-51", "51-63", "64-75"]))
c=bn.add(gum.LabelizedVariable  ('c','Cholesterol',["0-150", "150-300", "300-450"]))
e=bn.add(gum.LabelizedVariable  ('e','Exercise Induced Angina',2))
s=bn.add(gum.LabelizedVariable  ('s','Sex',2))
bp=bn.add(gum.LabelizedVariable ('bp','Resting Blood Pressure',5))
mhr=bn.add(gum.LabelizedVariable('m','Max Heart Rate Achieved',5))
ecg=bn.add(gum.LabelizedVariable('ecg','Resting ECG',5))
cp=bn.add(gum.LabelizedVariable ('cp','Chest Pain',4))
h=bn.add(gum.LabelizedVariable  ('h','Heart Attack',2))

# Defining Arcs (breadth-first order)
for link in [   (a,c), (a,mhr), (a,ecg), (c, bp), (e, cp),   #Parents
                (s,h), (bp,h), (mhr,h), (ecg, h), (cp, h)]: #All going to Heart Attack
    bn.addArc(*link)

### Adding weights (breadth-first order)

#Age
bn.cpt(a).fillWith([0.2, 0.3, 0.4, 0.1])
print(bn.cpt(a))

#Chol | Age
bn.cpt(c)[0] = [0.043478, 0.869565, 0.086957]
bn.cpt(c)[1] = [0.021186, 0.877119, 0.101695]
bn.cpt(c)[2] = [0.020772, 0.839763, 0.139466]
bn.cpt(c)[3] = [0.043478, 0.771739, 0.184783]
print(bn.cpt(c))


# Age vs Chol Level Counts Tabulation
# Age_Ranges          (27, 39]  (39, 51]  (51, 63]  (63, 75]
# Cholesterol_Ranges
# (0, 150]                   3         5         7         4
# (150, 300]                60       207       283        71
# (300, 450]                 6        24        47        17
# P(Chol|Age) Tabulation
# Age_Ranges          (27, 39]  (39, 51]  (51, 63]  (63, 75]
# Cholesterol_Ranges
# (0, 150]            0.043478  0.021186  0.020772  0.043478
# (150, 300]          0.869565  0.877119  0.839763  0.771739
# (300, 450]          0.086957  0.101695  0.139466  0.184783


# Saving network as bifxml file
gum.saveBN(bn,"Heart_Attack_BN.bifxml")
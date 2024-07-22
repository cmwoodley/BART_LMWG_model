from rdkit import Chem
from rdkit.Chem import AllChem
import pymc3 as pm
import arviz as az
import numpy as np
import pandas as pd
import dill
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

def get_pred(model, x):
    x_dist = model.μ.distribution.predict(np.array(x))
    pred = x_dist.mean(axis=0)
    x_reshape = x_dist.T
    ci = list(az.hdi(np.array(x), hxdi_prob=0.89) for x in x_reshape)
    ci_flat =np.stack(ci)
    ci_errors = (np.abs(ci_flat - pred.reshape(len(pred),1)))

    return x_dist, pred, ci, ci_errors

with open("../models/scaler.pkl", "rb") as f:
    scaler = dill.load(f)

with open("../models/g.pkl","rb") as f:
    model_g = dill.load(f)
with open("../models/gdbl.pkl","rb") as f:
    model_gdbl = dill.load(f)

xtrain_columns = ['ECFP_1', 'ECFP_9', 'ECFP_25', 'ECFP_41', 'ECFP_45', 'ECFP_46',
       'ECFP_59', 'ECFP_79', 'ECFP_94', 'ECFP_102', 'ECFP_106',
       'ECFP_117', 'ECFP_147', 'ECFP_173', 'ECFP_192', 'ECFP_197',
       'ECFP_203', 'ECFP_221', 'ECFP_249', 'ECFP_265', 'ECFP_283',
       'ECFP_294', 'ECFP_303', 'ECFP_305', 'ECFP_310', 'ECFP_322',
       'ECFP_366', 'ECFP_442', 'ECFP_486', 'ECFP_507', 'ECFP_572',
       'ECFP_573', 'ECFP_598', 'ECFP_628', 'ECFP_680', 'ECFP_684',
       'ECFP_718', 'ECFP_726', 'ECFP_728', 'ECFP_736', 'ECFP_781',
       'ECFP_802', 'ECFP_835', 'ECFP_841', 'ECFP_851', 'ECFP_875',
       'ECFP_926', 'ECFP_989', 'ECFP_1015', 'ECFP_1019', 'ECFP_1039',
       'ECFP_1057', 'ECFP_1064', 'ECFP_1087', 'ECFP_1088', 'ECFP_1104',
       'ECFP_1110', 'ECFP_1112', 'ECFP_1118', 'ECFP_1167', 'ECFP_1187',
       'ECFP_1329', 'ECFP_1343', 'ECFP_1357', 'ECFP_1385', 'ECFP_1410',
       'ECFP_1460', 'ECFP_1517', 'ECFP_1530', 'ECFP_1575', 'ECFP_1624',
       'ECFP_1665', 'ECFP_1691', 'ECFP_1776', 'ECFP_1829', 'ECFP_1844',
       'ECFP_1855', 'ECFP_1906', 'ECFP_1933', 'ECFP_1935', 'ECFP_1977',
       'ECFP_2001', 'FCFP_8', 'FCFP_224', 'FCFP_428', 'FCFP_792',
       'FCFP_806', 'FCFP_1304', 'FCFP_1395', 'FCFP_1668', 'FCFP_1727',
       'FCFP_1907', 'FCFP_1915', 'nRings']

def applicability(x, smiles):
    data = pd.read_csv("../data/raw/data.csv")
    train_unique = np.unique(data.iloc[np.where(data.label == "train")[0]].SMILES)

    smiles = np.array(smiles)

    test_unique = np.unique(smiles)
    xtrain = np.array(pd.read_csv("../data/split/xtrain.csv"))
    x = np.array(x)

    train_mol = [Chem.MolFromSmiles(x) for x in train_unique]
    test_mol = [Chem.MolFromSmiles(x) for x in test_unique]

    train_fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, useFeatures=True) for m in train_mol]
    test_fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, useFeatures=True) for m in test_mol]

    av_sim_tr = []
    for i in range(len(train_fp)):
        sim_tr = []
        for j in range(len(train_fp)):
            if i != j:
                sim_tr.append(Chem.DataStructs.FingerprintSimilarity(train_fp[j], train_fp[i], metric=Chem.DataStructs.TanimotoSimilarity))
        av_sim_tr.append(np.mean(sim_tr))
    ref_sim = np.mean(av_sim_tr)


    outliers_smiles = []
    for i in range(len(test_fp)):
        similarity = [Chem.DataStructs.FingerprintSimilarity(test_fp[i], fp, metric=Chem.DataStructs.TanimotoSimilarity) for fp in train_fp]
        similarity.sort()
        if np.mean(np.array(similarity)[-1:]) < ref_sim:
            outliers_smiles.append(i)

    tanimoto_out = list(([np.where(smiles == test_unique[i])[0] for i in outliers_smiles]))

    outliers = []

    for i in range(len(x)):
        inlier = True
        for j in range(len(xtrain.T)):
            if x[i,j].round(6) > max(xtrain[:,j].round(6)):
                inlier = False
                break
            elif x[i,j].round(6) < min(xtrain[:,j].round(6)):
                inlier = False
                break
        if inlier == False:
            outliers.append(i)

    outliers = list(outliers) + list(set(np.concatenate(tanimoto_out))-set(outliers))
    outliers.sort()

    domain_check = []
    for i in range(len(x)):
        if i in outliers:
            domain_check.append("Out")
        else:
            domain_check.append("In")

    return domain_check

def predict_single(smiles, conc):
    '''Predicts LogG′ and LogG′′ from a smiles string at a given concentration.
    
    Takes SMILES string as a string, and concentration as an integer or a float'''

    assert type(smiles) == str
    assert type(conc) == float or type(conc) == int

    mol = Chem.MolFromSmiles(smiles)

    fcfp = np.zeros((1,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=True)
    Chem.DataStructs.ConvertToNumpyArray(fp, fcfp)

    ecfp = np.zeros((1,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=False)
    Chem.DataStructs.ConvertToNumpyArray(fp, ecfp)

    num_rings = np.array([Chem.rdMolDescriptors.CalcNumRings(mol)])

    cols = ["nRings"] + ["FCFP_"+str(i) for i in range(2048)] + ["ECFP_"+str(i) for i in range(2048)]

    df = pd.DataFrame(np.concatenate([num_rings, fcfp, ecfp]), index=cols).T*conc
    df = pd.DataFrame(scaler.transform(df[xtrain_columns]), columns=xtrain_columns)

    g_dist, g_pred, gci, gci_errors = get_pred(model_g, df)
    gdbl_dist, gdbl_pred, gdblci, gdblci_errors = get_pred(model_gdbl, df)

    ## Plot Prediction Summary Graph

    fig = plt.figure(figsize=(14,10), layout="constrained")
    ax1 = plt.subplot(2,2,1)
    ax1.axis("off")
    ax1.imshow(Draw.MolToImage(mol, (500,400)))
    ax2 = plt.subplot(2,2,3)
    ax3 = plt.subplot(2,2,4)
    ax4 = plt.subplot(2,2,2)

    sns.kdeplot(g_dist, fill=True, label = "Distribution", ax = ax2, palette="viridis")
    ax2.axvline(g_pred,0,1, color="red", ls="--", label="Predicted Value")
    ax2.axvline(gci[0][0], 0,1, color="green", label="89% Bayesian CI")
    ax2.axvline(gci[0][1], 0,1, color="green")
    ax2.set_title("G′ Distribution",fontsize=20)
    ax2.set_xlabel("Predicted LogG′", fontsize=20)

    sns.kdeplot(gdbl_dist, fill=True, label = "Distribution", ax = ax3)
    ax3.axvline(gdbl_pred,0,1, color="red", ls="--", label="Predicted Value")
    ax3.axvline(gdblci[0][0], 0,1, color="green", label="89% Bayesian CI")
    ax3.axvline(gdblci[0][1], 0,1, color="green")
    ax3.set_title("G″ Distribution",fontsize=20)
    ax3.set_xlabel("Predicted LogG″", fontsize=20)

    for ax in [ax2,ax3]:
        ax.legend(fontsize=15, loc="upper left")
        ax.set_ylabel("Density", fontsize="20")
        ax.tick_params(axis='both', which='major', labelsize=15)


    for ax, pred in zip([ax2,ax3],[g_pred, gdbl_pred]):
        ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.7,ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.8, "Predicted Value:\n{}".format(np.round(pred,3)[0]), backgroundcolor="white", fontsize=15)

    ax1.set_title("Prediction Summary", fontsize=25)


    #report table

    results = pd.DataFrame([[g_pred[0], gdbl_pred[0]],[gci[0][0],gdblci[0][0]],[gci[0][1],gdblci[0][1]]],index=["Predicted Value","Lower CI","Upper CI"], columns=["Gprime","Gdblprime"]).T
    results["CI Range"] = results.iloc[:,2]-results.iloc[:,1]
    results

    ax4.axis("off")
    ax4.text(0.25,0.6,str(np.round(results.iloc[0,0],3)),fontsize=15)
    ax4.text(0.45,0.6,str(np.round(results.iloc[0,1],3)),fontsize=15)
    ax4.text(0.60,0.6,str(np.round(results.iloc[0,2],3)),fontsize=15)
    ax4.text(0.75,0.6,str(np.round(results.iloc[0,3],3)),fontsize=15)

    ax4.text(0.1,0.6,"LogG′",fontsize=15)
    ax4.text(0.25,0.75,"Predicted\nValue",fontsize=15)
    ax4.text(0.45,0.75,"Lower\nCI",fontsize=15)
    ax4.text(0.60,0.75,"Upper\nCI",fontsize=15)
    ax4.text(0.75,0.75,"CI\nRange",fontsize=15)

    ax4.text(0.1,0.45,"LogG″",fontsize=15)
    ax4.text(0.25,0.45,str(np.round(results.iloc[1,0],3)),fontsize=15)
    ax4.text(0.45,0.45,str(np.round(results.iloc[1,1],3)),fontsize=15)
    ax4.text(0.60,0.45,str(np.round(results.iloc[1,2],3)),fontsize=15)
    ax4.text(0.75,0.45,str(np.round(results.iloc[1,3],3)),fontsize=15)
        
    plt.show()

def predict_batch(smiles, conc):
    '''Predicts LogG′ and LogG′′ from a batch of smiles strings at given concentrations.
    
    Takes SMILES strings as list or 1D array, takes conc as list or 1D array'''
    
    mols = [Chem.MolFromSmiles(x) for x in smiles]

    num_rings = []
    fcfps = []
    ecfps = []
    for mol in mols:
        fcfp = np.zeros((1,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=True)
        Chem.DataStructs.ConvertToNumpyArray(fp, fcfp)
        fcfps.append(fcfp)

        ecfp = np.zeros((1,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=False)
        Chem.DataStructs.ConvertToNumpyArray(fp, ecfp)
        ecfps.append(ecfp)

        num_rings.append(np.array([Chem.rdMolDescriptors.CalcNumRings(mol)]))

    cols = ["nRings"] + ["FCFP_"+str(i) for i in range(2048)] + ["ECFP_"+str(i) for i in range(2048)]

    num_rings = np.stack(num_rings)
    fcfps = np.stack(fcfps)
    ecfps = np.stack(ecfps)

    df = pd.DataFrame(np.concatenate([num_rings,fcfps,ecfps],axis=1), columns=cols)
    for col in df.columns:
        df[col] = df[col]*conc
    df = pd.DataFrame(scaler.transform(df[xtrain_columns]), columns=xtrain_columns)

    g_dist, g_pred, gci, gci_errors = get_pred(model_g, df)
    gdbl_dist, gdbl_pred, gdblci, gdblci_errors = get_pred(model_gdbl, df)

    gci = np.stack(gci)
    gdblci = np.stack(gdblci)

    results = pd.DataFrame([g_pred, gci[:,0], gci[:,1], gdbl_pred, gdblci[:,0], gdblci[:,1]], index=["G′ Predicted Value","G′ Lower CI","G′ Upper CI","G″ Predicted Value","G″ Lower CI","G″ Upper CI"]).T
    results ["G′ CI Range"] = results.iloc[:,2]-results.iloc[:,1]
    results ["G″ CI Range"] = results.iloc[:,5]-results.iloc[:,4]
    results["Conc"] = conc
    results["In Domain?"] = applicability(df, smiles)

    return results[["G′ Predicted Value","G′ Lower CI","G′ Upper CI","G′ CI Range","G″ Predicted Value","G″ Lower CI","G″ Upper CI","G″ CI Range","Conc", "In Domain?"]]
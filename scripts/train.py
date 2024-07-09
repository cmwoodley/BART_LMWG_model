import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
import dill
import pymc3 as pm
import arviz as az

## Helper Functions

def variance_threshold_selector(data, threshold=0.00):
	selector = VarianceThreshold(threshold)
	selector.fit(data)
	return data[data.columns[selector.get_support(indices=True)]]

def get_ci_error(model, x, y, pred, error):
    x_dist = model.μ.distribution.predict(np.array(x))
    x_reshape = x_dist.T
    ci = list(az.hdi(np.array(x), hxdi_prob=0.89) for x in x_reshape)
    ci_flat =np.stack(ci)
    ci_errors = (np.abs(ci_flat - pred.reshape(len(pred),1)))

    error_lim = np.log10(10**y+np.array(error).reshape(1,-1))-y


    return x_dist, ci, ci_errors, error_lim

## Load in data

data = pd.read_csv("../data/raw/data.csv")
train = np.where(data.label == "train")[0]
test = np.where(data.label == "test")[0]

smiles_full = data.SMILES
mols_full = [Chem.MolFromSmiles(x) for x in smiles_full]
g = data.g
gdbl = data.gdbl

gtrain_error = data.err_g[train]
gtest_error = data.err_g[test] 
gdbltrain_error = data.err_gdbl[train]
gdbltest_error = data.err_gdbl[test] 

## Get Descriptors

df = pd.DataFrame()
df["nRings"] = [Chem.rdMolDescriptors.CalcNumRings(x) for x in mols_full]

fcfp = []
for mol in mols_full:
    ar = np.zeros((1,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=True)
    Chem.DataStructs.ConvertToNumpyArray(fp,ar)
    fcfp.append(ar)
fcfp = pd.DataFrame(np.stack(fcfp), columns=["FCFP_"+str(i) for i in range(2048)])
ecfp = []
for mol in mols_full:
    ar = np.zeros((1,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=False)
    Chem.DataStructs.ConvertToNumpyArray(fp,ar)
    ecfp.append(ar)
ecfp = pd.DataFrame(np.stack(ecfp), columns=["ECFP_"+str(i) for i in range(2048)])
df = pd.concat([ecfp,fcfp,df],axis=1)


for col in df.columns:
    df[col] = df[col]*data.conc

## Split into predetermined train, test and validation sets

xtrain = df.iloc[train]
xtest = df.iloc[test].reset_index(drop=True)

gtrain = g.iloc[train]
gtest = g.iloc[test]

gdbltrain = gdbl.iloc[train]
gdbltest = gdbl.iloc[test]


## Remove correlated and zero variance descriptors

scaler = StandardScaler()
VT = VarianceThreshold(threshold=0.0).fit(xtrain)
xtrain = pd.DataFrame(VT.transform(xtrain), columns=VT.get_feature_names_out())
corr_matrix = xtrain.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
xtrain = xtrain[[x for x in xtrain.columns if x not in to_drop]]
xtrain = pd.DataFrame(scaler.fit_transform(xtrain), columns=xtrain.columns)
xtest = pd.DataFrame(scaler.transform(xtest[xtrain.columns]), columns=xtrain.columns)

with open("../models/scaler.pkl", "wb") as f:
    dill.dump(scaler, f)

## Save split data

xtrain.to_csv("../data/split/xtrain.csv", index=False)
xtest.to_csv("../data/split/xtest.csv", index=False)

gtrain.to_csv("../data/split/gtrain.csv", index=False)
gtest.to_csv("../data/split/gtest.csv", index=False)

gdbltrain.to_csv("../data/split/gdbltrain.csv", index=False)
gdbltest.to_csv("../data/split/gdbltest.csv", index=False)

## BART requires y data as np array

gtrain = np.ravel(gtrain)
gtest = np.ravel(gtest)

gdbltrain = np.ravel(gdbltrain)
gdbltest = np.ravel(gdbltest)

## Build and save models

models = []
traces = []
for y,m,alpha in [(gtrain,25,0.2), (gdbltrain,25,0.4)]:
    with pm.Model() as model:
        σ = pm.HalfNormal('σ', 1)
        μ = pm.BART('μ', xtrain, np.array(y), m=m, alpha=alpha)
        y = pm.Normal('y', μ, σ, observed=np.array(y))
        trace = pm.sample(1000, cores=1, tune=1000, return_inferencedata=False, random_seed = 42)

    models.append(model)
    traces.append(trace)

model_g = models[0]
trace_g = traces[0]
model_gdbl = models[1]
trace_gdbl = traces[1]  

with open("../models/g.pkl","wb") as f:
    dill.dump(model_g, f)

with open("../models/gdbl.pkl","wb") as f:
    dill.dump(model_gdbl, f)

with open("../models/g_trace.pkl","wb") as f:
    dill.dump(trace_g, f)

with open("../models/gdbl_trace.pkl","wb") as f:
    dill.dump(trace_gdbl, f)

## Get Predictions and produce report

gtrain_pred = model_g.µ.distribution.predict(np.array(xtrain)).mean(axis=0)
gdbltrain_pred = model_gdbl.µ.distribution.predict(np.array(xtrain)).mean(axis=0)
gtest_pred = model_g.µ.distribution.predict(np.array(xtest)).mean(axis=0)
gdbltest_pred = model_gdbl.µ.distribution.predict(np.array(xtest)).mean(axis=0)


gtrain_pred_res = np.abs(gtrain_pred - gtrain)
gdbltrain_pred_res = np.abs(gdbltrain_pred - gdbltrain)
gtest_pred_res = np.abs(gtest_pred - gtest)
gdbltest_pred_res = np.abs(gdbltest_pred - gdbltest)


gtrain_dist, gtrain_ci, gtrain_ci_errors, gtrain_exp_errors = get_ci_error(model_g, xtrain, gtrain, gtrain_pred, gtrain_error)
gtest_dist, gtest_ci, gtest_ci_errors, gtest_exp_errors = get_ci_error(model_g, xtest, gtest, gtest_pred, gtest_error)
gdbltrain_dist, gdbltrain_ci, gdbltrain_ci_errors, gdbltrain_exp_errors = get_ci_error(model_gdbl, xtrain, gdbltrain, gdbltrain_pred, gdbltrain_error)
gdbltest_dist, gdbltest_ci, gdbltest_ci_errors, gdbltest_exp_errors = get_ci_error(model_gdbl, xtest, gdbltest, gdbltest_pred, gdbltest_error)

ci_errors_fullg = np.concatenate([gtrain_ci_errors, gtest_ci_errors])
ci_errors_1g = []
ci_errors_2g = []
for i in range(len(ci_errors_fullg)):
    ci_errors_1g.append(ci_errors_fullg[i][0])
    ci_errors_2g.append(ci_errors_fullg[i][1])
    
ci_errors_fullgdbl = np.concatenate([gdbltrain_ci_errors, gdbltest_ci_errors])
ci_errors_1gdbl = []
ci_errors_2gdbl = []
for i in range(len(ci_errors_fullgdbl)):
    ci_errors_1gdbl.append(ci_errors_fullgdbl[i][0])
    ci_errors_2gdbl.append(ci_errors_fullgdbl[i][1])

ci_errors_1g = pd.Series(ci_errors_1g, name="ci1_g")
ci_errors_2g = pd.Series(ci_errors_2g, name="ci2_g")
ci_errors_1gdbl = pd.Series(ci_errors_1g, name="ci1_gdbl")
ci_errors_2gdbl = pd.Series(ci_errors_2g, name="ci2_gdbl")

all_errg = pd.Series(list(gtrain_exp_errors[0]) + list(gtest_exp_errors[0]) , name = "err_g")
all_errgdbl = pd.Series(list(gdbltrain_exp_errors[0]) + list(gdbltest_exp_errors[0]) , name = "err_gdbl")

train_results = pd.DataFrame([gtrain, gtrain_pred, gtrain_pred_res, gdbltrain, gdbltrain_pred, gdbltrain_pred_res], index=["g", "g_pred", "g_pred_res", "gdbl", "gdbl_pred", "gdbl_pred_res"]).T
train_results["label"] = ["train" for x in range(len(train_results))]
test_results = pd.DataFrame([gtest, gtest_pred, gtest_pred_res, gdbltest, gdbltest_pred, gdbltest_pred_res], index=["g", "g_pred", "g_pred_res", "gdbl", "gdbl_pred", "gdbl_pred_res"]).T
test_results["label"] = ["test" for x in range(len(test_results))]



results = pd.concat([train_results, test_results], axis=0).reset_index(drop=True)

results = pd.concat([results, all_errg, ci_errors_1g, ci_errors_2g, all_errgdbl, ci_errors_1gdbl, ci_errors_2gdbl, smiles_full],axis=1)
results.to_csv("../reports/predictions.csv", index=False)

prediction_summary = pd.DataFrame()
for ind in [train, test]:
    gr2 = r2_score(results.g[ind], results.g_pred[ind])
    grmse = np.sqrt(mean_squared_error(results.g[ind], results.g_pred[ind]))
    gdblr2 = r2_score(results.gdbl[ind], results.gdbl_pred[ind])
    gdblrmse = np.sqrt(mean_squared_error(results.gdbl[ind], results.gdbl_pred[ind]))
    pred_rep = pd.DataFrame([gr2, grmse, gdblr2, gdblrmse],index=["gr2", "grmse", "gdblr2", "gdblrmse"]).T
    prediction_summary = pd.concat([prediction_summary, pred_rep], axis=0).reset_index(drop=True)
prediction_summary.rename({0:"train",1:"test"}, axis=0, inplace=True)
prediction_summary.to_csv("../reports/prediction_summary.csv", index=True)
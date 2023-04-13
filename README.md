<body>
	<h1>BART Model for the Rheological Property Prediction from LMWGs</h1>
	<p>This repository contains the dataset and scripts used to train BART models for the prediction of low molecular weight gelator rheological properties. The notebooks folder contains an example notebook for the prediction of rheological properties from smiles strings.</p>
	<p>For ease of use, we also provide a <a href="https://colab.research.google.com/github/cmwoodley/BART_LMWG_model/blob/master/notebooks/BART_LMWG.ipynb">Google Colab implementation</a> of our code to predict rheological properties in a web browser.</p> 
	<h2>Requirements</h2>
	<ul>
		<li>pymc3</li>
		<li>arviz</li>
		<li>rdkit</li>
		<li>sklearn</li>
		<li>matplotlib</li>
		<li>seaborn</li>		
	</ul>
	<h2>Installation</h2>
	<ol>
		<li>Clone this repository: <code>git clone https://github.com/cmwoodley/BART_LMWG_model.git</code></li>
		<li>Create conda environment and install the required packages: <code>conda create -n BART_LMWG python=3.8 pymc3==3.11.5 arviz rdkit matplotlib numpy=1.20 numba=0.56 pandas dill==0.3.5.1 seaborn scikit-learn ipykernel -c conda-forge</code></li>
	</ol>
	<h2>Usage</h2>
	<ol>
		<li>To build the models locally, run the training script provided in scripts/train.py: </li>
	</ol>
	<pre><code> python train.py
</code></pre>
	<ol start="2">
		<li>Serialised models are saved in models. Summary of predictions and scoring metrics are saved the reports folder. </li>
	</ol>
	<ol start="3">
		<li>An example notebook (notebooks/notebook1.ipynb) is provided with examples of predictions on a single LMWG and batches of LMWG </li>
	</ol>
	<h2>License</h2>
	<p>This project is licensed under the MIT License - see the <a href="LICENSE.md">LICENSE.md</a> file for details.</p>
	<h2>Contact</h2>
	<p>If you have any questions or suggestions, please feel free to contact me at cwoodley@liverpool.ac.uk.</p>
</body>

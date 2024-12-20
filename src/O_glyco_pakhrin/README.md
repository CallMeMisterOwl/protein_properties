# OglyPred-PLM 
Human *O*-linked Glycosylation Site Prediction Using Pretrained Protein Language Model

To guarantee proper use of our tool, please follow all the steps in the presented order.
<br>
<br>
>[!Note]
> All the programs provided in this repository are properly tested and accurate for its functionality! 
## How it Works
OglyPred-PLM is a multi-layer perceptron based approach that leverages contexualized embeddings generated
from the ProT5-XL-UniRef50 Protein Language Model (Referred here as `ProT5 PLM`) to predict Human O-linked 
glycosylation sites (`"S/T"`).

## Environment Details

OglyPred-PLM was developed in the following environment:

- Python version: 3.8.3.final.0
- Python bits: 64
- Operating System: Linux
  - OS release: 5.8.0-38-generic
- Machine architecture: x86_64
- Processor architecture: x86_64

### Python Packages and Dependencies

The model relies on the following Python packages and their specific versions:

- Pandas: 1.0.5
- NumPy: 1.18.5
- Pip: 20.1.1
- SciPy: 1.4.1
- scikit-learn: 0.23.1
- TensorFlow: 2.3.1

Please ensure that you have a compatible Python environment with these dependencies to use the model effectively.

## Create a Virtual Enviornment

To install the required dependencies for this project, you can create a virtual environment and use the provided `requirements.txt` file. Make sure your Python environment matches the specified versions above to ensure compatibility.

```
python -m venv myenv  # Create a virtual environment
source myenv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt  # Install the project dependencies
```

## Download Protein Sequences From [Uniprot.org](https://www.uniprot.org/)  

You can use our model with all human protein sequences provided by UniPort.
  - We have included a file named `Q63HQ2.fasta` as an example in this repository

Once you have downloaded the protein sequence you can send the sequence as input to the ProT5 PLM.


## Generate Contexualized Embeddings Using ProT5 PLM

### ProT5 PLM Installation 

In order to run the ProT5 PLM, you need to install the following modules:
```bash
pip install torch
pip install transformers
pip install sentencepiece
```

For more information, refer to this ["ProtTrans"](https://github.com/agemagician/ProtTrans) Repository.


Once the model is installed you can write your own code to generate the contexualized embeddings **OR**
to make this process easier for you, we have also provided a `Generate_Contexualized_Embeddings_ProT5.ipynb` file.

Here's how you can use the provided file for a single protein sequence:

You need to make changes to only two lines of the code:
```
basedir = "/project/pakhrin/salman/after_cd_hit_files"     # here you will paste the location where your .fasta file is located
name = "Q63HQ2.fasta"                                      # here you will add your protein name
```
Rest of the lines will stay the same.

Once the code is successfully run, It will generate a file named `"Q63HQ2_Prot_Trans_.csv"`.

Here's how the output file will look:

<img max-width = 100% alt="image" src="https://github.com/PakhrinLab/OglyPred-PLM/blob/main/images/ProT5_Output.png">
<br>

Rows = Length of The Protein Sequence  
Columns = 1025 (1 column to represent the protein residue + 1024 embeddings)
<br>

>[!NOTE]
>Make sure to keep all of the files being used in the same directory


## Extracting "S/T" Sites From Contexualized Embeddings

In order to extract only the "S/T" sites from the contexualized embeddings you can use the following code:

``` bash
import pandas as pd

df = pd.read_csv("Q63HQ2_Prot_Trans_.csv", header = None)                     # replace with your ProtT5 embeddings file
Header = ["Residue"]+[int(i) for i in range(1,1025)]
df.columns = Header
df_S_T_Residue_Only = df[df["Residue"].isin(["S","T"])]
df_S_T_Residue_Only.to_csv("Q63HQ2_S_T_Sites.csv", index = False)            # saves the embeddings of only S and T residues

```

Once the process is complete, you will have a .csv file containing the embeddings of "S/T" sites.

Here's an example output:

<img max-width = 100% alt="image" src="https://github.com/PakhrinLab/OglyPred-PLM/blob/main/images/Extraction_S_T_Ouput.png">
<br>

## Sending Sites Into OglyPred-PLM For Predection

Now send the `Q63HQ2_S_T_Sites.csv` from the previous step as an input to the OglyPred-PLM.

We have provided a file named `OglyPred-PLM.ipynb` and
our model`Prot_T5_my_model_O_linked_Glycosylation370381Prot_T5_Subash_Salman_Neha.h5`which should be downloaded and kept in the same directory to avoid any issues.


## Any Questions?

If you need any help don't hesitate to get in touch with Dr. Subash Chandra Pakhrin (pakhrins@uhd.edu)



# This is for Peer Review Purposes 

OglyPred-PLM: Human O-linked Glycosylation Site Prediction Using Pretrained Protein Language Model

Programs were executed using Anaconda version: 2020.07

The programs were developed in the following environment. python: 3.8.3.final.0, python-bits: 64, OS: Linux, OS-release: 5.8.0-38-generic, machine: x86_64, processor: x86_64, pandas: 1.0.5, numpy: 1.18.5, pip: 20.1.1, scipy: 1.4.1, sci-kit-learn: 0.23.1., Keras: 2.4.3, tensorflow: 2.3.1.

Please place all the following files in the same directory comparison with SPRINT-Gly.ipynb, Feature_Extraction_July_30_Sprint_Gly_Negative_Independent_Test_3376_.txt, Feature_Extraction_July_30_Sprint_Gly_Positive_Independent_Test_79_.txt (In the publicly shared google drive), Compare_with_Sprint_Gly486__103928619___.h5, and execute the Comparision with SPRINT-Gly.ipynb program to see the reported result.

Please place all the following files in the same directory Comparison with Captor Independent  O-linked Glycosylation Test Dataset.ipynb, Feature_Extraction_August_3_Captor_Independent_Negative_Test_1308_or_less.txt, Feature_Extraction_August_3_Captor_Independent_Positive_Test_341_or_less.txt  (In the publicly shared google drive), Compare_with_captor375__1438311___.h5, and execute the Comparison with Captor Independent  O-linked Glycosylation Test Dataset.ipynb program to see the reported result.

Please place all the following files in the same directory O-linked Glycosylation ProtT5 Independent Testing with undersampling result.ipynb, Feature_Extraction_O_linked_Testing_Negative_11466_Sites_less.txt, Feature_Extraction_O_linked_Testing_Positive_375_Sites_less.txt (In the publicly shared google drive), Prot_T5_my_model_O_linked_Glycosylation370381Prot_T5_Subash_Salman_Neha.h5, and execute the O-linked Glycosylation ProtT5 Independent Testing with undersampling result.ipynb program to see the reported result.

Please run the ANN_ESM2_3B_O_linked_glycosylation_Independent_Testing.py by placing the feature file in the corresponding directory to get the reported result.

To check the comparison with Alkuhlani et al. please observe the "Comparing with Alkhulani et al. Independent Test Dataset.ipynb" program.

*** For your convenience we have uploaded the ProtT5 feature extraction program (analyze_Cell_Mem_ER_Extrac_Protein (1).py) for the protein sequence from ProtT5 as well as the corresponding 1024 feature vector extraction program (Sprint Gly Independent Negative Feature Extraction.ipynb) from the ProtT5 file. We have uploaded the Q63HQ2_Prot_Trans_.csv ProtT5 feature file of protein Q63HQ2.fasta for your convenience ***


All the training and independent test data are uploaded at the following Google Drive link: https://drive.google.com/drive/folders/1MzfuMltEGF7jxKjzwLZx0gNuSDsLT7tO?usp=sharing


If you need any help don't hesitate to get in touch with Dr. Subash Chandra Pakhrin (pakhrins@uhd.edu)




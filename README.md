# Prediction of Protein Properties
Welcome to the repository for my Master’s thesis! This repository contains all the code and data used in my thesis work. This repository is still work in progress !

The thesis focuses on the prediction of three key protein properties: O/N glycosylation, B-factor, and relative surface accessibility. Specifically, it explores the potential of protein Language Model (pLM) embedding-based predictors in these areas.

### Dependencies
To set up the project, you can either:

- Use pixi: To install Pixi simply execute the following: `curl -fsSL https://pixi.sh/install.sh | bash`. For more information about shell integration, please refere to the [documentation](https://pixi.sh/dev/#autocompletion). After ensuring that pixi is installed on your system, run `pixi install` from within the root directory of the repository. This will create a `.pixi` directory which will contain all the dependencies. To remove the dependencies just delete the `.pixi` folder and optionally run `pixi clean cache`
  
  ```
  curl -fsSL https://pixi.sh/install.sh | bash
  pixi install
  ```
- Use Conda: If you prefer not to use pixi, you can alternatively install the dependencies via Conda: `conda env create -f environment.yml`. The new enviroment can be activated using `conda activate prot_prop`
  ```
  conda env create -f environment.yml
  conda activate prot_prop
  ```
### Get data
- The tsv and fasta files can be optained by downloading the provided zip file and extracting it within the root directory of this repository: `wget -i data/data_download.txt && unzip data.zip`
- Afterwards the embeddings can be generated ... (note this requires a fairly large GPU with a minumum of 12GB of VRAM)


### TODO
- [x] transition from poetry to pixi and conda requirments file 
- [x] automate data download -> maybe use pixi task
- [ ] automate embeddings creation (for train and test time)
- [ ] add prediction script/notebook
- [ ] finish README.md  

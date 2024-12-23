# Prediction of Protein Properties
Welcome to the repository for my Master’s thesis! This repository contains all the code and data used in my thesis work. This repository is still work in progress !

The thesis focuses on the prediction of three key protein properties: O/N glycosylation, B-factor, and relative surface accessibility. Specifically, it explores the potential of protein Language Model (pLM) embedding-based predictors in these areas.

### Dependencies
To set up the project, you can either:

- Use pixi: Ensure that pixi is installed on your system and run `pixi install` from within the root directory of the repository. This will create a `.pixi` directory which will contain all the dependencies. To remove the dependencies just delete the `.pixi` folder and optionally run `pixi clean cache` 
- Use Conda: If you prefer not to use pixi, you can alternatively install the dependencies via Conda


### TODO
- [ ] transition from poetry to pixi and conda requirments file 
- [ ] automate data download
- [ ] automate embeddings creation (for train and test time)
- [ ] add prediction script/notebook
- [ ] finish README.md  

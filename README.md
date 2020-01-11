# Machine Learning Engineer Nanodegree

## Installing requirements
`pip install -r requirements.txt` This command is going to install all libraries used in this project. It is extremely recommended that you run it in an virtual environment, avoiding to install this libraries in your global python. Python used in this project is *Python 3.7.4*

## Downloading the files
There is a make file that helps to download all files. There are more than 500GB of data provided by Google.
`make download-content`

## The project layout

### Capstone folder

You will find the tex file, images and references, used to generate the PDF. It is important to notice that the project report was created using latex. If you are using linux, I strongly recommended `TexStudio` to run and edit the code.

### Proposal folder

It is the same definition of above but for the proposal of the project.

### Project folder

In project folder there are all python code created to the project.

The folder `data` is ignored to the git, all downloaded data and data generated it must be saved in there.

The folder `notebooks` and `notebooks_utils` have all exploration and analytical code created to develop the project. A part of the notebooks, based on their importance, were exported as an HTML to a folder inside the notebooks folder to persist an online visualization.

The folder `model` has all the model code in fact, in here you will find the implementation of the loss function `loss.py`, the auxiliar code developed to do this `iou.py` and `smooth_l1.py`, all the ssd models created using tensorflow 2.0 + keras `ssd_*` files, the `evaluator.py` script used to evaluate the models, and two custom layers created to support the models in folder `layers`.

The folder `utils` have all python code useful for notebooks and model developing, most of them are threated as a script and have an implemantation to be run via terminal.

Last but not least, the files in the `project` root folder are the useful scripts to handle data. In it, there is also the code responsible to train the model, the `train_full.py`, in this file you will encounter all the hyper parameters chosen and the strategies created to train the desired model. To run it, you could play it with the command `python project/train_full.py <model>`. 

## Running Train
`export MODEL=vgg && PYTHONPATH=$PWD python project/train_full.py $MODEL &> trainfull-$MODEL.log`

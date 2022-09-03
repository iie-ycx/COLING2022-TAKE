cd ./knowSelect
python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train
python -u ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference
cd ..
cp -r ./knowSelect/output/TAKE_WoW/ks_pred ./dialogen/output/TAKE_WoW
cd dialogen
python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train
python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference


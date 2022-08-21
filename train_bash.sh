python ./knowSelect/TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train
python -u ./knowSelect/TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference
python ./dialogen/TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train
cp -r ./knowSelect/output/TAKE_WoW/ks_pred ./dialogen/output/TAKE_WoW
python -u ./dialogen/TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference


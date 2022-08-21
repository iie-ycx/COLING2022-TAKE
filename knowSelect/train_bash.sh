python TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train 2>&1|tee logs/log1.txt
python -u TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference

module load new gcc/4.8.2 python/3.7.1
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip3 install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:~/release_21